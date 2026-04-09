import sys
from pathlib import Path
import os
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
import json
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_utils import AgentConfig, get_llm_model
from data_model import *
from dataclasses import dataclass, field


@dataclass
class RunConfig:
    """All configurable run parameters, passed via function args (no module-level globals)."""

    kg_query_num: int = 10  # queries generated from KG per iteration
    og_query_num: int = 10  # queries generated from OG per iteration
    search_provider: str = "bing"  # bing / serper

import knowledge_graph_module
import search_module
import write_module
import outline_module
import terminal_module
from utils_module import (
    clear_usage_data_if_exists,
    update_elapsed_time,
    update_action_elapsed_time,
)  # All modules share the same lock for concurrency safety

BASE = Path(__file__).resolve().parent.parent  # .../examples/deepresearch
ENV_PATH = Path(__file__).resolve().parent / ".env"

def load_env_or_explain():
    """Load .env config file if it exists; otherwise fall back to OS env + defaults."""
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=False, verbose=True)
        logging.info(f"Loaded .env from {ENV_PATH}")
    else:
        logging.warning(f"No .env at {ENV_PATH}; using OS env + defaults")


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    cleaned = value.strip().strip("'\"")
    return cleaned if cleaned else default


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().strip("'\"").lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip().strip("'\""))
    except ValueError:
        logging.warning(f"Invalid float for {name}: {value}; using default {default}")
        return default


def _atomic_write_text(file_path: str, content: str, encoding: str = "utf-8") -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "w", encoding=encoding) as f:
        f.write(content)
    os.replace(tmp_path, file_path)


# ---------------------------------------------------------------------------
# LLM model factory
# ---------------------------------------------------------------------------

def _build_base_url() -> str:
    """Resolve and normalize LLM base URL from env vars."""
    raw = env_str("LLM_BASE_URL") or env_str("AZURE_OPENAI_ENDPOINT")
    base_url = raw if raw else "https://api.openai.com/v1"
    if base_url.rstrip("/").lower().endswith("/openai"):
        base_url = base_url.rstrip("/")[: -len("/openai")]
    return base_url.rstrip("/")


def _make_agent_config(model_name: str, base_url: str) -> AgentConfig:
    """Create an AgentConfig from env vars with the given model name."""
    api_key = env_str("LLM_API_KEY")
    if api_key == "":
        api_key = None
    return AgentConfig(
        llm_provider=env_str("LLM_PROVIDER", "azure_openai"),
        llm_model_name=model_name,
        llm_api_key=api_key,
        llm_base_url=base_url,
        llm_temperature=env_float("LLM_TEMPERATURE", 0.0),
        llm_use_aad=env_bool("AZURE_OPENAI_USE_AAD", False),
        azure_client_id=env_str("AZURE_CLIENT_ID"),
        azure_client_secret=env_str("AZURE_CLIENT_SECRET"),
        azure_endpoint=base_url,
    )


def create_llm_model(model_name: str):
    """Create a single LLM model instance."""
    base_url = _build_base_url()
    conf = _make_agent_config(model_name, base_url)
    return get_llm_model(conf=conf)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


@dataclass
class DatasetBundle:
    """Result of loading an evaluation dataset."""

    selected_ids: List[int]
    queries: List[str]
    idx_to_language: Dict[int, str] = field(default_factory=dict)
    idx_to_blocked_urls: Dict[int, set] = field(default_factory=dict)


def load_eval_dataset(
    eval_dataset_name: str,
    id_range: List[int],
    report_dir: str,
    language: str,
) -> DatasetBundle:
    """Load an evaluation dataset and return filtered IDs + queries.

    Args:
        eval_dataset_name: Name of the dataset to load (subfolder under eval_dataset/).
        id_range: [start, end] inclusive range of IDs to process.
        report_dir: Directory where reports are saved (for skip-existing logic).
        language: Default language for datasets without per-query language.

    Returns:
        DatasetBundle with selected_ids, queries, and per-ID metadata.
    """
    selected_ids = list(range(id_range[0], id_range[1] + 1))

    # Skip already-completed reports
    existing_ids: set = set()
    if os.path.isdir(report_dir):
        existing_ids = {
            int(f.split(".md")[0])
            for f in os.listdir(report_dir)
            if f.endswith(".md") and f.split(".md")[0].isdigit()
        }
    selected_ids = [i for i in selected_ids if i not in existing_ids]

    bundle = _load_query_jsonl(eval_dataset_name, selected_ids)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Dataset: {eval_dataset_name}")
    print(f"  Existing reports: {len(existing_ids)}")
    print(f"  To process: {len(bundle.selected_ids)}")
    print(f"  IDs: {bundle.selected_ids}")
    print(f"{'='*80}\n")

    return bundle


def get_language_for_report(
    report_id: int,
    eval_dataset_name: str,
    idx_to_language: Dict[int, str],
    default_language: str,
) -> str:
    """Return the language for a given report ID based on dataset conventions."""
    lang_code = idx_to_language.get(report_id)
    if lang_code:
        return "Chinese" if lang_code == "zh" else "English"
    return default_language


# --- Dataset loader ---


def _load_query_jsonl(eval_dataset_name: str, selected_ids) -> DatasetBundle:
    """Generic loader: reads eval_dataset/{name}/query.jsonl with {id, prompt} per line."""
    jsonl_path = BASE.parent / "eval_dataset" / eval_dataset_name / "query.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {jsonl_path}\n"
            f"Please place a query.jsonl file in eval_dataset/{eval_dataset_name}/. "
            f"See eval_dataset/README.md for format details."
        )
    with open(jsonl_path, mode="r", encoding="utf-8") as f:
        all_queries = [json.loads(line) for line in f if line.strip()]
    query_dict = {q.get("id", -100): q.get("prompt") for q in all_queries}

    valid_ids, queries = [], []
    for i in selected_ids:
        if i in query_dict:
            valid_ids.append(i)
            queries.append(query_dict[i])
        else:
            print(f"[Warning] ID {i} not found in {jsonl_path.name}, skipping")

    return DatasetBundle(selected_ids=valid_ids, queries=queries)

def process_single_report_og_kg(
    report_id: int,
    root_query: str,
    report_dir: str,
    product_dir: str,
    language: str,
    llm_model,
    usage_file: str,
    max_iter: int,
    cfg: RunConfig,
    disable_early_stopping: bool = False,
    blocked_urls: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Process a single report ID through the full OG_KG pipeline.

    Args:
        report_id: Report ID.
        root_query: Root query string.
        report_dir: Report output directory.
        product_dir: Intermediate artifacts directory.
        language: Report language.
        llm_model: LLM model object.
        usage_file: Path to the usage tracking file.

    Returns:
        Dict with success, report_id, error, etc.
    """
    result = {"success": False, "report_id": report_id, "error": None}

    # Concurrency guard: re-check if report already exists before starting
    report_path = os.path.join(report_dir, f"{report_id}.md")
    if os.path.exists(report_path):
        print(f"[Skip] Report {report_id} already exists, skipping")
        result["success"] = True
        result["skipped"] = True
        return result

    # Clear any residual usage data from a previous abnormal termination
    clear_usage_data_if_exists(report_id, usage_file)

    start_time = time.time()

    print(f"\n{'='*80}")
    print(f"[Start] Report ID: {report_id}")
    print(f"{'='*80}\n")

    try:

        def _dedup_preserve_order(items: List[str]) -> List[str]:
            # Python 3.7+ dicts preserve insertion order; used for stable dedup
            return list(dict.fromkeys([x for x in items if x]))

        # ========================================================================
        # Optional: Neo4j KG persistence (does not affect in-memory KG logic)
        # ========================================================================
        # The main flow still uses the in-memory KnowledgeGraph for all modules.
        # When NEO4J_KG_ENABLED is on, snapshots are synced to Neo4j after each
        # build/update for downstream GDS clustering, etc.
        neo4j_kg_enabled = env_bool("NEO4J_KG_ENABLED", default=False)
        # When enabled, automatically runs GDS Leiden after each Neo4j write and
        # writes results back to KGNode.cluster_id.
        # If the target Neo4j/Aura lacks GDS, the exception is caught and
        # the system degrades to write-only (no clustering).
        neo4j_gds_leiden_enabled = env_bool("NEO4J_GDS_LEIDEN_ENABLED", default=True)
        neo4j_store: Optional["Neo4jKGStore"] = None
        if neo4j_kg_enabled:
            from neo4j_kg_store import Neo4jKGStore

            neo4j_store = Neo4jKGStore.from_env(report_id=report_id)

        def _sync_clusters_back_to_memory(
            kg: KnowledgeGraph, node_id_to_cluster: Dict[int, str]
        ) -> None:
            if not node_id_to_cluster:
                return
            for node in kg.knowledge_nodes:
                if node.id in node_id_to_cluster:
                    node.cluster_id = str(node_id_to_cluster[node.id])

        # ========================================================================
        # Initialize research state
        # ========================================================================
        knowledge_graph = KnowledgeGraph()
        history_search_queries = (
            []
        )  # Only tracks actually-executed queries, to avoid duplicate searches
        visited_urls = set()  # Dedup set to avoid re-fetching the same page
        if blocked_urls:
            visited_urls.update(blocked_urls)

        # ========================================================================
        # (1) Create Iterate - Initial iteration (iter0)
        # ========================================================================
        # Step 1: Generate research outline from root query (root_query -> OG)
        ## root_query -> OG
        create_outline_start_time = time.time()
        outline = outline_module.create_outline(
            root_query,
            llm_model=llm_model,
            language=language,
            report_id=report_id,
            usage_file=usage_file,
        )
        create_outline_end_time = time.time()
        update_action_elapsed_time(
            "create_outline",
            report_id,
            usage_file,
            elapsed_time=create_outline_end_time - create_outline_start_time,
        )
        # Save initial outline for debugging
        with open(
            f"{product_dir}/outline_case{report_id}_iter0.txt", "w", encoding="utf-8"
        ) as f:
            f.write(outline)

        # Step 2: Generate search queries from outline (OG -> SQ)
        ## OG -> SQ
        generate_search_queries_start_time = time.time()
        outline_search_queries_iter0 = outline_module.generate_search_queries(
            outline,
            history_search_queries=history_search_queries,
            llm_model=llm_model,
            language=language,
            report_id=report_id,
            usage_file=usage_file,
            is_use_history_search_queries=False,
            query_num=cfg.og_query_num,
        )
        generate_search_queries_end_time = time.time()
        update_action_elapsed_time(
            "generate_search_queries",
            report_id,
            usage_file,
            elapsed_time=generate_search_queries_end_time
            - generate_search_queries_start_time,
        )
        outline_search_queries_iter0 = _dedup_preserve_order(
            outline_search_queries_iter0
        )

        search_queries = outline_search_queries_iter0  # iter0: search via OG first

        # Step 3: Execute search and collect evidence (SQ -> SR)
        ## SQ -> SR
        evidence_nodes_batch = []  # List[List[EvidenceNode]]

        search_iter0_start_time = time.time()
        for search_query in search_queries:
            new_evidences_this_query = (
                search_module.search_with_filtering_visited_urls(
                    query=search_query,
                    root_query=root_query,
                    llm_model=llm_model,
                    language=language,
                    visited_urls=visited_urls,
                    report_id=report_id,
                    usage_file=usage_file,
                    search_provider=cfg.search_provider,
                )
            )
            # Convert evidence dicts to KG evidence nodes
            evidence_nodes_this_query = []
            for new_evidence in new_evidences_this_query:
                new_evidence_node = knowledge_graph.add_evidence_node(
                    source_title=new_evidence["source_title"],
                    source_url=new_evidence["source_url"],
                    content=new_evidence["content"],
                )
                visited_urls.add(new_evidence["source_url"])
                evidence_nodes_this_query.append(new_evidence_node)
            evidence_nodes_batch.append(
                evidence_nodes_this_query
            )
            # Append search results to file
            with open(
                f"{product_dir}/search_results_case{report_id}_iter0.txt",
                "a+",
                encoding="utf-8",
            ) as f:
                f.write(f"# Search Query: {search_query}\n")
                for ev in new_evidences_this_query:
                    f.write(
                        f"# Evidence:\ntitle: {ev['source_title']}\nurl: {ev['source_url']}\ncontent: {ev['content']}\n\n"
                    )
        search_iter0_end_time = time.time()
        update_action_elapsed_time(
            "search_iter0",
            report_id,
            usage_file,
            elapsed_time=search_iter0_end_time - search_iter0_start_time,
        )
        history_search_queries.extend(
            outline_search_queries_iter0
        )
        # Save executed search queries after iter0
        with open(
            f"{product_dir}/history_search_queries_case{report_id}_iter0.txt",
            "w",
            encoding="utf-8",
        ) as f:
            for idx, query in enumerate(history_search_queries, 1):
                f.write(f"{idx}. {query}\n")
        # Step 4: Build knowledge graph from evidence (SR -> KG)
        ## SR -> KG
        create_knowledge_graph_start_time = time.time()
        knowledge_graph = knowledge_graph_module.create_knowledge_graph(
            root_query=root_query,
            knowledge_graph=knowledge_graph,
            evidence_nodes_batch=evidence_nodes_batch,
            search_queries=search_queries,
            llm_model=llm_model,
            report_id=report_id,
            usage_file=usage_file,
        )
        create_knowledge_graph_end_time = time.time()
        update_action_elapsed_time(
            "create_knowledge_graph",
            report_id,
            usage_file,
            elapsed_time=create_knowledge_graph_end_time
            - create_knowledge_graph_start_time,
        )

        with open(
            os.path.join(product_dir, f"knowledge_graph_case{report_id}_iter0.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(knowledge_graph.model_dump_json())

        # Sync KG to Neo4j (iter0)
        if neo4j_store is not None:
            neo4j_store.replace_graph(knowledge_graph, iter_idx=0)
            if neo4j_gds_leiden_enabled:
                try:
                    node_id_to_cluster = neo4j_store.run_gds_leiden_writeback(
                        write_property="cluster_id"
                    )
                    _sync_clusters_back_to_memory(knowledge_graph, node_id_to_cluster)
                except Exception as e:
                    print(f"[Neo4j][GDS] Leiden clustering failed, falling back to write-only: {e}")

        # Step 5: Generate explore queries from KG (KG -> EQ)
        generate_explore_queries_start_time = time.time()
        this_time_search_chains, kg_search_queries, chains, visited_edges = (
            knowledge_graph_module.generate_search_chains_and_search_queries(
                root_query=root_query,
                outline=outline,
                knowledge_graph=knowledge_graph,
                llm_model=llm_model,
                visited_edges=set(),  # Empty at initial iteration
                language=language,
                report_id=report_id,
                usage_file=usage_file,
                is_use_history_search_queries=True,
                history_search_queries=history_search_queries,
                kg_query_num=cfg.kg_query_num,
            )
        )
        generate_explore_queries_end_time = time.time()
        update_action_elapsed_time(
            "generate_explore_queries",
            report_id,
            usage_file,
            elapsed_time=generate_explore_queries_end_time
            - generate_explore_queries_start_time,
        )
        # OG_KG queries = KG-generated + OG-generated (fair comparison with OG_only)
        # iter0 already executed outline_search_queries_iter0, so remove those
        # from planned queries to avoid duplicate execution in iter1.
        # Note: these are "planned" queries for the next round, not yet executed --
        # do NOT add them to history_search_queries yet.
        _planned_search_queries = _dedup_preserve_order(
            (kg_search_queries or []) + outline_search_queries_iter0
        )
        _executed_set = set(history_search_queries or [])
        kg_search_queries_after_filter = [
            q for q in _planned_search_queries if q not in _executed_set
        ]
        search_queries = kg_search_queries_after_filter


        with open(
            os.path.join(product_dir, f"explore_chains_case{report_id}_iter0.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            for chain in this_time_search_chains:
                f.write(f"{chain}\n")

        with open(
            os.path.join(product_dir, f"sq_case{report_id}_iter0.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            # f.write("\n# OG-generated queries (already executed):\n")
            for query in outline_search_queries_iter0:
                f.write(f"{query}\n")
            # f.write("\n# Filtered KG queries (for next iteration):\n")
            for query in kg_search_queries_after_filter:
                f.write(f"{query}\n")

        # ========================================================================
        # (2) Update Iterate - Iterative refinement
        # ========================================================================
        for iter in range(max_iter):
            # Step 1: Execute search with queries from previous iteration (SQ -> SR)
            evidence_nodes_batch = []  # List[List[EvidenceNode]]
            search_iter_start_time = time.time()
            for search_query in search_queries:
                new_evidences_this_query = (
                    search_module.search_with_filtering_visited_urls(
                        query=search_query,
                        root_query=root_query,
                        llm_model=llm_model,
                        language=language,
                        visited_urls=visited_urls,
                        report_id=report_id,
                        usage_file=usage_file,
                        search_provider=cfg.search_provider,
                    )
                )
                # Convert evidence to KG nodes
                evidence_nodes_this_query = []
                for new_evidence in new_evidences_this_query:
                    new_evidence_node = knowledge_graph.add_evidence_node(
                        source_title=new_evidence["source_title"],
                        source_url=new_evidence["source_url"],
                        content=new_evidence["content"],
                    )
                    # Record visited URL if non-empty
                    if new_evidence.get("source_url", "") != "":
                        visited_urls.add(new_evidence["source_url"])
                    evidence_nodes_this_query.append(new_evidence_node)
                evidence_nodes_batch.append(evidence_nodes_this_query)
                # Append search results to file
                with open(
                    f"{product_dir}/search_results_case{report_id}_iter{iter+1}.txt",
                    "a+",
                    encoding="utf-8",
                ) as f:
                    f.write(f"# Search Query: {search_query}\n")
                    for ev in new_evidences_this_query:
                        f.write(
                            f"# Evidence:\ntitle: {ev['source_title']}\nurl: {ev['source_url']}\ncontent: {ev['content']}\n\n"
                        )
            search_iter_end_time = time.time()
            if search_queries:  # Only record time if searches were actually executed
                update_action_elapsed_time(
                    "search_iter",
                    report_id,
                    usage_file,
                    elapsed_time=search_iter_end_time - search_iter_start_time,
                )
            # This iteration's search_queries are now executed: record in history
            history_search_queries = _dedup_preserve_order(
                history_search_queries + (search_queries or [])
            )
            # Save all executed search queries for tracking and review
            with open(
                f"{product_dir}/history_search_queries_case{report_id}_iter{iter+1}.txt",
                "w",
                encoding="utf-8",
            ) as f:
                for idx, query in enumerate(history_search_queries, 1):
                    f.write(f"{idx}. {query}\n")

            # Step 2: Update knowledge graph (SR -> KG)
            # TODO 3. Will context be too long when LLM reads KG + new_evidence to generate Nodes/Edges?
            update_knowledge_graph_start_time = time.time()
            knowledge_graph = knowledge_graph_module.update_knowledge_graph(
                root_query,
                knowledge_graph,
                evidence_nodes_batch,
                search_queries,
                llm_model=llm_model,
                report_id=report_id,
                usage_file=usage_file,
            )
            update_knowledge_graph_end_time = time.time()
            update_action_elapsed_time(
                "update_knowledge_graph",
                report_id,
                usage_file,
                elapsed_time=update_knowledge_graph_end_time
                - update_knowledge_graph_start_time,
            )

            with open(
                os.path.join(
                    product_dir, f"knowledge_graph_case{report_id}_iter{iter+1}.txt"
                ),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(knowledge_graph.model_dump_json())

            # Sync KG to Neo4j (iter1..)
            if neo4j_store is not None:
                neo4j_store.replace_graph(knowledge_graph, iter_idx=iter + 1)
                if neo4j_gds_leiden_enabled:
                    try:
                        node_id_to_cluster = neo4j_store.run_gds_leiden_writeback(
                            write_property="cluster_id"
                        )
                        _sync_clusters_back_to_memory(
                            knowledge_graph, node_id_to_cluster
                        )
                    except Exception as e:
                        print(f"[Neo4j][GDS] Leiden clustering failed, falling back to write-only: {e}")

            # Step 3: Update outline (KG -> OG)
            update_outline_start_time = time.time()
            outline = outline_module.update_outline_by_kg(
                root_query,
                outline,
                evidences=[en for batch in evidence_nodes_batch for en in batch],
                knowledge_graph=knowledge_graph,
                llm_model=llm_model,
                language=language,
                report_id=report_id,
                usage_file=usage_file,
            )
            update_outline_end_time = time.time()
            update_action_elapsed_time(
                "update_outline_by_kg",
                report_id,
                usage_file,
                elapsed_time=update_outline_end_time - update_outline_start_time,
            )

            with open(
                f"{product_dir}/outline_case{report_id}_iter{iter+1}.txt",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(outline)

            # Step 4: Generate new search queries (KG -> SQ_KG, OG -> SQ_OG)
            generate_explore_queries_iter_start_time = time.time()
            this_time_search_chains, kg_search_queries, chains, visited_edges = (
                knowledge_graph_module.generate_search_chains_and_search_queries(
                    root_query=root_query,
                    outline=outline,
                    knowledge_graph=knowledge_graph,
                    llm_model=llm_model,
                    visited_edges=visited_edges,
                    language=language,
                    report_id=report_id,
                    usage_file=usage_file,
                    is_use_history_search_queries=True,
                    history_search_queries=history_search_queries,
                    kg_query_num=cfg.kg_query_num,
                )
            )
            generate_explore_queries_iter_end_time = time.time()
            update_action_elapsed_time(
                "generate_explore_queries",
                report_id,
                usage_file,
                elapsed_time=generate_explore_queries_iter_end_time
                - generate_explore_queries_iter_start_time,
            )

            # Also add OG queries each iteration (retain OG_only capability + KG incremental)
            generate_search_queries_iter_start_time = time.time()
            og_search_queries = outline_module.generate_search_queries(
                outline,
                history_search_queries=history_search_queries,
                llm_model=llm_model,
                language=language,
                report_id=report_id,
                usage_file=usage_file,
                is_use_history_search_queries=True,
                pending_search_queries=kg_search_queries,
                query_num=cfg.og_query_num,
            )
            generate_search_queries_iter_end_time = time.time()
            update_action_elapsed_time(
                "generate_search_queries",
                report_id,
                usage_file,
                elapsed_time=generate_search_queries_iter_end_time
                - generate_search_queries_iter_start_time,
            )
            # Build next-round planned queries, removing already-executed ones (LLM may repeat history queries)
            _planned_search_queries = _dedup_preserve_order(
                (kg_search_queries or []) + (og_search_queries or [])
            )
            _executed_set = set(history_search_queries or [])
            search_queries = [
                q for q in _planned_search_queries if q not in _executed_set
            ]


            with open(
                f"{product_dir}/sq_case{report_id}_iter{iter+1}.txt",
                "w",
                encoding="utf-8",
            ) as f:
                for query in search_queries:
                    f.write(query + "\n")

            with open(
                os.path.join(
                    product_dir, f"explore_chains_case{report_id}_iter{iter+1}.txt"
                ),
                "w",
                encoding="utf-8",
            ) as f:
                for chain in this_time_search_chains:
                    f.write(f"{chain}\n")


            with open(
                os.path.join(product_dir, f"chains_case{report_id}_iter{iter+1}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                for c in chains:
                    f.write(f"{c.model_dump_json()}\n")

            # ========================================================================
            # (3) Judge Termination
            # ========================================================================
            judge_terminal_start_time = time.time()
            is_terminal = terminal_module.judge_terminal_by_outline(
                root_query=root_query,
                outline=outline,
                # evidence_nodes_list=evidence_nodes_list,  # Optional param (commented out)
                llm_model=llm_model,
                report_id=report_id,
                usage_file=usage_file,
                disable_early_stopping=disable_early_stopping,
            )
            judge_terminal_end_time = time.time()
            update_action_elapsed_time(
                "judge_terminal",
                report_id,
                usage_file,
                elapsed_time=judge_terminal_end_time - judge_terminal_start_time,
            )

            if is_terminal:
                break
        # ========================================================================
        # (4) Write Report
        # ========================================================================
        try:
            write_report_start_time = time.time()
            report = write_module.write_report_by_outline_kg(
                root_query=root_query,
                outline=outline,
                knowledge_graph=knowledge_graph,
                llm_model=llm_model,
                language=language,
                report_id=report_id,
                usage_file=usage_file,
                is_use_kg_to_write_report=False,  # Disable passing KG to Write
            )
            write_report_end_time = time.time()
            update_action_elapsed_time(
                "write_report",
                report_id,
                usage_file,
                elapsed_time=write_report_end_time - write_report_start_time,
            )
            print(report)

            # Concurrency guard: re-check before saving to prevent write conflicts
            report_path = os.path.join(report_dir, f"{report_id}.md")
            if not os.path.exists(report_path):
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"[Done] Report {report_id} generated and saved")
            else:
                print(f"[Warning] Report {report_id} already exists at save time, possibly processed by another worker")


            evidence_path = os.path.join(
                product_dir, f"evidence_node_list_{report_id}.txt"
            )
            with open(evidence_path, "w", encoding="utf-8") as f:
                for en in knowledge_graph.evidence_nodes:
                    f.write(
                        f"ID: id_{en.id}\nTitle: {en.source_title}\nURL: {en.source_url}\nContent: {en.content}\n\n"
                    )

            result["success"] = True
            elapsed_time = time.time() - start_time
            update_elapsed_time(report_id, usage_file, elapsed_time)
            print(f"[Timing] Report {report_id} took {elapsed_time:.2f}s")
        except Exception as e:
            # Save error info and intermediate results for debugging
            import traceback

            with open(
                os.path.join(report_dir, f"error_report_outline_{report_id}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(outline)
            with open(
                os.path.join(report_dir, f"error_report_kg_{report_id}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(knowledge_graph.model_dump_json())
            with open(
                os.path.join(report_dir, f"error_report_{report_id}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(
                    f"Error in writing report for {report_id}. {e}\n{traceback.format_exc()}"
                )
            result["error"] = f"Report generation failed: {str(e)}"
            raise
    except Exception as e:
        print(f"\n[Error] Exception while processing report {report_id}: {e}")
        import traceback

        error_msg = f"Error processing report {report_id}: {e}\n\nTraceback:\n{traceback.format_exc()}"
        with open(
            os.path.join(report_dir, f"error_report_{report_id}.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(error_msg)
        result["error"] = str(e)
        print(f"[Continue] Report {report_id} failed, error logged\n")

    return result


if __name__ == "__main__":
    # ---- CLI argument parsing ----
    import argparse

    parser = argparse.ArgumentParser(description="DeepResearch baseline runner")
    parser.add_argument(
        "--models", nargs="+", default=["gpt-4.1-20250414-2"], help="LLM model name(s)"
    )
    parser.add_argument(
        "--version", default="v1", help="Version stamp for output dirs"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["example"],
        help="Eval dataset name(s) — subfolder under eval_dataset/ containing query.jsonl",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Disable early stopping (default: early stopping enabled)",
    )
    parser.add_argument(
        "--kg-query-num",
        type=int,
        default=10,
        help="Query count from KG per iter (default: 10)",
    )
    parser.add_argument(
        "--og-query-num",
        type=int,
        default=10,
        help="Query count from OG per iter (default: 10)",
    )
    parser.add_argument(
        "--id-range",
        type=int,
        nargs=2,
        default=[1, 1],
        metavar=("START", "END"),
        help="Query ID range [START, END], e.g. --id-range 1 20 means ID 1..20 (default: 1 1)",
    )
    parser.add_argument(
        "--search-provider",
        default="bing",
        choices=["bing", "serper"],
        help="Search backend to use (default: bing)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Max number of concurrent threads for batch processing (default: 5)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Max number of research iterations per query (default: 5)",
    )
    parser.add_argument(
        "--language",
        default="English",
        choices=["English", "Chinese"],
        help="Report language (default: English)",
    )
    args = parser.parse_args()

    cfg = RunConfig(
        kg_query_num=args.kg_query_num,
        og_query_num=args.og_query_num,
        search_provider=args.search_provider,
    )

    DISABLE_EARLY_STOPPING = args.disable_early_stopping
    llm_model_name_list = args.models
    version_stamp = args.version
    eval_dataset_name_list = args.datasets

    effective_version_stamp = (
        version_stamp + "_NOES" if DISABLE_EARLY_STOPPING else version_stamp
    )

    print(
        f"[CLI] models={llm_model_name_list}, version={version_stamp}"
    )
    print(
        f"[CLI] datasets={eval_dataset_name_list}, early_stopping={not DISABLE_EARLY_STOPPING}, id_range={args.id_range}"
    )
    print(f"[CLI] {cfg}")

    # Initialization
    import log_tee

    log_name = f"./logs/run-{time.strftime('%Y%m%d%H%M%S')}.log"
    os.makedirs("./logs", exist_ok=True)
    sys.stdout = log_tee.Tee(log_name)
    sys.stderr = sys.stdout
    load_env_or_explain()
    for llm_model_name in llm_model_name_list:
        for eval_dataset_name in eval_dataset_name_list:
            max_iter = args.max_iter
            report_dir = f"og_kg_reports_{llm_model_name}_{effective_version_stamp}"
            product_dir = f"og_kg_products_{llm_model_name}_{effective_version_stamp}"
            usage_file = f"og_kg_usage_{llm_model_name}_{effective_version_stamp}"

            language = args.language

            save_dir = str(BASE.parent / "exp_saves")
            os.makedirs(save_dir, exist_ok=True)

            report_dir = os.path.join(save_dir, report_dir)
            report_dir = report_dir + f"_MAX_{max_iter}_{eval_dataset_name}"
            product_dir = os.path.join(save_dir, product_dir)
            product_dir = product_dir + f"_MAX_{max_iter}_{eval_dataset_name}"

            usage_file = os.path.join(save_dir, usage_file + ".json")
            if usage_file.endswith(".json"):
                usage_file = (
                    usage_file[:-5] + f"_MAX_{max_iter}_{eval_dataset_name}.json"
                )

            if not os.path.exists(usage_file):
                with open(usage_file, "w", encoding="utf-8") as f:
                    json.dump({}, f)

            os.makedirs(report_dir, exist_ok=True)
            os.makedirs(product_dir, exist_ok=True)

            # Initialize LLM model
            llm_model = create_llm_model(llm_model_name)

            # Load dataset
            ds = load_eval_dataset(
                eval_dataset_name=eval_dataset_name,
                id_range=args.id_range,
                report_dir=report_dir,
                language=language,
            )
            selected_ids = ds.selected_ids
            queries = ds.queries

            if not selected_ids:
                print("All IDs already processed, skipping.")
                continue

            # Prepare task list
            tasks = list(zip(selected_ids, queries))

            print(f"\n{'='*80}")
            print(f"Starting thread pool, max concurrency: {args.max_concurrency}")
            print(f"{'='*80}\n")

            completed_count = 0
            failed_count = 0
            skipped_count = 0

            with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
                future_to_report = {}
                for idx, (report_id, root_query) in enumerate(tasks):
                    # Dynamically set language based on report_id
                    task_language = get_language_for_report(
                        report_id, eval_dataset_name, ds.idx_to_language, language
                    )

                    # Get blocked URLs for this task (bench2 only)
                    task_blocked_urls = ds.idx_to_blocked_urls.get(report_id)

                    future = executor.submit(
                        process_single_report_og_kg,
                        report_id,
                        root_query,
                        report_dir,
                        product_dir,
                        task_language,
                        llm_model,
                        usage_file,
                        max_iter,
                        cfg,
                        DISABLE_EARLY_STOPPING,
                        task_blocked_urls,
                    )
                    future_to_report[future] = report_id

                with tqdm(total=len(tasks), desc="Processing reports", unit="report") as pbar:
                    for future in as_completed(future_to_report):
                        report_id = future_to_report[future]
                        try:
                            result = future.result()
                            if result.get("skipped", False):
                                skipped_count += 1
                                pbar.set_postfix(
                                    {
                                        "status": f"skipped {report_id}",
                                        "ok": completed_count,
                                        "skip": skipped_count,
                                        "fail": failed_count,
                                    }
                                )
                            elif result.get("success", False):
                                completed_count += 1
                                pbar.set_postfix(
                                    {
                                        "status": f"done {report_id}",
                                        "ok": completed_count,
                                        "skip": skipped_count,
                                        "fail": failed_count,
                                    }
                                )
                            else:
                                failed_count += 1
                                pbar.set_postfix(
                                    {
                                        "status": f"failed {report_id}",
                                        "ok": completed_count,
                                        "skip": skipped_count,
                                        "fail": failed_count,
                                    }
                                )
                        except Exception as e:
                            print(
                                f"[Exception] Uncaught exception for report {report_id}: {e}"
                            )
                            failed_count += 1
                            pbar.set_postfix(
                                {
                                    "status": f"exception {report_id}",
                                    "ok": completed_count,
                                    "skip": skipped_count,
                                    "fail": failed_count,
                                }
                            )
                        finally:
                            pbar.update(1)

            print(f"\n{'='*80}")
            print(f"[Complete] All pending reports processed")
            print(f"  - Succeeded: {completed_count}")
            print(f"  - Skipped (already exist): {skipped_count}")
            print(f"  - Failed: {failed_count}")
            print(f"  - Total: {len(tasks)}")
            print(f"{'='*80}\n")
