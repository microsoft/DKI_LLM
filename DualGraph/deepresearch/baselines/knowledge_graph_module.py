import sys
from llm_utils import call_llm_model, get_llm_model, embed_texts
from data_model import *
from typing import List, Set, Tuple, Dict, Optional, Any
from collections import defaultdict
import yaml
from utils_module import safe_json_loads, update_llm_usage
from string import Template
import json
import os
import time
import logging
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROMPT_LIB_DIR = Path(__file__).resolve().parent / "prompt_lib"
# Add deepresearch/utils to sys.path so cloudgpt_aoai.py can be imported
UTILS_DIR = BASE / "utils"
if UTILS_DIR.exists():
    _utils_dir_str = str(UTILS_DIR)
    if _utils_dir_str not in sys.path:
        sys.path.append(_utils_dir_str)


def _semantic_cluster_knowledge_nodes_hdbscan(
    knowledge_graph: "KnowledgeGraph",
    embedding_model: str = "text-embedding-3-large",
    min_cluster_size: int = 3,
    min_samples: Optional[int] = None,
) -> Dict:
    """
    用 HDBSCAN 对非核心实体节点（concept/attribute 类节点）做语义聚类，不需要指定簇数。

    输出格式直接对齐 data_model.KnowledgeGraph.apply_clustering_results() 的输入：
    {
      "clusters": [
        {
          "cluster_id": "...",
          "representative_concept": "...",
          "source_node_ids": ["n2","n3",...],
          "similarity_justification": "..."
        }
      ]
    }
    """
    # Allow disabling via env var (avoids blocking when CloudGPT auth is not configured)
    if os.getenv("DEEPRESEARCH_ENABLE_SEMANTIC_CLUSTERING", "").strip().lower() in {
        "false",
        "False",
    }:
        return {"clusters": []}

    # Only cluster non-core-entity nodes
    concept_nodes = [
        n
        for n in knowledge_graph.knowledge_nodes
        if not getattr(n, "is_core_entity", False)
    ]
    if len(concept_nodes) < 2:
        return {"clusters": []}

    # Reuse cached embeddings; only compute embeddings for new nodes
    texts = [n.knowledge for n in concept_nodes]
    vectors = []
    texts_to_embed = []
    indices_to_embed = []

    if not hasattr(knowledge_graph, "embedding_cache"):
        knowledge_graph.embedding_cache = {}

    for idx, text in enumerate(texts):
        cache_key = (text, embedding_model)
        if cache_key in knowledge_graph.embedding_cache:
            vectors.append(knowledge_graph.embedding_cache[cache_key])
        else:
            vectors.append(None)
            texts_to_embed.append(text)
            indices_to_embed.append(idx)

    # Only call embedding API for nodes not in cache
    cached_count = len(concept_nodes) - len(texts_to_embed)
    if cached_count > 0:
        logging.info(
            f"[semantic_cluster] Embedding cache hit: {cached_count}/{len(concept_nodes)} nodes from cache, "
            f"recomputing: {len(texts_to_embed)} nodes"
        )

    if texts_to_embed:
        new_vectors = embed_texts(texts=texts_to_embed, embedding_model=embedding_model)
        if len(new_vectors) != len(texts_to_embed):
            logging.warning(
                f"[semantic_cluster] Embedding count mismatch: expected={len(texts_to_embed)} actual={len(new_vectors)}, skipping semantic clustering"
            )
            return {"clusters": []}

        # Fill computed embeddings into vectors and store in cache
        for i, (idx, new_vec) in enumerate(zip(indices_to_embed, new_vectors)):
            vectors[idx] = new_vec
            cache_key = (texts[idx], embedding_model)
            knowledge_graph.embedding_cache[cache_key] = new_vec

        logging.info(
            f"[semantic_cluster] Computed and cached {len(new_vectors)} new node embeddings"
        )

    if any(v is None for v in vectors):
        logging.warning(f"[semantic_cluster] Some node embeddings are missing, skipping semantic clustering")
        return {"clusters": []}

    if len(vectors) != len(concept_nodes):
        logging.warning(
            f"[semantic_cluster] Embedding count mismatch: nodes={len(concept_nodes)} vectors={len(vectors)}, skipping semantic clustering"
        )
        return {"clusters": []}

    # HDBSCAN（用 cosine distance，更贴近语义）
    try:
        import numpy as np
        import hdbscan
        from sklearn.metrics.pairwise import cosine_distances
    except Exception as e:
        logging.warning(
            f"[semantic_cluster] Missing dependencies (hdbscan/numpy/sklearn): {e}, skipping semantic clustering"
        )
        return {"clusters": []}

    X = np.asarray(vectors, dtype=np.float32)
    # Precompute cosine distance matrix (O(n^2); concept nodes are typically few)
    D = cosine_distances(X)
    D = D.astype(np.float64)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
    )
    labels = clusterer.fit_predict(D)

    # labels: -1 = noise
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label is None or int(label) < 0:
            continue
        label_to_indices[int(label)].append(idx)

    clusters = []
    for label, idxs in sorted(label_to_indices.items(), key=lambda kv: kv[0]):
        if len(idxs) < 2:
            continue

        # Select representative: use medoid (node with smallest average distance within cluster)
        sub = D[np.ix_(idxs, idxs)]
        medoid_pos = int(np.argmin(sub.sum(axis=1)))
        rep_idx = idxs[medoid_pos]

        rep_text = concept_nodes[rep_idx].knowledge
        source_node_ids = [f"n{concept_nodes[i].id}" for i in idxs]

        clusters.append(
            {
                "cluster_id": f"hdbscan_{label}",
                "representative_concept": rep_text,
                "source_node_ids": source_node_ids,
                "similarity_justification": f"HDBSCAN semantic cluster (label={label}, size={len(idxs)})",
            }
        )

    return {"clusters": clusters}


def _leiden_community_detection(
    knowledge_graph: "KnowledgeGraph",
    resolution_parameter: float = 1.0,
    n_iterations: int = 2,
    random_seed: Optional[int] = None,
) -> Dict[int, str]:
    """
    Use igraph + leidenalg for Leiden community detection on the knowledge graph.
    Pure Python implementation, no Neo4j dependency.

    Args:
        knowledge_graph: Knowledge graph object
        resolution_parameter: Controls community granularity (higher = smaller communities, default 1.0)
        n_iterations: Number of iterations (default 2, usually sufficient for convergence)
        random_seed: Random seed for reproducibility

    Returns:
        Dict[int, str]: node_id -> community_id mapping (format: leiden_0, leiden_1, ...)
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError as e:
        logging.warning(
            f"[leiden] Missing dependencies (igraph/leidenalg): {e}. "
            "Install with: pip install igraph leidenalg"
        )
        return {}

    if not knowledge_graph.knowledge_nodes or not knowledge_graph.knowledge_edges:
        logging.info("[leiden] Knowledge graph is empty, skipping community detection")
        return {}

    # Build node ID to index mapping
    node_ids = [node.id for node in knowledge_graph.knowledge_nodes]
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    index_to_node_id = {idx: node_id for node_id, idx in node_id_to_index.items()}

    # Build edge list with deduplication (igraph uses indices, not raw IDs)
    edges_set = set()
    for edge in knowledge_graph.knowledge_edges:
        src_idx = node_id_to_index.get(edge.source_id)
        tgt_idx = node_id_to_index.get(edge.target_id)
        if src_idx is not None and tgt_idx is not None and src_idx != tgt_idx:
            # Undirected: always put smaller index first
            if src_idx < tgt_idx:
                edges_set.add((src_idx, tgt_idx))
            else:
                edges_set.add((tgt_idx, src_idx))

    if not edges_set:
        logging.warning("[leiden] No valid edges, skipping community detection")
        return {}

    g = ig.Graph(edges=list(edges_set), directed=False)

    if g.vcount() < 2:
        logging.info("[leiden] Graph has fewer than 2 nodes, skipping community detection")
        return {}

    # Run Leiden algorithm with RBConfigurationVertexPartition (supports resolution_parameter)
    try:
        # leidenalg API: find_partition passes args directly to the partition class
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution_parameter,
            n_iterations=n_iterations,
            seed=random_seed,
        )
    except Exception as e:
        logging.warning(f"[leiden] Leiden algorithm failed: {e}")
        return {}

    # Build node_id -> community_id mapping
    node_id_to_community: Dict[int, str] = {}
    for idx, community_id in enumerate(partition.membership):
        node_id = index_to_node_id[idx]
        node_id_to_community[node_id] = f"leiden_{community_id}"

    num_communities = len(set(partition.membership))
    logging.info(
        f"[leiden] Community detection complete: {num_communities} communities, "
        f"covering {len(node_id_to_community)} nodes"
    )

    return node_id_to_community


def apply_leiden_clustering_to_kg(
    knowledge_graph: "KnowledgeGraph",
    resolution_parameter: float = 1.0,
    n_iterations: int = 2,
    write_property: str = "community_id",
    random_seed: Optional[int] = None,
) -> Dict[int, str]:
    """
    Apply Leiden community detection to the knowledge graph and write results
    to node community_id properties. Uses community_id (not cluster_id) to
    avoid overwriting semantic clustering results.

    Args:
        knowledge_graph: Knowledge graph object
        resolution_parameter: Resolution parameter (default 1.0, configurable via DEEPRESEARCH_LEIDEN_RESOLUTION)
        n_iterations: Number of iterations (default 2, configurable via DEEPRESEARCH_LEIDEN_ITERATIONS)
        write_property: Property name to write (default "community_id")
        random_seed: Random seed (configurable via DEEPRESEARCH_LEIDEN_SEED)

    Returns:
        Dict[int, str]: node_id -> community_id mapping
    """
    # Allow disabling via env var
    if os.getenv("DEEPRESEARCH_ENABLE_LEIDEN", "").strip().lower() in {
        "false",
        "False",
    }:
        return {}

    # Read parameters from env vars (if not explicitly provided)
    if resolution_parameter == 1.0:
        resolution_parameter = float(os.getenv("DEEPRESEARCH_LEIDEN_RESOLUTION", "1.0"))
    if n_iterations == 2:
        n_iterations = int(os.getenv("DEEPRESEARCH_LEIDEN_ITERATIONS", "2"))
    if random_seed is None:
        seed_str = os.getenv("DEEPRESEARCH_LEIDEN_SEED", "")
        random_seed = int(seed_str) if seed_str.strip() else None

    node_id_to_community = _leiden_community_detection(
        knowledge_graph=knowledge_graph,
        resolution_parameter=resolution_parameter,
        n_iterations=n_iterations,
        random_seed=random_seed,
    )

    if not node_id_to_community:
        return {}

    # Apply results: sets node.community_id and updates knowledge_graph.community_list
    knowledge_graph.apply_community_detection_results(node_id_to_community)

    return node_id_to_community


def apply_community_detection(
    knowledge_graph: "KnowledgeGraph",
    neo4j_store: Optional[Any] = None,
) -> Dict[int, str]:
    """
    Unified community detection interface, selects implementation based on environment variables.

    Supported methods:
    - "leiden": Python igraph + leidenalg (default)
    - "neo4j_gds": Neo4j GDS plugin (requires neo4j_store)
    - "none": Disable community detection

    Environment variable configuration:
    - DEEPRESEARCH_COMMUNITY_DETECTION_ENABLED: Enable/disable (true/false, default true)
    - DEEPRESEARCH_COMMUNITY_DETECTION_METHOD: Method (leiden/neo4j_gds/none, default leiden)
    - DEEPRESEARCH_LEIDEN_RESOLUTION: Leiden resolution parameter (default 1.0)
    - DEEPRESEARCH_LEIDEN_ITERATIONS: Leiden iteration count (default 2)
    - DEEPRESEARCH_LEIDEN_SEED: Leiden random seed (optional)

    Args:
        knowledge_graph: Knowledge graph object
        neo4j_store: Optional Neo4jKGStore object (required for neo4j_gds method)

    Returns:
        Dict[int, str]: node_id -> community_id mapping
    """
    # 检查是否启用社区发现
    enabled = (
        os.getenv("DEEPRESEARCH_COMMUNITY_DETECTION_ENABLED", "true").strip().lower()
    )
    if enabled in {"false", "0", "no", "off"}:
        logging.info("[community_detection] Community detection disabled via environment variable")
        return {}

    # 获取实现方式
    method = (
        os.getenv("DEEPRESEARCH_COMMUNITY_DETECTION_METHOD", "leiden").strip().lower()
    )

    if method == "none":
        logging.info("[community_detection] Community detection method set to none, skipping")
        return {}

    elif method == "neo4j_gds":
        if neo4j_store is None:
            logging.warning(
                "[community_detection] Method set to neo4j_gds but neo4j_store not provided, "
                "falling back to leiden"
            )
            method = "leiden"
        else:
            try:
                # Neo4j GDS writes to community_id (not cluster_id)
                node_id_to_community = neo4j_store.run_gds_leiden_writeback(
                    write_property="community_id"
                )
                if node_id_to_community:
                    knowledge_graph.apply_community_detection_results(
                        node_id_to_community
                    )
                    logging.info(
                        f"[community_detection][neo4j_gds] Applied community detection to {len(node_id_to_community)} nodes"
                    )
                return node_id_to_community
            except Exception as e:
                logging.warning(
                    f"[community_detection][neo4j_gds] Neo4j GDS Leiden failed: {e}, "
                    "falling back to leiden"
                )
                method = "leiden"

    # Default: Python Leiden implementation
    if method == "leiden":
        try:
            leiden_communities = apply_leiden_clustering_to_kg(
                knowledge_graph=knowledge_graph,
                resolution_parameter=float(
                    os.getenv("DEEPRESEARCH_LEIDEN_RESOLUTION", "1.0")
                ),
                n_iterations=int(os.getenv("DEEPRESEARCH_LEIDEN_ITERATIONS", "2")),
                write_property="community_id",
                random_seed=None,  # 可通过环境变量 DEEPRESEARCH_LEIDEN_SEED 设置
            )
            if leiden_communities:
                logging.info(
                    f"[community_detection][leiden] Applied community detection to {len(leiden_communities)} nodes"
                )
            return leiden_communities
        except Exception as e:
            logging.exception(f"[community_detection][leiden] Leiden community detection failed: {e}")
            return {}

    else:
        logging.warning(
            f"[community_detection] Unknown method: {method}, "
            "supported methods: leiden, neo4j_gds, none"
        )
        return {}


def _generate_enrich_chains(
    kg: KnowledgeGraph,
    core_entity_ids: Dict[int, str],
    concept_ids: Dict[int, str],
    visited_edges: Set[Tuple[int, int, str]],
    enrich_evidence_threshold: int = 2,
) -> List[Chain]:
    """
    Generate enrich-type chains (for edges with insufficient evidence).

    Args:
        kg: Knowledge graph object
        core_entity_ids: Core entity ID to knowledge content mapping
        concept_ids: Concept ID to knowledge content mapping
        visited_edges: Set of already visited edges
        enrich_evidence_threshold: Threshold for evidence enrichment

    Returns:
        List of enrich-type chains
    """
    chains = []
    visited_enrich: Set[Tuple[int, int]] = set()
    node_id_to_knowledge = {node.id: node.knowledge for node in kg.knowledge_nodes}

    for edge in kg.knowledge_edges:
        if len(edge.evidence_nodes) < enrich_evidence_threshold:
            key = (edge.source_id, edge.target_id)
            if key not in visited_enrich:
                visited_enrich.add(key)
                src_name = node_id_to_knowledge.get(edge.source_id, "?")
                tgt_name = node_id_to_knowledge.get(edge.target_id, "?")
                chain_content = (
                    f"({src_name}) -> [{edge.relation_name}] -> ({tgt_name})"
                )
                chain_type = "enrich"
                chain_key = (edge.source_id, edge.target_id, chain_type)
                if chain_key not in visited_edges:
                    chains.append(
                        Chain(
                            id=0,
                            type=chain_type,
                            nodes=[edge.source_id, edge.target_id],
                            reason=f"Low evidence count ({len(edge.evidence_nodes)})",
                            content=chain_content,
                            is_visited=False,
                        )
                    )
    return chains


def _generate_sbm_explore_chains(
    kg: KnowledgeGraph,
    core_entity_ids: Dict[int, str],
    concept_ids: Dict[int, str],
    existing_relations: Set[Tuple[int, int]],
    visited_edges: Set[Tuple[int, int, str]],
) -> List[Chain]:
    """
    Generate SBM-based explore chains (explore_sbm_probability and explore_sbm_entropy).

    Args:
        kg: Knowledge graph object
        core_entity_ids: Core entity ID to knowledge content mapping
        concept_ids: Concept ID to knowledge content mapping
        existing_relations: Set of existing relations
        visited_edges: Set of already visited edges

    Returns:
        List of SBM-related explore chains
    """
    chains = []

    def compute_bernoulli_entropy(p: float, eps: float = 1e-10) -> float:
        """计算Bernoulli分布的熵：H(p) = -p*log(p) - (1-p)*log(1-p)"""
        p = max(eps, min(1 - eps, p))
        if p <= 0 or p >= 1:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    # Only execute SBM candidate edge generation when community info exists and >1 communities
    if kg.community_list and len(kg.community_list) > 1:
        # Build node-to-community and community-to-nodes mappings
        node_to_community: Dict[int, str] = {}
        community_to_nodes: Dict[str, List[int]] = {}
        all_node_ids = set()

        for comm in kg.community_list:
            comm_id = comm.get("community_id", "")
            node_ids = []
            for node_id_str in comm.get("source_node_ids", []):
                try:
                    node_id = int(node_id_str.replace("n", ""))
                    node_ids.append(node_id)
                    node_to_community[node_id] = comm_id
                    all_node_ids.add(node_id)
                except (ValueError, AttributeError):
                    continue
            if node_ids:
                community_to_nodes[comm_id] = node_ids

        # Skip SBM generation if no valid community mappings
        if node_to_community:
            # Build undirected adjacency for counting inter-community edges
            community_pair_edges: Dict[Tuple[str, str], int] = defaultdict(int)
            community_pair_pairs: Dict[Tuple[str, str], int] = defaultdict(int)
            seen_undirected_edges: Set[Tuple[int, int]] = set()

            for edge in kg.knowledge_edges:
                src_comm = node_to_community.get(edge.source_id)
                tgt_comm = node_to_community.get(edge.target_id)
                if src_comm and tgt_comm:
                    undirected_edge = tuple(sorted([edge.source_id, edge.target_id]))
                    if undirected_edge not in seen_undirected_edges:
                        seen_undirected_edges.add(undirected_edge)
                        comm_pair = tuple(sorted([src_comm, tgt_comm]))
                        community_pair_edges[comm_pair] += 1

            # Count possible pairs per community pair
            comm_ids = list(community_to_nodes.keys())
            for i, comm1 in enumerate(comm_ids):
                for j, comm2 in enumerate(comm_ids):
                    if i <= j:
                        comm_pair = tuple(sorted([comm1, comm2]))
                        n1 = len(community_to_nodes[comm1])
                        n2 = len(community_to_nodes[comm2])
                        if comm1 == comm2:
                            community_pair_pairs[comm_pair] = n1 * (n1 - 1) // 2
                        else:
                            community_pair_pairs[comm_pair] = n1 * n2

            # Estimate block connection probability matrix B
            smoothing = 0.1
            B_matrix: Dict[Tuple[str, str], float] = {}
            for comm_pair, possible_pairs in community_pair_pairs.items():
                actual_edges = community_pair_edges.get(comm_pair, 0)
                prob = (actual_edges + smoothing) / (possible_pairs + 2 * smoothing)
                prob = max(0.0, min(1.0, prob))
                B_matrix[comm_pair] = prob

            # Collect all cross-community missing edges
            cross_community_missing_edges: List[Tuple[int, int, str, str]] = []
            all_node_list = list(all_node_ids)

            for i, node_i in enumerate(all_node_list):
                comm_i = node_to_community.get(node_i)
                if not comm_i:
                    continue
                for j, node_j in enumerate(all_node_list):
                    if i >= j:
                        continue
                    comm_j = node_to_community.get(node_j)
                    if not comm_j or comm_i == comm_j:
                        continue

                    if (node_i, node_j) not in existing_relations and (
                        node_j,
                        node_i,
                    ) not in existing_relations:
                        cross_community_missing_edges.append(
                            (node_i, node_j, comm_i, comm_j)
                        )

            # 方案A：SBM-Probability Top-20
            scheme_a_candidates: List[Tuple[float, int, int, str, str]] = []
            for node_i, node_j, comm_i, comm_j in cross_community_missing_edges:
                comm_pair = tuple(sorted([comm_i, comm_j]))
                p_ij = B_matrix.get(comm_pair, 0.0)
                scheme_a_candidates.append((p_ij, node_i, node_j, comm_i, comm_j))

            scheme_a_candidates.sort(key=lambda x: x[0], reverse=True)
            top_20_probability = scheme_a_candidates[:20]

            # 生成方案A的chains
            for p_ij, node_i, node_j, comm_i, comm_j in top_20_probability:
                node_i_name = (
                    core_entity_ids.get(node_i)
                    or concept_ids.get(node_i)
                    or f"Node_{node_i}"
                )
                node_j_name = (
                    core_entity_ids.get(node_j)
                    or concept_ids.get(node_j)
                    or f"Node_{node_j}"
                )

                chain_content = f"({node_i_name}) -> [?] -> ({node_j_name}) [cross-community: {comm_i} -> {comm_j}, SBM-Prob={p_ij:.4f}]"
                chain_type = "explore_sbm_probability"
                chain_key = (node_i, node_j, chain_type)

                if chain_key not in visited_edges:
                    chains.append(
                        Chain(
                            id=0,
                            type=chain_type,
                            nodes=[node_i, node_j],
                            reason=f"SBM-Probability candidate (p={p_ij:.4f}): High-probability missing cross-community edge between {comm_i} and {comm_j}",
                            content=chain_content,
                            is_visited=False,
                        )
                    )

            # 方案B：SBM-Entropy Top-20
            scheme_b_candidates: List[Tuple[float, int, int, str, str]] = []
            for node_i, node_j, comm_i, comm_j in cross_community_missing_edges:
                comm_pair = tuple(sorted([comm_i, comm_j]))
                p_ij = B_matrix.get(comm_pair, 0.0)
                entropy_ij = compute_bernoulli_entropy(p_ij)
                scheme_b_candidates.append((entropy_ij, node_i, node_j, comm_i, comm_j))

            scheme_b_candidates.sort(key=lambda x: x[0], reverse=True)
            top_20_entropy = scheme_b_candidates[:20]

            # 生成方案B的chains
            for entropy_ij, node_i, node_j, comm_i, comm_j in top_20_entropy:
                node_i_name = (
                    core_entity_ids.get(node_i)
                    or concept_ids.get(node_i)
                    or f"Node_{node_i}"
                )
                node_j_name = (
                    core_entity_ids.get(node_j)
                    or concept_ids.get(node_j)
                    or f"Node_{node_j}"
                )

                comm_pair = tuple(sorted([comm_i, comm_j]))
                p_ij = B_matrix.get(comm_pair, 0.0)

                chain_content = f"({node_i_name}) -> [?] -> ({node_j_name}) [cross-community: {comm_i} -> {comm_j}, SBM-Entropy={entropy_ij:.4f}, p={p_ij:.4f}]"
                chain_type = "explore_sbm_entropy"
                chain_key = (node_i, node_j, chain_type)

                if chain_key not in visited_edges:
                    chains.append(
                        Chain(
                            id=0,
                            type=chain_type,
                            nodes=[node_i, node_j],
                            reason=f"SBM-Entropy candidate (H={entropy_ij:.4f}, p={p_ij:.4f}): High-uncertainty missing cross-community edge between {comm_i} and {comm_j}",
                            content=chain_content,
                            is_visited=False,
                        )
                    )

    return chains


def _generate_entity_concept_explore_chains(
    kg: KnowledgeGraph,
    core_entity_ids: Dict[int, str],
    concept_ids: Dict[int, str],
    existing_relations: Set[Tuple[int, int]],
    visited_edges: Set[Tuple[int, int, str]],
) -> List[Chain]:
    """
    Generate explore_entity_concept type chains (core entity - concept pair exploration).
    Scores and filters top-20 by semantic similarity.

    Args:
        kg: Knowledge graph object
        core_entity_ids: Core entity ID to knowledge content mapping
        concept_ids: Concept ID to knowledge content mapping
        existing_relations: Set of existing relations
        visited_edges: Set of already visited edges

    Returns:
        Filtered list of explore_entity_concept chains (top-20, sorted by semantic similarity)
    """
    # Build node ID to embedding mapping
    node_to_embedding: Dict[int, Optional[List[float]]] = {}
    embedding_model = "text-embedding-3-large"

    if not hasattr(kg, "embedding_cache"):
        kg.embedding_cache = {}

    nodes_to_embed: List[Tuple[int, str]] = []

    for node in kg.knowledge_nodes:
        cache_key = (node.knowledge, embedding_model)
        if cache_key in kg.embedding_cache:
            node_to_embedding[node.id] = kg.embedding_cache[cache_key]
        else:
            nodes_to_embed.append((node.id, node.knowledge))

    if nodes_to_embed:
        texts_to_embed = [knowledge for _, knowledge in nodes_to_embed]
        try:
            new_vectors = embed_texts(
                texts=texts_to_embed, embedding_model=embedding_model
            )
            if len(new_vectors) == len(nodes_to_embed):
                for (node_id, knowledge), new_vec in zip(nodes_to_embed, new_vectors):
                    cache_key = (knowledge, embedding_model)
                    kg.embedding_cache[cache_key] = new_vec
                    node_to_embedding[node_id] = new_vec
                logging.info(
                    f"[entity_concept_explore] Computed and cached {len(new_vectors)} missing node embeddings"
                )
            else:
                logging.warning(
                    f"[entity_concept_explore] Embedding count mismatch: expected={len(nodes_to_embed)} actual={len(new_vectors)}"
                )
        except Exception as e:
            logging.warning(f"[entity_concept_explore] Error computing embeddings: {e}")

    # Compute semantic similarity (cosine similarity)
    def compute_semantic_similarity(core_entity_id: int, concept_id: int) -> float:
        """Compute cosine similarity between a core entity and concept node."""
        core_emb = node_to_embedding.get(core_entity_id)
        concept_emb = node_to_embedding.get(concept_id)

        if core_emb is None or concept_emb is None:
            return 0.0

        try:
            import numpy as np

            core_vec = np.array(core_emb)
            concept_vec = np.array(concept_emb)
            dot_product = np.dot(core_vec, concept_vec)
            norm_core = np.linalg.norm(core_vec)
            norm_concept = np.linalg.norm(concept_vec)
            if norm_core > 0 and norm_concept > 0:
                return float(dot_product / (norm_core * norm_concept))
            return 0.0
        except Exception:
            return 0.0

    # Score all core entity - concept pairs by semantic similarity
    scored_pairs: List[Tuple[float, int, int]] = []

    for core_entity_id in core_entity_ids.keys():
        for concept_id in concept_ids.keys():
            chain_type = "explore_entity_concept"
            chain_key = (core_entity_id, concept_id, chain_type)
            if chain_key in visited_edges:
                continue

            similarity = compute_semantic_similarity(core_entity_id, concept_id)
            scored_pairs.append((similarity, core_entity_id, concept_id))

    # Sort by semantic similarity descending
    scored_pairs.sort(key=lambda x: x[0], reverse=True)

    # Select top-20 and create Chain objects
    top_20_pairs = scored_pairs[:20]

    chains = []
    for similarity, core_entity_id, concept_id in top_20_pairs:
        chain_content = (
            f"({core_entity_ids[core_entity_id]}) -> [?] -> ({concept_ids[concept_id]})"
        )
        chain_type = "explore_entity_concept"
        chains.append(
            Chain(
                id=0,
                type=chain_type,
                nodes=[core_entity_id, concept_id],
                reason=f"High semantic similarity: {similarity:.4f}",
                content=chain_content,
                is_visited=False,
            )
        )

    return chains


def _generate_cross_community_explore_chains(
    kg: KnowledgeGraph,
    existing_relations: Set[Tuple[int, int]],
    visited_edges: Set[Tuple[int, int, str]],
) -> List[Chain]:
    """
    Generate explore_cross_community type chains (structural hole exploration).

    Args:
        kg: Knowledge graph object
        existing_relations: Set of existing relations
        visited_edges: Set of already visited edges

    Returns:
        List of explore_cross_community chains
    """
    chains = []

    if kg.community_list and len(kg.community_list) > 1:
        community_to_nodes: Dict[str, List[int]] = {}
        node_to_community: Dict[int, str] = {}
        for comm in kg.community_list:
            comm_id = comm.get("community_id", "")
            node_ids = []
            # Extract node IDs from source_node_ids (format: "n{id}")
            for node_id_str in comm.get("source_node_ids", []):
                try:
                    node_id = int(node_id_str.replace("n", ""))
                    node_ids.append(node_id)
                    node_to_community[node_id] = comm_id
                except (ValueError, AttributeError):
                    continue
            if node_ids:
                community_to_nodes[comm_id] = node_ids

        # Compute neighbor sets and degree for each node
        node_neighbors: Dict[int, Set[int]] = defaultdict(set)
        node_degree: Dict[int, int] = defaultdict(int)
        for edge in kg.knowledge_edges:
            node_neighbors[edge.source_id].add(edge.target_id)
            node_neighbors[edge.target_id].add(edge.source_id)
            node_degree[edge.source_id] += 1
            node_degree[edge.target_id] += 1

        # Compute cross-community neighbor count and bridging score per node
        node_cross_neighbors: Dict[int, int] = {}
        node_bridging_score: Dict[int, float] = {}
        for node_id, neighbors in node_neighbors.items():
            node_comm = node_to_community.get(node_id)
            if not node_comm:
                node_cross_neighbors[node_id] = 0
                node_bridging_score[node_id] = 0.0
                continue

            cross_neighbors = 0
            for neighbor_id in neighbors:
                neighbor_comm = node_to_community.get(neighbor_id)
                if neighbor_comm and neighbor_comm != node_comm:
                    cross_neighbors += 1

            node_cross_neighbors[node_id] = cross_neighbors

            deg = len(neighbors)
            bridging_score = cross_neighbors / max(deg, 1)
            node_bridging_score[node_id] = bridging_score

        # Compute intra-community degree per node
        node_in_community_degree: Dict[int, int] = {}
        for node_id, neighbors in node_neighbors.items():
            node_comm = node_to_community.get(node_id)
            if not node_comm:
                node_in_community_degree[node_id] = 0
                continue

            in_comm_degree = 0
            for neighbor_id in neighbors:
                neighbor_comm = node_to_community.get(neighbor_id)
                if neighbor_comm == node_comm:
                    in_comm_degree += 1

            node_in_community_degree[node_id] = in_comm_degree

        # Node selection (role separation): bridge nodes, hub nodes, representatives
        max_bridge_nodes_per_community = 3
        max_hub_nodes_per_community = 2

        community_bridge_nodes: Dict[str, List[int]] = {}
        community_hub_nodes: Dict[str, List[int]] = {}
        community_boundary_hub_nodes: Dict[str, List[int]] = {}

        for comm_id, node_ids in community_to_nodes.items():
            # Bridge/boundary candidate nodes
            bridge_candidates = [
                (nid, node_bridging_score.get(nid, 0.0))
                for nid in node_ids
                if node_cross_neighbors.get(nid, 0) > 0
            ]
            bridge_candidates.sort(key=lambda x: x[1], reverse=True)
            bridge_nodes = [
                nid for nid, _ in bridge_candidates[:max_bridge_nodes_per_community]
            ]
            if bridge_nodes:
                community_bridge_nodes[comm_id] = bridge_nodes

            # Hub nodes
            hub_candidates = [
                (nid, node_in_community_degree.get(nid, 0))
                for nid in node_ids
                if nid not in bridge_nodes
            ]
            hub_candidates.sort(key=lambda x: x[1], reverse=True)
            hub_nodes = [nid for nid, _ in hub_candidates[:max_hub_nodes_per_community]]
            if hub_nodes:
                community_hub_nodes[comm_id] = hub_nodes

            # Merge bridge and hub nodes
            boundary_hub_nodes = bridge_nodes + hub_nodes
            if boundary_hub_nodes:
                community_boundary_hub_nodes[comm_id] = boundary_hub_nodes

        # Representative nodes (semantic anchors)
        community_representatives: Dict[str, int] = {}
        for comm in kg.community_list:
            comm_id = comm.get("community_id", "")
            node_ids = community_to_nodes.get(comm_id, [])
            if node_ids:
                representative = max(
                    node_ids,
                    key=lambda nid: (
                        node_in_community_degree.get(nid, 0),
                        node_degree.get(nid, 0),
                    ),
                )
                community_representatives[comm_id] = representative

        # Channel 1: Structural hole exploration (bridge/hubs -> other community rep), quota 20
        max_cross_community_pairs = 20
        cross_community_pairs = []

        comm_ids = list(community_to_nodes.keys())
        for i, comm_id1 in enumerate(comm_ids):
            if comm_id1 not in community_boundary_hub_nodes:
                continue
            boundary_hub_nodes1 = community_boundary_hub_nodes[comm_id1]

            for comm_id2 in comm_ids[i + 1 :]:
                if comm_id2 not in community_representatives:
                    continue
                rep_node2 = community_representatives[comm_id2]

                # For each bridge/hub node, create candidate pairs with another community's representative
                for boundary_hub_node in boundary_hub_nodes1:
                    if (boundary_hub_node, rep_node2) not in existing_relations:
                        cross_community_pairs.append(
                            (boundary_hub_node, rep_node2, comm_id1, comm_id2)
                        )
                    if len(cross_community_pairs) >= max_cross_community_pairs:
                        break
                if len(cross_community_pairs) >= max_cross_community_pairs:
                    break
            if len(cross_community_pairs) >= max_cross_community_pairs:
                break

        # 创建跨社区探索链
        for src_id, tgt_id, comm_id1, comm_id2 in cross_community_pairs[
            :max_cross_community_pairs
        ]:
            src_node = kg.get_knowledge_node_by_id(src_id)
            tgt_node = kg.get_knowledge_node_by_id(tgt_id)
            if src_node and tgt_node:
                chain_content = f"({src_node.knowledge}) -> [?] -> ({tgt_node.knowledge}) [cross-community: {comm_id1} -> {comm_id2}]"
                chain_type = "explore_cross_community"
                chain_key = (src_id, tgt_id, chain_type)
                if chain_key not in visited_edges:
                    chains.append(
                        Chain(
                            id=0,
                            type=chain_type,
                            nodes=[src_id, tgt_id],
                            reason=f"Cross-community potential edge (structural hole) between community {comm_id1} and {comm_id2}",
                            content=chain_content,
                            is_visited=False,
                        )
                    )

    return chains


def _build_scoring_context(
    kg: KnowledgeGraph,
    core_entity_ids: Dict[int, str],
) -> Dict[str, Any]:
    """
    Build all auxiliary data structures needed for scoring.

    Args:
        kg: Knowledge graph object
        core_entity_ids: Core entity ID to knowledge content mapping

    Returns:
        Dict containing all scoring data
    """
    node_to_cluster_id: Dict[int, Optional[str]] = {}
    node_to_community_id: Dict[int, Optional[str]] = {}
    for node in kg.knowledge_nodes:
        node_to_cluster_id[node.id] = node.cluster_id
        node_to_community_id[node.id] = node.community_id

    node_importance_dict: Dict[int, float] = {}
    for node in kg.knowledge_nodes:
        node_importance_dict[node.id] = 1.0 if node.is_core_entity else 0.0

    node_degree: Dict[int, int] = defaultdict(int)
    for edge in kg.knowledge_edges:
        node_degree[edge.source_id] += 1
        node_degree[edge.target_id] += 1

    # Compute cross-community ratio per node (used for Enrich Score)
    node_cross_ratio: Dict[int, float] = {}
    node_neighbors_for_cross: Dict[int, Set[int]] = defaultdict(set)
    for edge in kg.knowledge_edges:
        node_neighbors_for_cross[edge.source_id].add(edge.target_id)
        node_neighbors_for_cross[edge.target_id].add(edge.source_id)

    for node in kg.knowledge_nodes:
        node_id = node.id
        neighbors = node_neighbors_for_cross.get(node_id, set())
        deg = len(neighbors)
        node_comm = node_to_community_id.get(node_id)

        if node_comm and deg > 0:
            cross_neighbors = 0
            for neighbor_id in neighbors:
                neighbor_comm = node_to_community_id.get(neighbor_id)
                if neighbor_comm and neighbor_comm != node_comm:
                    cross_neighbors += 1
            node_cross_ratio[node_id] = cross_neighbors / max(deg, 1)
        else:
            node_cross_ratio[node_id] = 0.0

    # Normalize node degree
    node_degree_normalized: Dict[int, float] = {}
    if node_degree:
        max_degree = max(node_degree.values())
        if max_degree > 0:
            for node_id, degree in node_degree.items():
                node_degree_normalized[node_id] = degree / max_degree
        else:
            for node_id in node_degree:
                node_degree_normalized[node_id] = 0.0
    else:
        for node in kg.knowledge_nodes:
            node_degree_normalized[node.id] = 0.0

    # Compute clustering coefficient per node (for gap score)
    adjacency_list: Dict[int, Set[int]] = defaultdict(set)
    for edge in kg.knowledge_edges:
        adjacency_list[edge.source_id].add(edge.target_id)
        adjacency_list[edge.target_id].add(edge.source_id)

    node_clustering_coefficient: Dict[int, float] = {}
    node_gap_score: Dict[int, float] = {}

    for node in kg.knowledge_nodes:
        node_id = node.id
        neighbors = adjacency_list.get(node_id, set())
        degree = len(neighbors)

        if degree < 2:
            node_clustering_coefficient[node_id] = 0.0
            node_gap_score[node_id] = 1.0
        else:
            edges_between_neighbors = 0
            neighbors_list = list(neighbors)
            for i in range(len(neighbors_list)):
                for j in range(i + 1, len(neighbors_list)):
                    u, v = neighbors_list[i], neighbors_list[j]
                    if v in adjacency_list.get(u, set()):
                        edges_between_neighbors += 1

            max_possible_edges = degree * (degree - 1) / 2.0
            if max_possible_edges > 0:
                clustering_coeff = edges_between_neighbors / max_possible_edges
            else:
                clustering_coeff = 0.0

            node_clustering_coefficient[node_id] = clustering_coeff
            node_gap_score[node_id] = 1.0 - clustering_coeff

    for node in kg.knowledge_nodes:
        if node.id not in node_gap_score:
            node_gap_score[node.id] = 1.0

    # Build node ID to embedding mapping
    node_to_embedding: Dict[int, Optional[List[float]]] = {}
    embedding_model = "text-embedding-3-large"
    if hasattr(kg, "embedding_cache"):
        for node in kg.knowledge_nodes:
            cache_key = (node.knowledge, embedding_model)
            if cache_key in kg.embedding_cache:
                node_to_embedding[node.id] = kg.embedding_cache[cache_key]

    return {
        "node_to_cluster_id": node_to_cluster_id,
        "node_to_community_id": node_to_community_id,
        "node_importance_dict": node_importance_dict,
        "node_degree": node_degree,
        "node_cross_ratio": node_cross_ratio,
        "node_degree_normalized": node_degree_normalized,
        "node_clustering_coefficient": node_clustering_coefficient,
        "node_gap_score": node_gap_score,
        "node_to_embedding": node_to_embedding,
    }


def _score_enrich_chains(
    chains: List[Chain],
    core_entity_ids: Dict[int, str],
    edge_lookup: Dict[Tuple[int, int], Any],
    scoring_context: Dict[str, Any],
    min_chain_score: float = 0.0,
) -> Tuple[List[Tuple[Tuple[float, float], Chain]], List[Chain]]:
    """
    Score and filter enrich-type chains.

    Args:
        chains: List of all chains
        core_entity_ids: Core entity ID to knowledge content mapping
        edge_lookup: Edge lookup dict
        scoring_context: Scoring context with auxiliary data structures
        min_chain_score: Minimum chain score threshold

    Returns:
        (scored_enrich_chains, explore_chains) tuple
    """
    node_cross_ratio = scoring_context["node_cross_ratio"]

    def cross_community(u_id: int, v_id: int) -> float:
        u_cross_ratio = node_cross_ratio.get(u_id, 0.0)
        v_cross_ratio = node_cross_ratio.get(v_id, 0.0)
        return 0.5 * (u_cross_ratio + v_cross_ratio)

    def compute_node_importance(u_id: int, v_id: int) -> float:
        u_is_core = u_id in core_entity_ids
        v_is_core = v_id in core_entity_ids
        if u_is_core and v_is_core:
            return 1.0
        elif u_is_core or v_is_core:
            return 0.5
        else:
            return 0.0

    # Enrich 评分函数
    def score_enrich(chain: Chain) -> Tuple[float, float]:
        """返回 (sort_key, original_score) 元组"""
        if len(chain.nodes) < 2:
            return (1.0, 0.0)

        src_id, tgt_id = chain.nodes[0], chain.nodes[1]
        edge = edge_lookup.get((src_id, tgt_id))
        if edge is None:
            return (1.0, 0.0)

        evidence_count = len(edge.evidence_nodes)
        node_imp = compute_node_importance(src_id, tgt_id)
        cross_comm = cross_community(src_id, tgt_id)

        numerator = 1.0 + node_imp + cross_comm
        denominator = 1.0 + evidence_count

        if denominator == 0:
            original_score = 0.0
        else:
            original_score = numerator / denominator

        sort_key = 0.0 if evidence_count == 0 else 1.0
        return (sort_key, original_score)

    # 对所有链进行评分
    scored_chains: List[Tuple[Tuple[float, float], Chain]] = []
    explore_chains: List[Chain] = []

    for chain in chains:
        if chain.type == "enrich":
            score_tuple = score_enrich(chain)
            sort_key, original_score = score_tuple
            if original_score >= min_chain_score:
                scored_chains.append((score_tuple, chain))
        else:
            explore_chains.append(chain)

    return scored_chains, explore_chains


def _deduplicate_and_sort_enrich_chains(
    scored_enrich_chains: List[Tuple[Tuple[float, float], Chain]],
) -> List[Chain]:
    """
    Deduplicate and sort enrich-type chains.

    Args:
        scored_enrich_chains: Scored enrich chain list

    Returns:
        Deduplicated and sorted enrich chain list
    """
    # Keep only the highest-scoring chain per edge
    edge_to_best_chain: Dict[Tuple[int, int], Tuple[Tuple[float, float], Chain]] = {}
    for score_tuple, chain in scored_enrich_chains:
        if len(chain.nodes) >= 2:
            edge_key = (chain.nodes[0], chain.nodes[1])
            if edge_key not in edge_to_best_chain:
                edge_to_best_chain[edge_key] = (score_tuple, chain)
            else:
                best_sort_key, best_original_score = edge_to_best_chain[edge_key][0]
                current_sort_key, current_original_score = score_tuple
                if current_sort_key < best_sort_key or (
                    current_sort_key == best_sort_key
                    and current_original_score > best_original_score
                ):
                    edge_to_best_chain[edge_key] = (score_tuple, chain)

    # Two-level sort: 1) sort_key ascending (evidence=0 first), 2) original_score descending
    deduplicated_chains_with_score = list(edge_to_best_chain.values())
    deduplicated_chains_with_score.sort(key=lambda x: (x[0][0], -x[0][1]))

    return [chain for _, chain in deduplicated_chains_with_score]


def _allocate_quota_by_type(
    deduplicated_enrich_chains: List[Chain],
    explore_chains: List[Chain],
    max_chains: int,
) -> List[Chain]:
    """
    Allocate quota by type and filter chains.

    Args:
        deduplicated_enrich_chains: Deduplicated and sorted enrich chain list
        explore_chains: Explore type chain list
        max_chains: Maximum number of chains to return

    Returns:
        Filtered chain list
    """
    # Group explore chains by type
    explore_chains_by_type: Dict[str, List[Chain]] = defaultdict(list)
    for chain in explore_chains:
        chain_type = chain.type if chain.type else "unknown"
        explore_chains_by_type[chain_type].append(chain)

    # Quota weights: enrich:2, entity_concept:2, cross_community:2, sbm_prob:1, sbm_entropy:1
    target_types = [
        "enrich",
        "explore_entity_concept",
        "explore_cross_community",
        "explore_sbm_probability",
        "explore_sbm_entropy",
    ]

    quota_weights = {
        "enrich": 2,
        "explore_entity_concept": 2,
        "explore_cross_community": 2,
        "explore_sbm_probability": 1,
        "explore_sbm_entropy": 1,
    }
    total_weights = sum(quota_weights.values())

    selected_chains_by_type: Dict[str, List[Chain]] = {}
    for chain_type in target_types:
        quota = int(max_chains * quota_weights[chain_type] / total_weights)
        if chain_type == "enrich":
            selected_chains_by_type[chain_type] = deduplicated_enrich_chains[:quota]
        else:
            available_chains = explore_chains_by_type.get(chain_type, [])
            selected_chains_by_type[chain_type] = available_chains[:quota]

    # 计算剩余名额
    actual_counts = {
        chain_type: len(selected_chains_by_type[chain_type])
        for chain_type in target_types
    }
    total_selected = sum(actual_counts.values())
    remaining_slots = max_chains - total_selected

    # 按优先级补充剩余名额（按权重优先级）
    if remaining_slots > 0:
        priority_order = [
            "enrich",
            "explore_entity_concept",
            "explore_cross_community",
            "explore_sbm_probability",
            "explore_sbm_entropy",
        ]

        for chain_type in priority_order:
            if remaining_slots <= 0:
                break

            if chain_type == "enrich":
                remaining_enrich = deduplicated_enrich_chains[
                    actual_counts[chain_type] :
                ]
                additional = remaining_enrich[:remaining_slots]
                selected_chains_by_type[chain_type].extend(additional)
                remaining_slots -= len(additional)
            else:
                available_chains = explore_chains_by_type.get(chain_type, [])
                remaining_explore = available_chains[actual_counts[chain_type] :]
                additional = remaining_explore[:remaining_slots]
                selected_chains_by_type[chain_type].extend(additional)
                remaining_slots -= len(additional)

    # 合并最终结果
    filtered_chains: List[Chain] = []
    for chain_type in target_types:
        filtered_chains.extend(selected_chains_by_type.get(chain_type, []))

    return filtered_chains[:max_chains]


def _finalize_chains(chains: List[Chain]) -> List[Chain]:
    """Finalize chains: renumber IDs and write debug file.

    Args:
        chains: List of chains.

    Returns:
        Renumbered list of chains.
    """
    for idx, chain in enumerate(chains):
        chain.id = idx + 1

    with open(PROMPT_LIB_DIR / "debug_filtered_chains.txt", "w", encoding="utf-8") as f:
        f.write(f"=== Filtered Chains ===\n")
        for chain in chains:
            f.write(
                f"Chain: {chain.id}. {chain.content} -- Reason: {chain.reason} -- type={chain.type} \n"
            )
        f.write("--------------------------------\n")

    return chains


def _build_search_chains(
    kg: KnowledgeGraph,
    visited_edges: Optional[Set[Tuple[int, int, str]]] = None,
    enrich_evidence_threshold: int = 2,
    min_chain_score: float = 0.0,
    max_chains: Optional[int] = 60,
) -> List[Chain]:
    if visited_edges is None:
        visited_edges = set()

    # Step 1:构建索引，方便后续计算
    # 初始化链列表
    chains = []
    core_entity_ids = {
        node.id: node.knowledge for node in kg.knowledge_nodes if node.is_core_entity
    }
    concept_ids = {
        node.id: node.knowledge
        for node in kg.knowledge_nodes
        if not node.is_core_entity
    }

    # Build relation set for O(1) lookup
    existing_relations = set()
    edge_map = defaultdict(list)  # src_id -> list of edges
    edge_lookup = {}  # (src_id, tgt_id) -> edge
    for edge in kg.knowledge_edges:
        existing_relations.add((edge.source_id, edge.target_id))
        edge_map[edge.source_id].append(edge)
        edge_lookup[(edge.source_id, edge.target_id)] = edge

    # 1. Enrich Chains
    enrich_chains = _generate_enrich_chains(
        kg=kg,
        core_entity_ids=core_entity_ids,
        concept_ids=concept_ids,
        visited_edges=visited_edges,
        enrich_evidence_threshold=enrich_evidence_threshold,
    )
    chains.extend(enrich_chains)

    # 2. Explore Chains
    # --- Step 2.0: SBM-based Candidate Edge Generation (Two Schemes) ---
    sbm_chains = _generate_sbm_explore_chains(
        kg=kg,
        core_entity_ids=core_entity_ids,
        concept_ids=concept_ids,
        existing_relations=existing_relations,
        visited_edges=visited_edges,
    )
    chains.extend(sbm_chains)

    # --- Step 2.1: Explore Missing Entity-Concept Pairs ---
    entity_concept_chains = _generate_entity_concept_explore_chains(
        kg=kg,
        core_entity_ids=core_entity_ids,
        concept_ids=concept_ids,
        existing_relations=existing_relations,
        visited_edges=visited_edges,
    )
    chains.extend(entity_concept_chains)

    # --- Step 2.3: Explore Cross-Community Potential Edges (Structural Holes) ---
    cross_community_chains = _generate_cross_community_explore_chains(
        kg=kg,
        existing_relations=existing_relations,
        visited_edges=visited_edges,
    )
    chains.extend(cross_community_chains)

    # Step 3: Build scoring context
    scoring_context = _build_scoring_context(kg=kg, core_entity_ids=core_entity_ids)

    # Step 4: Score and filter chains
    scored_enrich_chains, explore_chains = _score_enrich_chains(
        chains=chains,
        core_entity_ids=core_entity_ids,
        edge_lookup=edge_lookup,
        scoring_context=scoring_context,
        min_chain_score=min_chain_score,
    )

    # Step 5: Deduplicate and sort enrich chains
    deduplicated_enrich_chains = _deduplicate_and_sort_enrich_chains(
        scored_enrich_chains
    )

    # Write all chains to debug file
    with open(PROMPT_LIB_DIR / "debug_all_chains.txt", "w", encoding="utf-8") as f:
        f.write(f"=== All Chains ===\n")
        for chain in chains:
            f.write(f"Chain: {chain.id}. {chain.content} -- Reason: {chain.reason} \n")
        f.write("--------------------------------\n")
    # Step 6: Allocate quota by type
    if max_chains is not None:
        filtered_chains = _allocate_quota_by_type(
            deduplicated_enrich_chains=deduplicated_enrich_chains,
            explore_chains=explore_chains,
            max_chains=max_chains,
        )
    else:
        # If max_chains is None, return all chains
        filtered_chains = deduplicated_enrich_chains + explore_chains

    # Step 7: Finalize (renumber and write debug file)
    return _finalize_chains(filtered_chains)


def _extract_knowledge_by_original_prompt(
    root_query: str,
    knowledge_graph: KnowledgeGraph,
    evidence_nodes: List[EvidenceNode],
    search_query: str,
    llm_model,
    report_id: int = None,
    usage_file: str = None,
    is_first_batch: bool = True,
    use_summary_only: bool = True,
    show_kg: bool = False,
    **kwargs,
) -> dict:
    """Original path: single (query, evidence) LLM call for KG extraction.
    Output format is JSON (new_nodes, new_edges, evidences_map).

    Args:
        is_first_batch: True=use extract_knowledge_node.yaml, False=use update_knowledge_node.yaml
        use_summary_only: True=use only Summary from Evidence
        show_kg: True=include existing KG context
    """
    # Select prompt template
    if is_first_batch:
        yaml_file = PROMPT_LIB_DIR / "extract_knowledge_node.yaml"
    else:
        yaml_file = PROMPT_LIB_DIR / "update_knowledge_node.yaml"
    with open(yaml_file, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    system_prompt = yaml_data["system"]

    # Build evidence text
    evidence_texts = []
    for en in evidence_nodes:
        txt = (
            _extract_summary_from_content(en.content)
            if use_summary_only
            else en.content
        )
        evidence_texts.append(f"EN{en.id}: {txt}")
    evidences_str = "\n".join(evidence_texts)

    user_prompt = f"""Root Query: {root_query}

Search Query: {search_query}

Evidences: {evidences_str}"""

    if show_kg and knowledge_graph.knowledge_nodes:
        user_prompt += f"""

Current Knowledge Graph: {knowledge_graph.to_toon()}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    max_retries = 3
    start_time = time.time()
    for attempt in range(max_retries):
        try:
            response = call_llm_model(
                llm_model=llm_model,
                messages=messages,
                temperature=0.3,
                **kwargs,
            )
            end_time = time.time()
            llm_output = safe_json_loads(response.content)
            if llm_output is None:
                raise ValueError("safe_json_loads returned None")
            if report_id is not None and usage_file is not None:
                update_llm_usage(
                    response,
                    (
                        "extract_knowledge_nodes"
                        if is_first_batch
                        else "update_knowledge_node"
                    ),
                    report_id,
                    usage_file,
                    elapsed_time=getattr(
                        response, "_call_elapsed_time", end_time - start_time
                    ),
                )
            return llm_output
        except Exception:
            if attempt == max_retries - 1:
                raise


def _extract_knowledge_by_original_prompt_batch(
    root_query: str,
    knowledge_graph: KnowledgeGraph,
    query_evidence_pairs: List[tuple],
    llm_model,
    report_id: int = None,
    usage_file: str = None,
    is_first_batch: bool = True,
    use_summary_only: bool = True,
    show_kg: bool = False,
    **kwargs,
) -> dict:
    """Original path: batch multiple (search_query, evidence_nodes) into a single LLM call.
    Amortizes system prompt + KG context overhead, saving tokens.
    Output format is JSON (new_nodes, new_edges, evidences_map).

    Args:
        query_evidence_pairs: [(search_query, [EvidenceNode, ...]), ...]
        is_first_batch: True=use extract_knowledge_node.yaml, False=use update_knowledge_node.yaml
        use_summary_only: True=use only Summary portion
        show_kg: True=include existing KG context
    """
    # Select prompt template
    if is_first_batch:
        yaml_file = PROMPT_LIB_DIR / "extract_knowledge_node.yaml"
    else:
        yaml_file = PROMPT_LIB_DIR / "update_knowledge_node.yaml"
    with open(yaml_file, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    system_prompt = yaml_data["system"]

    # Combine multiple (query, evidence) groups
    combined_parts = []
    for search_query, evidence_nodes in query_evidence_pairs:
        evidence_texts = []
        for en in evidence_nodes:
            txt = (
                _extract_summary_from_content(en.content)
                if use_summary_only
                else en.content
            )
            evidence_texts.append(f"EN{en.id}: {txt}")
        evidences_str = "\n".join(evidence_texts)
        combined_parts.append(
            f"## Search Query: {search_query}\n\nEvidences: {evidences_str}"
        )

    combined_text = "\n\n".join(combined_parts)

    user_prompt = f"""Root Query: {root_query}
Note: The evidences come from multiple search queries. Process all of them together in a single JSON output.

{combined_text}"""

    if show_kg and knowledge_graph.knowledge_nodes:
        user_prompt += f"""

Current Knowledge Graph: {knowledge_graph.to_toon()}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    max_retries = 3
    start_time = time.time()
    for attempt in range(max_retries):
        try:
            response = call_llm_model(
                llm_model=llm_model,
                messages=messages,
                temperature=0.3,
                **kwargs,
            )
            end_time = time.time()
            llm_output = safe_json_loads(response.content)
            if llm_output is None:
                raise ValueError("safe_json_loads returned None")
            if report_id is not None and usage_file is not None:
                update_llm_usage(
                    response,
                    (
                        "extract_knowledge_nodes"
                        if is_first_batch
                        else "update_knowledge_node"
                    ),
                    report_id,
                    usage_file,
                    elapsed_time=getattr(
                        response, "_call_elapsed_time", end_time - start_time
                    ),
                )
            return llm_output
        except Exception:
            if attempt == max_retries - 1:
                raise


def create_knowledge_graph(
    root_query: str,
    knowledge_graph: KnowledgeGraph,
    evidence_nodes_batch: List[List[EvidenceNode]],
    search_queries: List[str],
    llm_model,
    report_id: int = None,
    usage_file: str = None,
    use_summary_only: bool = True,
    show_kg: bool = False,
    batch_queries: int = 5,
    **kwargs,
) -> KnowledgeGraph:
    """
    Create a knowledge graph by extracting knowledge nodes from evidence batches.

    Extracts knowledge from multiple batches of evidence nodes, builds the initial KG,
    and performs semantic merging.

    Args:
        root_query: Root query guiding knowledge extraction.
        knowledge_graph: KnowledgeGraph object (may be empty or partially populated).
        evidence_nodes_batch: List of evidence node batches.
        search_queries: Search query list, one-to-one with evidence_nodes_batch.
        llm_model: LLM model object for calling the language model.
        report_id: Report ID for tracking LLM usage (optional).
        use_summary_only: True=use only Summary from Evidence (default True).
        show_kg: True=include existing KG in extraction prompt (default False, saves tokens).
        batch_queries: How many (query,evidence) pairs to batch per LLM call (default 5, set 1 for per-call).
        **kwargs: Additional arguments passed to LLM calls.

    Returns:
        KnowledgeGraph: The created and merged knowledge graph.
    """
    assert len(evidence_nodes_batch) == len(
        search_queries
    ), "Mismatch in evidence nodes batch and search queries length"
    # 1. Extract Knowledge Nodes from Evidence Nodes
    create_flag = True
    # Batch by batch_queries
    pairs = list(zip(evidence_nodes_batch, search_queries))
    for i in range(0, len(pairs), batch_queries):
        chunk = pairs[i : i + batch_queries]
        chunk_ens_list = [p[0] for p in chunk]
        chunk_queries = [p[1] for p in chunk]
        all_chunk_ens = [en for ens in chunk_ens_list for en in ens]

        if len(chunk) == 1:
            llm_output = _extract_knowledge_by_original_prompt(
                root_query=root_query,
                knowledge_graph=knowledge_graph,
                evidence_nodes=chunk_ens_list[0],
                search_query=chunk_queries[0],
                llm_model=llm_model,
                report_id=report_id,
                usage_file=usage_file,
                is_first_batch=create_flag,
                use_summary_only=use_summary_only,
                show_kg=show_kg,
                **kwargs,
            )
        else:
            llm_output = _extract_knowledge_by_original_prompt_batch(
                root_query=root_query,
                knowledge_graph=knowledge_graph,
                query_evidence_pairs=list(zip(chunk_queries, chunk_ens_list)),
                llm_model=llm_model,
                report_id=report_id,
                usage_file=usage_file,
                is_first_batch=create_flag,
                use_summary_only=use_summary_only,
                show_kg=show_kg,
                **kwargs,
            )
        knowledge_graph.add_llm_generated_knowledge(llm_output, all_chunk_ens)
        create_flag = False

    # 2. Semantic merge of KG nodes
    with open(PROMPT_LIB_DIR / "merge_knowledge_node.yaml", "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    system_prompt = yaml_data["system"]
    user_prompt = f"""Root Query: {root_query}

    Current Knowledge Graph: {knowledge_graph.to_toon()}"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            merge_knowledge_node_start_time = time.time()
            response = call_llm_model(
                llm_model=llm_model,
                messages=messages,
                temperature=0.3,
            )
            merge_knowledge_node_end_time = time.time()
            llm_output = safe_json_loads(response.content)
            if llm_output is None:
                raise ValueError("safe_json_loads returned None")
            knowledge_graph.apply_merge_node_results(llm_output)
            if report_id is not None and usage_file is not None:
                update_llm_usage(
                    response,
                    "merge_knowledge_node",
                    report_id,
                    usage_file,
                    elapsed_time=getattr(
                        response,
                        "_call_elapsed_time",
                        merge_knowledge_node_end_time - merge_knowledge_node_start_time,
                    ),
                )
            break
        except Exception:
            if attempt == max_retries - 1:
                raise
    return knowledge_graph


def update_knowledge_graph(
    root_query: str,
    knowledge_graph: KnowledgeGraph,
    evidence_nodes_batch: List[List[EvidenceNode]],
    search_queries: List[str],
    llm_model,
    report_id: int = None,
    usage_file: str = None,
    use_summary_only: bool = True,
    show_kg: bool = False,
    batch_queries: int = 5,
    **kwargs,
) -> KnowledgeGraph:
    """
    Update an existing knowledge graph with new evidence batches.

    Adds new knowledge nodes and relations, then performs semantic merging and clustering.

    Args:
        root_query: Root query guiding knowledge update.
        knowledge_graph: Existing knowledge graph object.
        evidence_nodes_batch: List of evidence node batches.
        search_queries: Search query list, one-to-one with evidence_nodes_batch.
        llm_model: LLM model object for calling the language model.
        report_id: Report ID for tracking LLM usage (optional).
        use_summary_only: True=use only Summary from Evidence (default True).
        show_kg: True=include existing KG in extraction prompt (default False, saves tokens).
        batch_queries: How many (query,evidence) pairs to batch per LLM call (default 5, set 1 for per-call).
        **kwargs: Additional arguments passed to LLM calls.
    """
    assert len(evidence_nodes_batch) == len(
        search_queries
    ), "Mismatch in evidence nodes batch and search queries length"
    # 1. Extract Knowledge Nodes from Evidence Nodes
    pairs = list(zip(evidence_nodes_batch, search_queries))
    for i in range(0, len(pairs), batch_queries):
        chunk = pairs[i : i + batch_queries]
        chunk_ens_list = [p[0] for p in chunk]
        chunk_queries = [p[1] for p in chunk]
        all_chunk_ens = [en for ens in chunk_ens_list for en in ens]

        if len(chunk) == 1:
            llm_output = _extract_knowledge_by_original_prompt(
                root_query=root_query,
                knowledge_graph=knowledge_graph,
                evidence_nodes=chunk_ens_list[0],
                search_query=chunk_queries[0],
                llm_model=llm_model,
                report_id=report_id,
                usage_file=usage_file,
                is_first_batch=False,  # update always has existing KG
                use_summary_only=use_summary_only,
                show_kg=show_kg,
                **kwargs,
            )
        else:
            llm_output = _extract_knowledge_by_original_prompt_batch(
                root_query=root_query,
                knowledge_graph=knowledge_graph,
                query_evidence_pairs=list(zip(chunk_queries, chunk_ens_list)),
                llm_model=llm_model,
                report_id=report_id,
                usage_file=usage_file,
                is_first_batch=False,  # update always has existing KG
                use_summary_only=use_summary_only,
                show_kg=show_kg,
                **kwargs,
            )
        knowledge_graph.add_llm_generated_knowledge(llm_output, all_chunk_ens)

    # 2. Semantic merge of KG nodes
    with open(PROMPT_LIB_DIR / "merge_knowledge_node.yaml", "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    system_prompt = yaml_data["system"]
    user_prompt = f"""Root Query: {root_query}

    Current Knowledge Graph: {knowledge_graph.to_toon()}"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            merge_knowledge_node_start_time = time.time()
            response = call_llm_model(
                llm_model=llm_model,
                messages=messages,
                temperature=0.3,
                **kwargs,
            )
            merge_knowledge_node_end_time = time.time()
            llm_output = safe_json_loads(response.content)
            if llm_output is None:
                raise ValueError("safe_json_loads returned None")
            knowledge_graph.apply_merge_node_results(llm_output)
            if report_id is not None and usage_file is not None:
                update_llm_usage(
                    response,
                    "merge_knowledge_node",
                    report_id,
                    usage_file,
                    elapsed_time=getattr(
                        response,
                        "_call_elapsed_time",
                        merge_knowledge_node_end_time - merge_knowledge_node_start_time,
                    ),
                )
            break
        except Exception:
            if attempt == max_retries - 1:
                raise
    # Semantic clustering (no predefined cluster count): embedding -> HDBSCAN -> apply_clustering_results
    try:
        semantic_clusters = _semantic_cluster_knowledge_nodes_hdbscan(
            knowledge_graph=knowledge_graph,
            embedding_model="text-embedding-3-large",
            # Tunable: higher min_cluster_size = more stable clusters, lower = more sensitive
            min_cluster_size=int(
                os.getenv("DEEPRESEARCH_HDBSCAN_MIN_CLUSTER_SIZE", "3")
            ),
            min_samples=None,
        )
        if semantic_clusters.get("clusters"):
            # Default: non-destructive (tag cluster_id + record cluster summaries for downstream LLM context)
            knowledge_graph.apply_semantic_clustering_results(semantic_clusters)
            # Optional: enable destructive merge via environment variable
            if os.getenv("DEEPRESEARCH_SEMANTIC_CLUSTER_MERGE", "").strip().lower() in {
                "1",
                "true",
                "yes",
            }:
                knowledge_graph.apply_merge_node_results(semantic_clusters)
    except Exception as e:
        logging.exception(f"[semantic_cluster] Semantic clustering failed, skipping. error={e}")

    # Community detection (graph-structure based)
    # Uses community_id (not cluster_id) to avoid overwriting semantic clustering (HDBSCAN) results
    # cluster_id = semantic clustering, community_id = graph-structure community detection
    # Select method via DEEPRESEARCH_COMMUNITY_DETECTION_METHOD env var (leiden/neo4j_gds/none)
    try:
        communities = apply_community_detection(
            knowledge_graph=knowledge_graph,
            neo4j_store=None,
        )
        if communities:
            logging.info(
                f"[community_detection] Applied community detection to {len(communities)} nodes"
            )
    except Exception as e:
        logging.exception(f"[community_detection] Community detection failed, skipping. error={e}")

    return knowledge_graph


def _log_explore_chains_diagnostic(msg: str) -> None:
    """Append explore_chains diagnostic info to explore_chains_diagnostic.txt."""
    log_path = Path(__file__).resolve().parent / "explore_chains_diagnostic.txt"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} [explore_chains diagnostic] {msg}\n"
            )
    except Exception:
        pass


def generate_search_chains_and_search_queries(
    root_query,
    outline,
    knowledge_graph: KnowledgeGraph,
    llm_model,
    visited_edges: Optional[Set[Tuple[int, int, str]]] = None,
    language="en",
    report_id: int = None,
    usage_file: str = None,
    is_use_history_search_queries=False,
    history_search_queries=None,
    kg_query_num: int = 10,
    **kwargs,
) -> Tuple[List[str], List[str], List[Chain], Set[Tuple[int, int, str]]]:
    """
    Generate explore queries based on knowledge graph and explore chains.

    Builds explore chains from the KG, then uses LLM to select chains and generate
    corresponding search queries.

    Args:
        root_query: Root query.
        outline: Current outline.
        knowledge_graph: Current knowledge graph object.
        llm_model: LLM model object.
        visited_edges: Set of already-visited edges, each as (source_node_id, target_node_id, chain_type),
                       used to avoid duplicate searches.
        language: Language for generated queries (default "en").
        report_id: Report ID for tracking LLM usage (optional).
        is_use_history_search_queries: Whether to use historical search queries (default False).
        history_search_queries: Historical search query list.
        kg_query_num: Number of search queries to generate from KG per round.
        **kwargs: Additional arguments passed to LLM calls.

    Returns:
        Tuple of (explore_chain_contents, search_queries, all_chains, updated_visited_edges).
    """
    if visited_edges is None:
        visited_edges = set()

    chains = _build_search_chains(
        knowledge_graph,
        visited_edges=visited_edges,
    )
    if len(chains) == 0 or chains == []:
        # Fallback: convert all KG edges to Chains (same format as enrich)
        node_id_to_knowledge = {
            node.id: node.knowledge for node in knowledge_graph.knowledge_nodes
        }
        fallback_chains = []
        for idx, edge in enumerate(knowledge_graph.knowledge_edges):
            src_name = node_id_to_knowledge.get(edge.source_id, "?")
            tgt_name = node_id_to_knowledge.get(edge.target_id, "?")
            chain_content = f"({src_name}) -> [{edge.relation_name}] -> ({tgt_name})"
            fallback_chains.append(
                Chain(
                    id=idx + 1,
                    type="enrich",
                    nodes=[edge.source_id, edge.target_id],
                    reason=f"Evidence count: {len(edge.evidence_nodes)}",
                    content=chain_content,
                    is_visited=False,
                )
            )
        chains = fallback_chains
        # Write diagnostic (record that fallback was triggered)
        with open(
            PROMPT_LIB_DIR / "debug_explore_chains_empty.txt", "a+", encoding="utf-8"
        ) as f:
            f.write(
                "----------------chains was empty, fallback enabled: all KG edges as chains----------------\n"
            )
            f.write(f"root_query: {root_query}\n")
            f.write(f"outline: {outline}\n")
            f.write(f"knowledge_graph: {knowledge_graph.to_toon()}\n")
            f.write(f"chains(count): {len(chains)}\n")
            f.write("--------------------------------\n")
    # print("--------------------------------")
    # chains 是 List[Chain]，避免 str + list 触发 TypeError
    # print("候选搜索链:")
    # for c in chains:
    #     print(f"- {c.id}. {c.content} -- Reason: {c.reason} -- type={c.type} -- visited={c.is_visited}")
    # print("--------------------------------")
    # 写入
    # with open(PROMPT_LIB_DIR / 'debug_explore_chains.txt', 'a+', encoding='utf-8') as f:
    #     f.write("--------------------------------\n")
    #     f.write(f"root_query: {root_query}\n")
    #     for c in chains:
    #         f.write(f"- {c.id}. {c.content} -- Reason: {c.reason} -- type={c.type} -- visited={c.is_visited}\n")
    #     f.write("--------------------------------\n")
    # Load explore chain selection prompt template (CHAIN_NUM injected from kg_query_num)
    with open(
        PROMPT_LIB_DIR / "select_explore_chains.yaml", "r", encoding="utf-8"
    ) as f:
        yaml_data = yaml.safe_load(f)
    # Render system prompt with kg_query_num as CHAIN_NUM
    system_template = Template(yaml_data["system"])
    system_prompt = system_template.safe_substitute(CHAIN_NUM=kg_query_num)
    with open(PROMPT_LIB_DIR / "debug_kg_toon.txt", "w", encoding="utf-8") as kg_file:
        kg_file.write("--------------------------------\n")
        kg_file.write(knowledge_graph.to_toon())
        kg_file.write("\n--------------------------------\n")
    # print(system_prompt)
    if is_use_history_search_queries and history_search_queries is not None:
        user_prompt = f"""Root Query: {root_query}\n
        Current Outline: \n{outline}\n
        Knowledge Graph: \n{knowledge_graph.to_toon()}\n
        History Search Queries: \n{history_search_queries}\n
        Explore Chains:
        """
    else:
        user_prompt = f"""Root Query: {root_query}\n
        Current Outline: \n{outline}\n
        Knowledge Graph: \n{knowledge_graph.to_toon()}\n
        Explore Chains:
        """
    # Append unvisited explore chains to prompt
    for chain in chains:
        if chain.is_visited:
            continue
        user_prompt += f"{chain.id}. {chain.content} -- Reason: {chain.reason}\n"
    user_prompt += f"""Generate Search Queries in {language}."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    with open(PROMPT_LIB_DIR / "debug_explore_prompt.txt", "w", encoding="utf-8") as f:
        f.write(user_prompt)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            generate_explore_queries_start_time = time.time()
            response = call_llm_model(
                llm_model=llm_model,
                messages=messages,
                temperature=0.7,  # 设置温度参数
                **kwargs,  # 传递其他参数
            )
            llm_output = safe_json_loads(response.content)
            # Diagnostic: all three failure reasons raise to trigger retry and log to explore_chains_diagnostic.txt
            if llm_output is None:
                msg = f"Reason 1: JSON parse failed, safe_json_loads returned None. raw first 200 chars: {(response.content or '')[:200]}"
                _log_explore_chains_diagnostic(msg)
                print(f"[explore_chains diagnostic] {msg}")
                raise ValueError(
                    "explore_chains diagnostic: Reason 1 - safe_json_loads returned None"
                )

            explore_chain_ids_raw = llm_output.get("chains", [])
            search_queries = llm_output.get("search queries", [])
            if not explore_chain_ids_raw or explore_chain_ids_raw == []:
                msg = f"Reason 2: LLM returned empty chains or key is not 'chains'. llm_output.keys()={list(llm_output.keys() if hasattr(llm_output, 'keys') else [])} chains={explore_chain_ids_raw}"
                _log_explore_chains_diagnostic(msg)
                print(f"[explore_chains diagnostic] {msg}")
                raise ValueError(
                    "explore_chains diagnostic: Reason 2 - chains empty or key is not 'chains'"
                )

            # Fallback: normalize to int, strip prefix, only keep IDs present in candidate chains
            valid_chain_ids = {c.id for c in chains}

            def _normalize_chain_id(x):
                if isinstance(x, int):
                    return x
                if isinstance(x, str):
                    s = x.strip()
                    if (
                        len(s) > 1
                        and (s[0] == "c" or s[0] == "C")
                        and s[1:].strip().lstrip("-").isdigit()
                    ):
                        s = s[1:].strip()
                    try:
                        return int(s)
                    except ValueError:
                        return None
                return None

            explore_chain_ids = []
            for x in explore_chain_ids_raw:
                cid = _normalize_chain_id(x)
                if cid is not None and cid in valid_chain_ids:
                    explore_chain_ids.append(cid)
            explore_chain_ids = list(dict.fromkeys(explore_chain_ids))  # 去重保序

            if not explore_chain_ids:
                chain_ids_in_kg = [c.id for c in chains]
                msg = f"Reason 3: LLM returned chain IDs don't match candidate chain IDs (no valid IDs after normalization).\n raw={explore_chain_ids_raw} types={[type(x).__name__ for x in explore_chain_ids_raw]} candidate chain.id={chain_ids_in_kg[:20]}{'...' if len(chain_ids_in_kg) > 20 else ''}"
                msg += f"\n chain info: {chains}"
                _log_explore_chains_diagnostic(msg)
                print(f"[explore_chains diagnostic] {msg}")
                raise ValueError(
                    "explore_chains diagnostic: Reason 3 - chain IDs don't match candidate chain IDs"
                )

            # Update visited edges (mark selected chains as visited)
            updated_visited_edges = set(visited_edges)
            for chain in chains:
                if chain.id in explore_chain_ids:
                    chain.is_visited = True
                    # Build unique key from chain type and nodes
                    if len(chain.nodes) >= 2:
                        if len(chain.nodes) == 2:
                            chain_key = (chain.nodes[0], chain.nodes[1], chain.type)
                        else:
                            # For multi-node chains, use first and last nodes
                            chain_key = (chain.nodes[0], chain.nodes[-1], chain.type)
                        updated_visited_edges.add(chain_key)

            # Extract chain contents for selected chain IDs
            this_time_search_chains = [
                chain.content for chain in chains if chain.id in explore_chain_ids
            ]

            generate_explore_queries_end_time = time.time()
            if report_id is not None and usage_file is not None:
                update_llm_usage(
                    response,
                    "generate_explore_queries",
                    report_id,
                    usage_file,
                    elapsed_time=getattr(
                        response,
                        "_call_elapsed_time",
                        generate_explore_queries_end_time
                        - generate_explore_queries_start_time,
                    ),
                )
            # Success, break retry loop
            break
        except Exception:
            if attempt == max_retries - 1:
                raise

    return this_time_search_chains, search_queries, chains, updated_visited_edges
