"""
Outline Module

Provides core functionality for creating and updating research outlines:
- Create initial outline based on root query
- Update outline with new evidence
- Update outline using knowledge graph
- Generate search queries from outline

Interacts with LLM models using YAML-formatted prompt templates.
"""


from string import Template
from llm_utils import call_llm_model
from data_model import *
import yaml
from typing import List, Optional
import time
from pathlib import Path
from utils_module import update_llm_usage
BASE = Path(__file__).resolve().parent.parent
PROMPT_LIB_DIR = Path(__file__).resolve().parent / "prompt_lib"


def create_outline(
    root_query: str,
    llm_model,
    language: str,
    report_id: int = None,
    usage_file: str = None,
) -> SkeletonGraph:
    """
    Create an initial research outline based on the root query.

    Args:
        root_query: The core research question or topic.
        llm_model: LLM model instance for generating the outline.
        language: Output language for the outline.
        report_id: Report ID for tracking LLM usage (optional).

    Returns:
        SkeletonGraph: The generated outline content.
    """
    with open(PROMPT_LIB_DIR / "create_outline.yaml", "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    system_prompt = yaml_data["system"]
    user_prompt = f"""Root Query: {root_query}\nLanguage: {language}"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    create_outline_start_time = time.time()
    response = call_llm_model(
        llm_model=llm_model,
        messages=messages,
        temperature=0.7,
        num_retry=3,
    )
    create_outline_end_time = time.time()

    if report_id is not None and usage_file is not None:
        update_llm_usage(
            response,
            "create_outline",
            report_id,
            usage_file,
            elapsed_time=getattr(
                response,
                "_call_elapsed_time",
                create_outline_end_time - create_outline_start_time,
            ),
        )

    return response.content


def update_outline(
    root_query: str,
    outline: str,
    evidences: List[EvidenceNode],
    llm_model,
    language: str,
    report_id: int = None,
    usage_file: str = None,
) -> str:
    """
    Update an existing outline with new evidence.

    Args:
        root_query: The core research question.
        outline: Current outline text to be updated.
        evidences: List of newly discovered evidence nodes.
        llm_model: LLM model instance for generating the updated outline.
        language: Output language for the outline.
        report_id: Report ID for tracking LLM usage (optional).

    Returns:
        str: The updated outline text.
    """
    evidence_str = "\n".join([f"id_{en.id}: {en.content}" for en in evidences])

    with open(PROMPT_LIB_DIR / "update_outline.yaml", "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    system_prompt = yaml_data["system"]
    user_prompt = f"""
    Root Query: {root_query}
    
    Current Outline: {outline}
    
    New Evidences: {evidence_str}
    
    Language: {language}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    update_outline_start_time = time.time()
    response = call_llm_model(
        llm_model=llm_model,
        messages=messages,
        temperature=0.7,
    )
    update_outline_end_time = time.time()

    if report_id is not None and usage_file is not None:
        update_llm_usage(
            response,
            "update_outline",
            report_id,
            usage_file,
            elapsed_time=getattr(
                response,
                "_call_elapsed_time",
                update_outline_end_time - update_outline_start_time,
            ),
        )

    return response.content


def update_outline_by_kg(
    root_query: str,
    outline: str,
    evidences: List[EvidenceNode],
    knowledge_graph: KnowledgeGraph,
    llm_model,
    language: str,
    report_id: int = None,
    usage_file: str = None,
) -> str:
    """
    Update outline using knowledge graph context.

    Similar to update_outline, but additionally provides complete knowledge graph
    information so the LLM can better understand relationships between evidence.

    Args:
        root_query: The core research question.
        outline: Current outline text to be updated.
        evidences: List of newly discovered evidence nodes.
        knowledge_graph: Complete knowledge graph with node and edge relationships.
        llm_model: LLM model instance for generating the updated outline.
        language: Output language for the outline.
        report_id: Report ID for tracking LLM usage (optional).

    Returns:
        str: The updated outline text.
    """
    evidence_str = "\n".join([f"id_{en.id}: {en.content}" for en in evidences])

    with open(PROMPT_LIB_DIR / "update_outline_by_kg.yaml", "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    system_prompt = yaml_data["system"]

    # Include KG JSON representation for richer context
    user_prompt = f"""
    Root Query: {root_query}
    
    Current Outline: {outline}
    
    New Evidences: {evidence_str}
    
    Current Knowledge Graph: {knowledge_graph.to_json()}
    
    Language: {language}"""

    # 构建消息列表
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 记录本次LLM调用的开始时间
    update_outline_by_kg_start_time = time.time()
    # 调用LLM模型更新大纲，温度为0.7
    response = call_llm_model(
        llm_model=llm_model,
        messages=messages,
        temperature=0.7,
    )
    update_outline_by_kg_end_time = time.time()

    # 更新LLM使用情况统计
    if report_id is not None and usage_file is not None:
        # 记录本次LLM调用的耗时
        update_llm_usage(
            response,
            "update_outline_by_kg",
            report_id,
            usage_file,
            elapsed_time=getattr(
                response,
                "_call_elapsed_time",
                update_outline_by_kg_end_time - update_outline_by_kg_start_time,
            ),
        )

    return response.content


def generate_search_queries(
    outline: str,
    history_search_queries: List[str],
    llm_model,
    language: str,
    report_id: int = None,
    usage_file: str = None,
    is_use_history_search_queries: bool = False,
    pending_search_queries: Optional[List[str]] = None,
    query_num: int = 10,
) -> List[str]:
    """
    Generate search queries based on the current outline.

    Args:
        outline: Current research outline text.
        history_search_queries: Previously executed search queries (to avoid duplication).
        llm_model: LLM model instance for generating queries.
        language: Output language for the queries.
        report_id: Report ID for tracking LLM usage (optional).
        is_use_history_search_queries: Whether to include historical queries (default False).
        pending_search_queries: Planned but not-yet-executed queries (to avoid duplication).
        query_num: Number of queries to generate per round.

    Returns:
        List[str]: Generated search query list.
    """
    with open(
        PROMPT_LIB_DIR / "generate_search_query.yaml", "r", encoding="utf-8"
    ) as f:
        yaml_data = yaml.safe_load(f)
    system_template = Template(yaml_data["system"])
    system_prompt = system_template.safe_substitute(QUERY_NUM=query_num)
    user_lines: List[str] = [f"Outline: {outline}"]
    if is_use_history_search_queries and history_search_queries is not None:
        user_lines.append(
            f"Historical Search Queries (executed):{history_search_queries}"
        )
    if pending_search_queries:
        user_lines.append(
            f"Pending Search Queries (planned, NOT executed yet):{pending_search_queries}"
        )
    user_lines.append(f"Language: {language}")
    user_prompt = "\n\n".join(user_lines)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    generate_search_queries_start_time = time.time()
    response = call_llm_model(
        llm_model=llm_model,
        messages=messages,
        temperature=0.7,
    )
    generate_search_queries_end_time = time.time()

    if report_id is not None and usage_file is not None:
        update_llm_usage(
            response,
            "generate_search_queries",
            report_id,
            usage_file,
            elapsed_time=getattr(
                response,
                "_call_elapsed_time",
                generate_search_queries_end_time - generate_search_queries_start_time,
            ),
        )

    new_search_queries = response.content.strip().split("\n")
    new_search_queries = [item for item in new_search_queries if item.strip() != ""]

    return new_search_queries
