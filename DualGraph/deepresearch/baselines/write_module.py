
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import time
import re
import yaml
from typing import Any, Dict, List, Tuple, Set
from collections import OrderedDict
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from llm_utils import AgentConfig, call_llm_model, get_llm_model
from utils_module import update_llm_usage
from data_model import *

# 参考文献标题的英文和中文模式
REFERENCE_PATTERN_EN = "## References"
REFERENCE_PATTERN_ZH = "## 参考文献"

BASE = Path(__file__).resolve().parent.parent
PROMPT_LIB_DIR = Path(__file__).resolve().parent / "prompt_lib"


def dedup_and_renumber(report_text: str, language: str) -> str:
    """
    Deduplicate and renumber references in a report.
    1. Remove duplicate references
    2. Renumber references
    3. Update citation numbers in the body text

    Args:
        report_text: Original report text.
        language: Language type ('en'/'english' or other).

    Returns:
        Processed report text.
    """
    report_text = report_text.replace('id_', '')
    # 1) Split body and references section
    ref_header_re = re.compile(r'^\s*##\s*(References|参考文献)\s*:?\s*$', re.IGNORECASE | re.MULTILINE)
    header_match = ref_header_re.search(report_text)
    if not header_match:
        raise ValueError("'## References' or '## 参考文献' section not found")
    ref_start = header_match.start()
    ref_body_start = header_match.end()
    body = report_text[:ref_start].rstrip()
    refs_text = report_text[ref_body_start:].strip()

    # 2) Parse reference lines (format: “[12] content”)
    ref_lines = []
    for line in refs_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^\[(\d+)\]\s*(.+)$', line)
        if m:
            num = int(m.group(1))
            content = m.group(2).strip()
            # Use URL as unique key; fallback to full line if no URL
            url_m = re.search(r'https?://[^\s\]]+', content)
            key = url_m.group(0) if url_m else content
            ref_lines.append((num, content, key))

    if not ref_lines:
        return body.strip()

    # 3) Deduplicate and build old->new number mapping (preserve first occurrence order)
    seen = OrderedDict()
    old_to_new = {}
    for old_num, content, key in ref_lines:
        if key not in seen:
            seen[key] = (len(seen) + 1, content)
        old_to_new[old_num] = seen[key][0]

    # 4) Rebuild references section
    new_refs_lines = [f"[{new_num}] {content}" for new_num, content in seen.values()]
    new_refs_block = f"{REFERENCE_PATTERN_EN if language.lower() in ['en', 'english'] else REFERENCE_PATTERN_ZH}:\n" + "\n".join(new_refs_lines)
    
    # 5) Rewrite bracket citations in body, deduplicate and sort
    cite_re = re.compile(r'\[(\s*\d+(?:\s*,\s*\d+)*)\]')

    def replace_cite(m):
        """Map old citation numbers to new ones, deduplicate and sort."""
        nums = [int(x.strip()) for x in m.group(1).split(',') if x.strip().isdigit()]
        new_nums = []
        seen_local = set()
        for n in nums:
            if n in old_to_new:
                nn = old_to_new[n]
                if nn not in seen_local:
                    seen_local.add(nn)
                    new_nums.append(nn)
        if not new_nums:
            return ''
        return '[' + ', '.join(map(str, sorted(new_nums))) + ']'

    new_body = cite_re.sub(replace_cite, body)

    return new_body.strip() + "\n\n" + new_refs_block



def _parse_outline_line(line: str) -> Tuple[str, Set[int]]:
    """
    Parse an outline line, returning:
    1. Cleaned line content (with <citation> tags removed)
    2. Set of all evidence IDs cited in this line

    Args:
        line: A single line from the outline.

    Returns:
        (cleaned_line, citations): Cleaned text and set of evidence IDs.
    """
    citations = set()
    citation_pattern = r'<citation>(.*?)</citation>'
    matches = re.findall(citation_pattern, line)

    for match in matches:
        id_strs = [s.strip() for s in match.split(',') if s.strip()]
        for id_str in id_strs:
            if id_str.startswith('id_'):
                try:
                    evidence_id = int(id_str.split('_')[1])
                    citations.add(evidence_id)
                except (IndexError, ValueError):
                    continue
    
    # # 移除所有<citation>标签
    cleaned_line = re.sub(citation_pattern, '', line).strip()
    # # 移除行首编号（保留内容）但保持原始结构
    # cleaned_line = re.sub(r'^[IVXLCDM0-9a-z\.\(\)]+\s+', '', cleaned_line, count=1)
    # cleaned_line = line
    
    return cleaned_line, citations

def _group_sections(outline: str) -> List[Tuple[str, Set[int]]]:
    """
    Group outline by top-level sections (numbered as 1., 2., etc.).

    Returns: [(section_outline_text, evidence_ids_set), ...]
    - 1. = top-level heading
    - 1.1, 1.2 = second-level
    - 1.1.1 = third-level
    - Lines starting with lowercase letters are semantic summaries, not section breaks.

    Args:
        outline: Complete outline text.

    Returns:
        List of sections, each containing (section_outline_text, evidence_id_set).
    """
    lines = [line.strip() for line in outline.split('\n') if line.strip()]
    if not lines:
        return []

    sections = []
    current_section = []
    current_citations = set()

    # First line is usually the report title, skip it
    for line in lines[1:]:
        # Match top-level sections: digit(s) followed by dot and space (e.g. "1. ", "2. ")
        # This won't match "1.1" since 1.1 is followed by a digit, not a space
        is_top_level = bool(re.match(r'^\d+\.\s', line))

        cleaned_line, citations = _parse_outline_line(line)

        # When hitting a new top-level section, save the previous group
        if is_top_level and current_section:
            section_text = "\n".join(current_section)
            sections.append((section_text, set(current_citations)))
            current_section = []
            current_citations = set()

        current_section.append(cleaned_line)
        current_citations.update(citations)

    # Handle the last section
    if current_section:
        section_text = "\n".join(current_section)
        sections.append((section_text, set(current_citations)))

    return sections

def write_report_by_outline(
    root_query: str = Field(description="root query of deepresearch topic"),
    outline: str = Field(default="", description="outline guiding the report structure"),
    evidences_nodes_list: EvidenceNodeList = Field(default=None, description="list of evidence nodes from knowledge graph"),
    llm_model = None,
    language: str = "en",
    report_id: int = None,
    usage_file: str = None
) -> str:
    """
    Generate the final report based on outline and evidence nodes.

    Flow:
    1. Parse outline to get report title and section structure
    2. Group by top-level sections
    3. Generate content per section (with evidence integration and context continuity)

    Args:
        root_query: Root query of the deep research topic.
        outline: Outline guiding the report structure.
        evidences_nodes_list: List of evidence nodes.
        llm_model: LLM model instance.
        language: Report language (default English).

    Returns:
        The complete generated report text.
    """
    if not outline.strip():
        return f"# Report for: {root_query}\n\nNo outline provided."

    lines = [line.strip() for line in outline.split('\n') if line.strip()]
    report_title = lines[0] if lines else f"Report on {root_query}"
    section_groups = _group_sections(outline)
    full_report = f"{report_title}\n\n"

    with open(PROMPT_LIB_DIR / 'write_by_outline.yaml', 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    system_prompt = yaml_data['system']

    for i, (section_outline, evidence_ids) in enumerate(section_groups):
        # Skip if outline ends with a References section
        if any(keyword in section_outline.lower() for keyword in [". reference", ". 参考文献", ". 引用"]):
            continue
        evidence_nodes = []
        if evidences_nodes_list and evidence_ids:
            for eid in evidence_ids:
                node = evidences_nodes_list.get_evidence_node_by_id(eid)
                if node:
                    evidence_nodes.append(node)

        evidence_text = ""
        if evidence_nodes:
            evidence_text = "SUPPORTING EVIDENCE:\n"
            for node in evidence_nodes:
                evidence_text += (
                    f"- ID: id_{node.id}\n"
                    f"- Title: {node.source_title}\n"
                    f"  Content: {node.content}\n\n"
                )

        user_prompt = (
            f"REPORT TITLE: {report_title}\n\n"
            f"ROOT QUERY: {root_query}\n\n"
        )

        if full_report.strip():
            user_prompt += f"PREVIOUS CONTENT:\n{full_report}\n\n"
        
        user_prompt += (
            f"CURRENT SECTION OUTLINE:\n{section_outline}\n\n"
            f"{evidence_text}"
            "Write the full content for this section. Integrate evidence naturally. "
            "Connect logically to previous sections. Output ONLY the section content."
            f"Language: {language}"
        )

        write_report_by_outline_start_time = time.time()
        response = call_llm_model(
            llm_model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            num_retry=3,
        )
        write_report_by_outline_end_time = time.time()
        if report_id is not None and usage_file is not None:
            update_llm_usage(response, "write_report_by_outline", report_id, usage_file, elapsed_time=getattr(response, '_call_elapsed_time', write_report_by_outline_end_time - write_report_by_outline_start_time))
        section_content = response.content.strip()

        full_report += f"{section_content}\n\n"

    # Add References (only cited in this outline)
    cited_ids = set()
    for _, evidence_ids in section_groups:
        cited_ids.update(evidence_ids)
    evidence_dict = {node.id: node for node in evidences_nodes_list.evidence_nodes if node.id in cited_ids}
    references_text = f"{REFERENCE_PATTERN_EN if language.lower() in ["en", "english"] else REFERENCE_PATTERN_ZH}:\n"
    for id in sorted(evidence_dict.keys()):
        node = evidence_dict[id]
        references_text += f"[{node.id}] {node.source_title} - {node.source_url}\n"
    full_report += references_text

    return dedup_and_renumber(full_report, language=language)

def write_report_by_outline_kg(
    root_query: str = Field(description="root query of deepresearch topic"),
    outline: str = Field(default="", description="outline guiding the report structure"),
    knowledge_graph: KnowledgeGraph = Field(default=None, description="knowledge graph of searched evidences"),
    llm_model = None,
    language: str = "en",
    report_id: int = None,
    usage_file: str = None,
    is_use_kg_to_write_report: bool = False,
) -> str:
    """
    Generate the final report based on outline and knowledge graph.

    Flow:
    1. Parse outline to get report title and section structure
    2. Group by top-level sections
    3. Generate content per section (with evidence integration and context continuity)

    Args:
        root_query: Root query of the deep research topic.
        outline: Outline guiding the report structure.
        knowledge_graph: Knowledge graph of searched evidences.
        llm_model: LLM model instance.
        language: Report language (default English).

    Returns:
        The complete generated report text.
    """
    if not outline.strip():
        return f"# Report for: {root_query}\n\nNo outline provided."

    lines = [line.strip() for line in outline.split('\n') if line.strip()]
    report_title = lines[0] if lines else f"Report on {root_query}"
    section_groups = _group_sections(outline)
    full_report = f"{report_title}\n\n"

    # Select prompt file based on is_use_kg_to_write_report
    if is_use_kg_to_write_report:
        with open(PROMPT_LIB_DIR / 'write_by_outline_kg.yaml', 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        system_prompt = yaml_data['system']
    else:
        with open(PROMPT_LIB_DIR / 'write_by_outline.yaml', 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        system_prompt = yaml_data['system']
    
    for i, (section_outline, evidence_ids) in enumerate(section_groups):
        # Skip if outline ends with a References section
        if any(keyword in section_outline.lower() for keyword in [". reference", ". 参考文献", ". 引用"]):
            continue
        evidence_nodes = []
        if knowledge_graph and evidence_ids:
            for eid in evidence_ids:
                node = knowledge_graph.get_evidence_node_by_id(eid)
                if node:
                    evidence_nodes.append(node)
        evidence_text = ""
        if evidence_nodes:
            evidence_text = "SUPPORTING EVIDENCE:\n"
            for node in evidence_nodes:
                # Select ID format based on is_use_kg_to_write_report
                if is_use_kg_to_write_report:
                    node_id_str = f"{node.id}"
                else:
                    node_id_str = f"id_{node.id}"
                evidence_text += (
                    f"- ID: {node_id_str}\n"
                    f"- Title: {node.source_title}\n"
                    f"  Content: {node.content}\n\n"
                )
        
        # Build user prompt with different format depending on is_use_kg_to_write_report
        if is_use_kg_to_write_report:
            user_prompt = (
                f"Report Title: {report_title}\n\n"
                f"Root Query: {root_query}\n\n"
            )

            if full_report.strip():
                user_prompt += f"Previous Content:\n{full_report}\n\n"

            user_prompt += (
                f"Current Section Outline:\n{section_outline}\n\n"
                f"Relevant Knowledge Graph:\n{knowledge_graph.to_text_for_writer(evidence_ids)}\n\n"
                f"{evidence_text}\n\n"
                "Write the full content for this section. Integrate evidence naturally. "
                "Connect logically to previous sections. Output ONLY the section content."
                f"Language: {language}"
            )
        else:
            user_prompt = (
                f"REPORT TITLE: {report_title}\n\n"
                f"ROOT QUERY: {root_query}\n\n"
            )

            if full_report.strip():
                user_prompt += f"PREVIOUS CONTENT:\n{full_report}\n\n"

            user_prompt += (
                f"CURRENT SECTION OUTLINE:\n{section_outline}\n\n"
                f"{evidence_text}"
                "Write the full content for this section. Integrate evidence naturally. "
                "Connect logically to previous sections. Output ONLY the section content."
                f"Language: {language}"
            )

        write_report_by_outline_kg_start_time = time.time()
        response = call_llm_model(
            llm_model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        write_report_by_outline_kg_end_time = time.time()
        if report_id is not None and usage_file is not None:
            update_llm_usage(response, "write_report_by_outline_kg", report_id, usage_file, elapsed_time=getattr(response, '_call_elapsed_time', write_report_by_outline_kg_end_time - write_report_by_outline_kg_start_time))
        section_content = response.content.strip()

        full_report += f"{section_content}\n\n"

    # Add References
    evidence_dict = {}
    if knowledge_graph:
        for evidence_node in knowledge_graph.evidence_nodes:
            evidence_dict[evidence_node.id] = evidence_node
    references_text = f"{REFERENCE_PATTERN_EN if language.lower() in ['en', 'english'] else REFERENCE_PATTERN_ZH}:\n"
    for id in range(1, len(evidence_dict)+1):
        if id in evidence_dict:
            node = evidence_dict[id]
            references_text += f"[{node.id}] {node.source_title} - {node.source_url}\n"
            
    full_report += references_text
    return dedup_and_renumber(full_report, language=language)


if __name__ == "__main__":
    pass