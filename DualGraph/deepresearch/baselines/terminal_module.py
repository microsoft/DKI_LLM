import sys
from llm_utils import call_llm_model, get_llm_model
from data_model import *
from utils_module import update_llm_usage
import json
import yaml
import re
import json
import time
import logging
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # .../examples/deepresearch
PROMPT_LIB_DIR = Path(__file__).resolve().parent / "prompt_lib"


def repair_json(text):
    """Fix JSON strings wrapped in markdown code blocks and escape characters."""
    # 移除 markdown 代码块标记
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    return json.loads(text)


        
judgement_criteria = [
  {
    "name": "Instruction following",
    "description": "Evaluate how well the outline follows the user’s instructions for an outline. This includes adherence to the specified topic and scope, intended audience, purpose, constraints, required sections, level of detail, tone, and any formatting or length requirements. The evaluation should also check outline-specific expectations such as a clear hierarchical structure (e.g., H1/H2/H3 or bullet levels), logical ordering, consistent granularity across sections, correct numbering if requested, and inclusion of all required components (e.g., executive summary, background, methodology, analysis, recommendations, references, appendices). Penalize missing required elements, inclusion of prohibited items, incorrect scope or level of detail, or deviation from the requested format."
  },
  {
    "name": "Depth",
    "description": "Assess the comprehensiveness and analytical depth of the outline. High-depth outlines move beyond broad headings to include specific subpoints, key arguments, mechanisms or causal drivers, assumptions and uncertainties, methods to be used, metrics, and success criteria. They indicate sequencing and logic (what builds on what), note dependencies and open questions, and identify where evidence, examples, and visuals will be integrated. Shallow outlines list generic topics without meaningful substructure, rationale, or analytical scaffolding."
  },
  {
    "name": "Balance",
    "description": "Evaluate the fairness and objectivity of the outline. Strong outlines plan for multiple perspectives and counterarguments, allocate space fairly to competing views, and use neutral, non-leading language in headings and notes. Where issues are controversial or multi-faceted, the outline should explicitly include sections for trade-offs, limitations, and counter-evidence. Poor outlines display bias, give disproportionate space to one side without justification, or omit salient opposing views."
  },
  {
    "name": "Breadth",
    "description": "Evaluate how many distinct and relevant subtopics, perspectives, or contexts the outline covers, while staying focused on the brief. Excellent outlines include appropriate dimensions such as historical context, legal or regulatory considerations, economic or market factors, technical or operational aspects, ethical implications, social or cultural impacts, geographic or comparative analysis, stakeholder perspectives, risks and limitations, and implementation pathways. Coverage should be wide-ranging yet purposeful; simply presenting two sides of a debate is insufficient, and irrelevant tangents should be avoided."
  },
  {
    "name": "Support",
    "description": "Evaluate the outline’s evidentiary scaffolding and sourcing plan. Providing source URLs somewhere in the outline (for example, in a references section or via inline citations) is the minimum requirement; if no section provides source URLs, the score must be zero. Factual accuracy is necessary but not sufficient. Higher-quality outlines explicitly attribute planned factual claims to verifiable sources (such as peer-reviewed articles, government databases, or reputable news organizations) with traceable citations including author or outlet, date, and URL. Quantitative points specify concrete datasets or reports, time frames, and comparative benchmarks. Qualitative points identify concrete examples or case studies, clearly linked to the argument, with sources. Sources should be credible and balanced; cherry-picking or omission of clearly relevant counter-evidence is penalized. Original synthesis should build on cited material, not replace it."
  },
  {
    "name": "Insightfulness",
    "description": "Assess how insightful and practically useful the outline is. Excellent outlines go beyond common templates by offering original structure or framing, highlighting non-obvious but relevant connections, and sequencing sections to surface key insights efficiently. Recommendations and proposed analyses are concrete and actionable, clearly indicating what will be done, where it will appear, and how outcomes will be measured. Strong outlines call out specific real-world examples or comparator cases (who did what, when, what outcomes were observed, and how they were measured) and propose suitable exhibits such as tables, charts, or frameworks with a clear analytical purpose. Vague, generic, or purely aspirational notes cannot score highly."
  }
]


PASS_THRESHOLDS_HIGH = {
    'Depth': 9,
    'Balance': 9,
    'Breadth': 9,
    'Support': 5,
    'Insightfulness': 8,
    'Instruction following': 9,
}

PASS_THRESHOLDS_LOW = {
    'Depth': 8,
    'Balance': 8,
    'Breadth': 8,
    'Support': 5,
    'Insightfulness': 8,
    'Instruction following': 8,
}


def judge_terminal_by_outline(root_query, outline: str, llm_model, report_id: int = None, usage_file: str = None, disable_early_stopping: bool = False, threshold_level: str = "low") -> str:
    if disable_early_stopping == True:
        # Early stopping disabled, return False to let the model continue
        return False

    # 从 yaml 加载 prompt 模板
    with open(PROMPT_LIB_DIR / 'judge_terminal_by_outline.yaml', 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    system_prompt = yaml_data['system']

    # 构建所有维度的评分标准描述
    criteria_text = "\n".join(
        f"- **{c['name']}**: {c['description']}"
        for c in judgement_criteria
    )
    criteria_names = [c['name'] for c in judgement_criteria]

    user_prompt = yaml_data['user'].replace(
        '{criteria_text}', criteria_text
    ).replace(
        '{root_query}', root_query
    ).replace(
        '{outline}', outline
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    judge_terminal_start_time = time.time()
    response = call_llm_model(
        llm_model=llm_model,
        messages=messages,
        temperature=0.7,
    )
    judge_terminal_end_time = time.time()
    if report_id is not None and usage_file is not None:
        update_llm_usage(response, "judge_terminal_by_outline", report_id, usage_file, elapsed_time=getattr(response, '_call_elapsed_time', judge_terminal_end_time - judge_terminal_start_time))

    # 解析响应，失败则重试
    ratings = {}
    parsed = False
    for attempt in range(4):  # 1 initial + 3 retries
        try:
            response_json = repair_json(response.content)
            for name in criteria_names:
                ratings[name] = int(response_json[name]['rating'])
            parsed = True
            break
        except Exception:
            if attempt < 3:
                retry_start_time = time.time()
                response = call_llm_model(
                    llm_model=llm_model,
                    messages=messages,
                    temperature=0.7,
                )
                retry_end_time = time.time()
                if report_id is not None and usage_file is not None:
                    update_llm_usage(response, "judge_terminal_by_outline", report_id, usage_file, elapsed_time=getattr(response, '_call_elapsed_time', retry_end_time - retry_start_time))

    if not parsed:
        # All retries failed, score all dimensions as 0
        for name in criteria_names:
            ratings[name] = 0

    pass_thresholds = PASS_THRESHOLDS_HIGH if threshold_level == "high" else PASS_THRESHOLDS_LOW
    return all(ratings.get(name, 0) >= threshold for name, threshold in pass_thresholds.items())
