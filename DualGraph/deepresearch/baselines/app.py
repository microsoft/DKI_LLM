"""DualGraph DeepResearch Web UI — Chainlit Application

Usage:
    cd deepresearch/baselines
    python -m chainlit run app.py -w
"""

import sys
import os
import re
import json
import time
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import chainlit as cl

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_BASELINES_DIR = str(Path(__file__).resolve().parent)
if _BASELINES_DIR not in sys.path:
    sys.path.insert(0, _BASELINES_DIR)

_UTILS_DIR = str(Path(__file__).resolve().parent.parent / "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from main import (
    process_single_report_og_kg,
    RunConfig,
    load_env_or_explain,
    env_str,
    env_bool,
    env_float,
)
from llm_utils import AgentConfig, get_llm_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEMO_PRODUCT_DIR = str(
    Path(__file__).resolve().parent / "example_outputs" / "products"
)
_DEMO_REPORT_DIR = str(
    Path(__file__).resolve().parent / "example_outputs" / "reports"
)
_DEMO_CASE_ID = 1

# ---------------------------------------------------------------------------
# LLM model singleton
# ---------------------------------------------------------------------------
_llm_model = None
_llm_model_lock = threading.Lock()


def _get_or_create_llm_model():
    global _llm_model
    if _llm_model is not None:
        return _llm_model
    with _llm_model_lock:
        if _llm_model is not None:
            return _llm_model
        load_env_or_explain()
        provider = env_str("LLM_PROVIDER", "openai")
        raw_base_url = env_str("LLM_BASE_URL") or env_str("AZURE_OPENAI_ENDPOINT")
        base_url = raw_base_url if raw_base_url else "https://api.openai.com"
        # Azure-specific: strip trailing /openai so the SDK can rebuild it
        if provider == "azure_openai":
            if base_url.rstrip("/").lower().endswith("/openai"):
                base_url = base_url[: -len("/openai")]
            base_url = base_url.rstrip("/")
        llm_api_key = env_str("LLM_API_KEY")
        if llm_api_key == "":
            llm_api_key = None
        model_name = env_str("LLM_MODEL_NAME", "gpt-4.1-20250414")
        conf = AgentConfig(
            llm_provider=provider,
            llm_model_name=model_name,
            llm_api_key=llm_api_key,
            llm_base_url=base_url,
            llm_temperature=env_float("LLM_TEMPERATURE", 0.0),
            llm_use_aad=env_bool("AZURE_OPENAI_USE_AAD", False),
            azure_client_id=env_str("AZURE_CLIENT_ID"),
            azure_client_secret=env_str("AZURE_CLIENT_SECRET"),
            azure_endpoint=base_url,
        )
        _llm_model = get_llm_model(conf=conf)
        return _llm_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    """Return current timestamp string like [2026-04-05 14:23:01]."""
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


_REF_HEADER_RE = re.compile(
    r"^\s*##\s*(References|参考文献)\s*:?\s*$", re.IGNORECASE | re.MULTILINE
)


def _parse_references(report_text: str) -> List[Dict[str, str]]:
    match = _REF_HEADER_RE.search(report_text)
    if not match:
        return []
    refs_text = report_text[match.end() :].strip()
    references = []
    for line in refs_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\[(\d+)\]\s*(.+)$", line)
        if m:
            num = m.group(1)
            content = m.group(2).strip()
            url_m = re.search(r"https?://[^\s\]]+", content)
            url = url_m.group(0) if url_m else ""
            title = content[: url_m.start()].rstrip(" -") if url_m else content
            if not title:
                title = content
            references.append({"num": num, "title": title, "url": url})
    return references


def _strip_references_section(report_text: str) -> str:
    match = _REF_HEADER_RE.search(report_text)
    if match:
        return report_text[: match.start()].rstrip()
    return report_text


def _outline_to_markdown(outline_text: str) -> str:
    """Convert raw outline text to readable Markdown."""
    lines = outline_text.strip().splitlines()
    if not lines:
        return ""
    md_parts = []
    # First line is the title
    title = lines[0].strip()
    md_parts.append(f"## {title}\n")
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        # Convert <citation> tags to readable markdown format
        stripped = re.sub(
            r"\s*<citation>(.*?)</citation>",
            r" `[\1]`",
            stripped,
        )
        # Detect heading level by numbering pattern
        if re.match(r"^\d+\.\d+\.\d+[\.\s]", stripped):
            md_parts.append(f"#### {stripped}")
        elif re.match(r"^\d+\.\d+[\.\s]", stripped):
            md_parts.append(f"### {stripped}")
        elif re.match(r"^\d+[\.\s]", stripped):
            md_parts.append(f"## {stripped}")
        elif re.match(r"^[a-z]\.", stripped):
            # Content summary items (a. b. c. ...)
            md_parts.append(f"  - {stripped[2:].strip()}")
        else:
            md_parts.append(stripped)
    return "\n".join(md_parts)


def _find_max_iter(product_dir: str, case_id: int) -> int:
    """Find the highest iteration number available for a case."""
    max_found = -1
    for i in range(20):
        path = os.path.join(product_dir, f"outline_case{case_id}_iter{i}.txt")
        if os.path.exists(path):
            max_found = i
    return max_found


# ---------------------------------------------------------------------------
# Chainlit handlers
# ---------------------------------------------------------------------------


@cl.on_chat_start
async def on_start():
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="language",
                label="Language",
                values=["English", "Chinese"],
                initial_value="English",
            ),
            cl.input_widget.Slider(
                id="max_iter",
                label="Max Iterations",
                initial=5,
                min=1,
                max=10,
                step=1,
            ),
            cl.input_widget.Select(
                id="search_provider",
                label="Search Provider",
                values=["bing", "serper"],
                initial_value=env_str("SEARCH_PROVIDER", "serper") or "serper",
            ),
            cl.input_widget.Slider(
                id="kg_query_num",
                label="KG Query Count (per iter)",
                initial=10,
                min=1,
                max=30,
                step=1,
            ),
            cl.input_widget.Slider(
                id="og_query_num",
                label="OG Query Count (per iter)",
                initial=10,
                min=1,
                max=30,
                step=1,
            ),
            cl.input_widget.Select(
                id="readpage_method",
                label="ReadPage Method",
                values=["firecrawl", "jina"],
                initial_value=env_str("READPAGE_METHOD", "firecrawl") or "firecrawl",
            ),
            cl.input_widget.Select(
                id="enable_semantic_clustering",
                label="Semantic Clustering",
                values=["true", "false"],
                initial_value="true",
            ),
            cl.input_widget.Select(
                id="community_detection",
                label="Community Detection",
                values=["true", "false"],
                initial_value="true",
            ),
        ]
    ).send()
    _save_settings(settings)

    await cl.Message(
        content=(
            f"{_ts()} Welcome to **DualGraph DeepResearch**!\n\n"
            "Enter a research question to generate a report, "
            "or type **`demo`** to see a pre-generated example (no LLM calls).\n\n"
            "Adjust **Language** and **Max Iterations** in settings (gear icon)."
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    _save_settings(settings)


def _save_settings(settings):
    """Persist all settings to user session."""
    cl.user_session.set("language", settings.get("language", "English"))
    cl.user_session.set("max_iter", int(settings.get("max_iter", 5)))
    cl.user_session.set("search_provider", settings.get("search_provider", "serper"))
    cl.user_session.set("kg_query_num", int(settings.get("kg_query_num", 10)))
    cl.user_session.set("og_query_num", int(settings.get("og_query_num", 10)))
    cl.user_session.set("readpage_method", settings.get("readpage_method", "firecrawl"))
    cl.user_session.set("enable_semantic_clustering", settings.get("enable_semantic_clustering", "true"))
    cl.user_session.set("community_detection", settings.get("community_detection", "true"))


@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()
    if not query:
        await cl.Message(content=f"{_ts()} Please enter a research question.").send()
        return

    if query.lower() == "demo":
        await _run_demo()
    else:
        await _run_research(query)


# ---------------------------------------------------------------------------
# Demo mode — replay case 1 data with no LLM calls
# ---------------------------------------------------------------------------
async def _run_demo():
    """Replay case 1 output files as a simulated research run."""
    case_id = _DEMO_CASE_ID
    product_dir = _DEMO_PRODUCT_DIR
    report_dir = _DEMO_REPORT_DIR

    # Check data exists
    if not os.path.isdir(product_dir):
        await cl.Message(
            content=f"{_ts()} Demo data not found at:\n`{product_dir}`\n\nPlease make sure the example_outputs directory exists."
        ).send()
        return

    max_iter = _find_max_iter(product_dir, case_id)
    if max_iter < 0:
        await cl.Message(content=f"{_ts()} No iteration data found for demo case.").send()
        return

    await cl.Message(
        content=f"{_ts()} **Demo Mode** — Replaying Case {case_id} ({max_iter + 1} iterations)\n\nNo LLM calls will be made."
    ).send()

    # Show each iteration's outline
    for i in range(max_iter + 1):
        outline_path = os.path.join(product_dir, f"outline_case{case_id}_iter{i}.txt")
        if not os.path.exists(outline_path):
            continue

        with open(outline_path, "r", encoding="utf-8") as f:
            outline_text = f.read()

        outline_md = _outline_to_markdown(outline_text)

        async with cl.Step(name=f"{_ts()} Iteration {i} — Outline") as step:
            step.output = outline_md

        # Simulate processing time
        await asyncio.sleep(1)

        # Show search query count for this iteration
        sq_path = os.path.join(product_dir, f"sq_case{case_id}_iter{i}.txt")
        if os.path.exists(sq_path):
            with open(sq_path, "r", encoding="utf-8") as f:
                sq_lines = [l.strip() for l in f if l.strip()]
            async with cl.Step(name=f"{_ts()} Iteration {i} — Search Queries ({len(sq_lines)})") as step:
                step.output = "\n".join(f"- {q}" for q in sq_lines[:10])
                if len(sq_lines) > 10:
                    step.output += f"\n- ... and {len(sq_lines) - 10} more"

        await asyncio.sleep(0.5)

    # Show final report
    report_path = os.path.join(report_dir, f"{case_id}.md")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()

        report_body = _strip_references_section(report_text)
        await cl.Message(content=f"{_ts()} **Final Report**\n\n{report_body}").send()

        # References
        refs = _parse_references(report_text)
        if refs:
            ref_lines = []
            for ref in refs:
                if ref["url"]:
                    ref_lines.append(f"[{ref['num']}] [{ref['title']}]({ref['url']})")
                else:
                    ref_lines.append(f"[{ref['num']}] {ref['title']}")
            await cl.Message(
                content=f"{_ts()} ## References ({len(refs)})\n\n" + "\n\n".join(ref_lines)
            ).send()
    else:
        await cl.Message(content=f"{_ts()} Report file not found for demo case.").send()


# ---------------------------------------------------------------------------
# Real research run
# ---------------------------------------------------------------------------
async def _run_research(query: str):
    language = cl.user_session.get("language", "English")
    max_iter = cl.user_session.get("max_iter", 5)
    search_provider = cl.user_session.get("search_provider", "serper")
    kg_query_num = cl.user_session.get("kg_query_num", 10)
    og_query_num = cl.user_session.get("og_query_num", 10)
    readpage_method = cl.user_session.get("readpage_method", "firecrawl")
    enable_sc = cl.user_session.get("enable_semantic_clustering", "true") == "true"
    enable_cd = cl.user_session.get("community_detection", "true") == "true"

    # Push env overrides so downstream modules pick them up
    os.environ["READPAGE_METHOD"] = readpage_method
    os.environ["DEEPRESEARCH_ENABLE_SEMANTIC_CLUSTERING"] = str(enable_sc).lower()
    os.environ["DEEPRESEARCH_COMMUNITY_DETECTION_ENABLED"] = str(enable_cd).lower()

    report_id = int(time.time())

    # Prepare output directories
    base = Path(_BASELINES_DIR) / "web_runs"
    report_dir = str(base / "reports")
    product_dir = str(base / "products" / str(report_id))
    usage_file = str(base / "usage" / f"{report_id}.json")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(product_dir, exist_ok=True)
    os.makedirs(str(base / "usage"), exist_ok=True)
    with open(usage_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    # Initialize LLM
    async with cl.Step(name=f"{_ts()} Initializing LLM model") as step:
        try:
            llm_model = await asyncio.to_thread(_get_or_create_llm_model)
            step.output = "LLM model ready."
        except Exception as e:
            step.output = f"Failed: {e}"
            await cl.Message(content=f"{_ts()} Failed to initialize LLM:\n```\n{e}\n```").send()
            return

    cfg = RunConfig(
        kg_query_num=kg_query_num,
        og_query_num=og_query_num,
        og_query_num_in_og_only=20,
        enable_enrich_chain=True,
        enable_explore_chain=True,
        only_entity_concept_explore_chains=False,
        use_lightrag_kg=False,
        search_provider=search_provider,
    )

    start_time = time.time()
    _SPINNER = ["◐", "◓", "◑", "◒"]

    # Progress shown as a Step so it stays below outlines
    progress_step = cl.Step(name=f"{_ts()} Research in progress")
    await progress_step.send()

    # Launch blocking call (early stopping enabled by default)
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(
        None,
        lambda: process_single_report_og_kg(
            report_id=report_id,
            root_query=query,
            report_dir=report_dir,
            product_dir=product_dir,
            language=language,
            llm_model=llm_model,
            usage_file=usage_file,
            max_iter=max_iter,
            cfg=cfg,
            disable_early_stopping=False,
        ),
    )

    # Poll progress — show outlines as they appear
    last_iter_shown = -1
    tick = 0
    while not future.done():
        await asyncio.sleep(3)
        elapsed = time.time() - start_time
        mins, secs = int(elapsed) // 60, int(elapsed) % 60
        spinner = _SPINNER[tick % len(_SPINNER)]
        tick += 1

        current_max = _find_max_iter(product_dir, report_id)

        # Show new outlines (0-based)
        for i in range(last_iter_shown + 1, current_max + 1):
            outline_path = os.path.join(
                product_dir, f"outline_case{report_id}_iter{i}.txt"
            )
            if os.path.exists(outline_path):
                with open(outline_path, "r", encoding="utf-8") as f:
                    outline_text = f.read()
                outline_md = _outline_to_markdown(outline_text)
                async with cl.Step(name=f"{_ts()} Iteration {i} — Outline") as step:
                    step.output = outline_md
                last_iter_shown = i

        # Update progress with spinner (0-based, capped at max_iter - 1)
        if current_max < 0:
            phase = "Generating initial outline"
        else:
            display_iter = min(current_max + 1, max_iter - 1)
            phase = f"Iteration {display_iter}/{max_iter} — Searching & building knowledge graph"
        progress_step.name = (
            f"{_ts()} {spinner} {phase} | elapsed {mins}m {secs}s"
        )
        await progress_step.update()

    # Get result
    elapsed = time.time() - start_time
    mins, secs = int(elapsed) // 60, int(elapsed) % 60

    try:
        result = future.result()
    except Exception as e:
        progress_step.name = f"{_ts()} Research failed after {mins}m {secs}s"
        progress_step.output = f"```\n{e}\n```"
        await progress_step.update()
        return

    if not result.get("success"):
        error = result.get("error", "Unknown error")
        progress_step.name = f"{_ts()} Research failed after {mins}m {secs}s"
        progress_step.output = f"```\n{error}\n```"
        await progress_step.update()
        return

    progress_step.name = f"{_ts()} Research completed in {mins}m {secs}s"
    await progress_step.update()

    # Show any remaining outlines (0-based)
    final_max = _find_max_iter(product_dir, report_id)
    for i in range(last_iter_shown + 1, final_max + 1):
        outline_path = os.path.join(
            product_dir, f"outline_case{report_id}_iter{i}.txt"
        )
        if os.path.exists(outline_path):
            with open(outline_path, "r", encoding="utf-8") as f:
                outline_text = f.read()
            outline_md = _outline_to_markdown(outline_text)
            async with cl.Step(name=f"{_ts()} Iteration {i} — Outline") as step:
                step.output = outline_md

    # Read and display report
    report_path = os.path.join(report_dir, f"{report_id}.md")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
    else:
        report_text = "*Report file not found.*"

    report_body = _strip_references_section(report_text)
    await cl.Message(content=f"{_ts()} **Final Report**\n\n{report_body}").send()

    # References
    refs = _parse_references(report_text)
    if refs:
        ref_lines = []
        for ref in refs:
            if ref["url"]:
                ref_lines.append(f"[{ref['num']}] [{ref['title']}]({ref['url']})")
            else:
                ref_lines.append(f"[{ref['num']}] {ref['title']}")
        await cl.Message(
            content=f"{_ts()} ## References ({len(refs)})\n\n" + "\n\n".join(ref_lines)
        ).send()
