import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import json
import os
import time
import random

import requests
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from typing import Any, Dict, List

from llm_utils import AgentConfig, call_llm_model, get_llm_model

import logging
from data_model import *

from utils_module import update_llm_usage, update_filter_stats, update_readpage_stats
import yaml

import concurrent.futures
import threading
import json_repair

BASE = Path(__file__).resolve().parent.parent  # .../examples/deepresearch
PROMPT_LIB_DIR = Path(__file__).resolve().parent / "prompt_lib"


_thread_local = threading.local()


def _get_thread_local_session():
    """
    Get the requests.Session for the current thread.
    Each thread has its own independent Session to avoid connection pool conflicts.

    Returns:
        requests.Session: The current thread's Session object.
    """
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
        # Configure connection pool for high concurrency;
        # the outer layer uses 8 threads, each needing multiple concurrent connections
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=5,  # connection pools per host (one pool per host; set higher for safety)
            pool_maxsize=20,  # max connections per pool, allowing 5 concurrent connections per thread
            max_retries=0,  # disable automatic retries; handled by upper-level code
            pool_block=True,  # TODO was off before sleep; critical: block when pool is full instead of opening new connections
        )
        _thread_local.session.mount("http://", adapter)
        _thread_local.session.mount("https://", adapter)
    return _thread_local.session


# Long-lived global thread pool so thread-local Sessions are reused across calls
URL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=24, thread_name_prefix="url_worker"
)


def _search_bing(
    query: str = Field(description="Search keywords"),
    num_results: int = Field(default=10, description="Number of results to return"),
    safe_search: bool = Field(default=True, description="Whether to enable safe search filtering"),
    language: str = Field(default="zh-CN", description="Language code for search results"),
    country: str = Field(default="CN", description="Country code for search results"),
    visited_urls: set = None,
):
    """Search the web using the Bing Search API.

    Internal function: execute a Bing search and return formatted results.

    Features:
    - Bing Search API integration
    - Supports result count, safe search, localization
    - Formatted output (JSON, Markdown, plain text)

    Args:
        query: Search keywords.
        num_results: Number of results to return (1-10).
        safe_search: Whether to enable safe search.
        language: Result language.
        country: Country code.
        visited_urls: Set of already-visited URLs for deduplication.

    Returns:
        dict: Search results and metadata.
    """
    if isinstance(query, FieldInfo):
        query = query.default
    if isinstance(num_results, FieldInfo):
        num_results = num_results.default
    if isinstance(safe_search, FieldInfo):
        safe_search = safe_search.default
    if isinstance(language, FieldInfo):
        language = language.default
    if isinstance(country, FieldInfo):
        country = country.default
    # Only used to filter already-visited URLs; this function does NOT write to visited_urls.
    # Writing to visited_urls is deferred until a URL is selected and yields valid evidence.
    if visited_urls is None:
        visited_urls = set()
    try:
        if not os.getenv("BING_APP_ID"):
            raise ValueError(
                "BING_APP_ID environment variable is not set. Please configure the Bing Search API Key in your .env file."
            )

        params = {
            "q": query,
            # "count": num_results,
            "count": 20,  # fetch more initially for deduplication
            "mkt": f"{language}-{country}",
            "safeSearch": "Strict" if safe_search else "Off",
            "appid": os.getenv("BING_APP_ID"),
        }

        start_time = time.time()
        session = _get_thread_local_session()
        response = session.get(os.getenv("BING_ENDPOINT"), params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        search_results = []
        # Deduplicate within this search call (Bing may return duplicate URLs)
        local_seen_urls = set()
        if "webPages" in data and "value" in data["webPages"]:
            for i, item in enumerate(data["webPages"]["value"]):
                url = item.get("url", "")
                title = item.get("name", "")
                snippet = item.get("snippet", "")
                if url in visited_urls:
                    continue
                if url and url in local_seen_urls:
                    continue
                if url:
                    local_seen_urls.add(url)

                search_results.append(
                    {
                        "id": f"bing-{i}",
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                    }
                )
                if len(search_results) >= num_results:
                    break

        message_content = {
            "query": query,
            "results": search_results,
            "count": len(search_results),
        }

        return message_content

    except Exception as e:
        raise RuntimeError(f"Bing Search API request failed: {str(e)}")


def _search_serper(
    query: str,
    num_results: int = 10,
    safe_search: bool = True,
    language: str = "en",
    country: str = "us",
    visited_urls: set = None,
):
    """Search the web using the Serper API (google.serper.dev).

    Requires environment variable:
      - SERPER_KEY_ID: Serper API key

    Args:
        query: Search keywords.
        num_results: Number of results to return.
        safe_search: Whether to enable safe search (not directly supported by Serper; ignored).
        language: Result language.
        country: Country code.
        visited_urls: Set of already-visited URLs for deduplication.

    Returns:
        dict: Same format as _search_bing: {"query", "results", "count"}.
    """
    if visited_urls is None:
        visited_urls = set()

    serper_key = os.getenv("SERPER_KEY_ID")
    if not serper_key:
        raise ValueError(
            "SERPER_KEY_ID environment variable is not set. Please configure the Serper API Key in your .env file."
        )

    # Auto-detect Chinese queries and switch locale
    def _contains_chinese(text: str) -> bool:
        return any("\u4E00" <= ch <= "\u9FFF" for ch in text)

    if _contains_chinese(query):
        gl, hl, location = "cn", "zh-cn", "China"
    else:
        gl, hl, location = country.lower(), language.lower(), "United States"

    payload = json.dumps({
        "q": query,
        "num": min(num_results * 2, 20),  # fetch more for deduplication
        "gl": gl,
        "hl": hl,
        "location": location,
    })
    headers = {
        "X-API-KEY": serper_key,
        "Content-Type": "application/json",
    }

    try:
        session = _get_thread_local_session()
        response = session.post(
            "https://google.serper.dev/search",
            data=payload,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        search_results = []
        local_seen_urls = set()

        for i, item in enumerate(data.get("organic", [])):
            url = item.get("link", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")

            if url in visited_urls:
                continue
            if url and url in local_seen_urls:
                continue
            if url:
                local_seen_urls.add(url)

            search_results.append({
                "id": f"serper-{i}",
                "title": title,
                "url": url,
                "snippet": snippet,
            })
            if len(search_results) >= num_results:
                break

        return {
            "query": query,
            "results": search_results,
            "count": len(search_results),
        }

    except Exception as e:
        raise RuntimeError(f"Serper API request failed: {str(e)}")


def _crawl_summarize_with_filter(
    url,
    root_query,
    query,
    llm_model,
    title="",
    snippet="",
    language="en",
    report_id: int = None,
    usage_file: str = None,
):
    """
    Crawl and process a single search result.
    Extracts content from the URL, summarizes it with an LLM, and returns relevant evidence.

    Args:
        url: The URL to visit.
        root_query: Root query for contextual understanding.
        query: The search query string.
        llm_model: LLM model object used for content summarization.
        title: Web page title (optional).
        snippet: Search result snippet (optional).
        language: Language code, default "en".
        report_id: Report ID for LLM usage tracking (optional).
        usage_file: Usage statistics file path (optional).

    Returns:
        dict: Extracted evidence content or summary.
    """
    readpage_result_type = (
        "primary_success"  # primary_success / jina_fallback_success / fail
    )
    READPAGE_METHOD = os.getenv("READPAGE_METHOD", "crawl4ai")
    JINA_FALLBACK = os.getenv("JINA_FALLBACK", "false").lower() == "true"
    primary_failed = False
    url_content = None

    try:
        if READPAGE_METHOD.lower() == "crawl4ai":
            url_content = _readpage_crawl4ai(url)
        elif READPAGE_METHOD.lower() == "jina":
            url_content = _readpage_jina(url)
        elif READPAGE_METHOD.lower() == "firecrawl":
            url_content = _readpage_firecrawl(url)
        else:
            raise ValueError(f"Invalid READPAGE_METHOD: {READPAGE_METHOD}")
        if url_content == "[visit] Failed to read page.":
            logging.warning(
                f"Reader PAGE ERROR: {READPAGE_METHOD} returned error string"
            )
            primary_failed = True
    except Exception as e:
        logging.warning(f"Reader PAGE ERROR: {READPAGE_METHOD} raised exception: {e}")
        primary_failed = True

    # If primary method failed, optionally fall back to Jina
    if primary_failed:
        if JINA_FALLBACK and READPAGE_METHOD.lower() != "jina":
            logging.info(f"Jina fallback enabled, retrying with Jina for URL: {url}")
            try:
                jina_content = _readpage_jina(url)
                if jina_content != "[visit] Failed to read page.":
                    url_content = jina_content
                    readpage_result_type = "jina_fallback_success"
                    logging.info(f"Jina fallback succeeded for URL: {url}")
                else:
                    url_content = snippet
                    readpage_result_type = "fail"
            except Exception as jina_e:
                logging.warning(
                    f"Jina fallback also failed: {jina_e}, use snippet instead"
                )
                url_content = snippet
                readpage_result_type = "fail"
        else:
            logging.warning(f"use snippet instead for URL: {url}")
            url_content = snippet
            readpage_result_type = "fail"

    update_readpage_stats(report_id, usage_file, readpage_result_type)
    summarize_result_dict = _summarize_page_by_llm(
        url_content,
        root_query,
        title,
        llm_model,
        language,
        report_id,
        usage_file,
        search_goal=query,
    )
    # print(f"Crawl result for URL: {url}:\n {summarize_result_dict}")
    return summarize_result_dict


def _select_urls_to_visit(
    search_results,
    root_query,
    query,
    llm_model,
    language="en",
    report_id: int = None,
    usage_file: str = None,
):
    """
    Use an LLM to select which URLs to visit from search results.

    Args:
        search_results: List of search results, each containing title, url, snippet.
        root_query: Root query providing overall context.
        query: The search query string.
        llm_model: LLM model object.
        language: Language code, default "en".
        report_id: Report ID for LLM usage tracking (optional).
        usage_file: Usage statistics file path (optional).

    Returns:
        list: Selected URL list, each element is a dict with url, title, and snippet,
              e.g. [{"url": "...", "title": "...", "snippet": "..."}, ...].
    """
    if not search_results:
        return []

    with open(PROMPT_LIB_DIR / "select_url_to_visit.yaml", "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    system_prompt = yaml_data["system"]

    search_results_str = ""
    for i, result in enumerate(search_results):
        search_results_str += f"{i+1}. Title: {result.get('title', 'N/A')}\n"
        search_results_str += f"   URL: {result.get('url', 'N/A')}\n"
        search_results_str += f"   Snippet: {result.get('snippet', 'N/A')}\n\n"

    user_prompt = f"""## Input
    ### root query
    {root_query}

    ### search query
    {query}

    ### search results
    {search_results_str}

    language: {language}
    """

    select_urls_to_visit_start_time = time.time()
    response = call_llm_model(
        llm_model=llm_model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,  # lower temperature for more consistent selection
    )
    select_urls_to_visit_end_time = time.time()

    if report_id is not None and usage_file is not None:
        update_llm_usage(
            response,
            "select_urls_to_visit",
            report_id,
            usage_file,
            elapsed_time=getattr(
                response,
                "_call_elapsed_time",
                select_urls_to_visit_end_time - select_urls_to_visit_start_time,
            ),
        )

    try:
        content = response.content.strip()
        if content.startswith("```"):
            start_idx = content.find("```")
            if start_idx != -1:
                content = (
                    content[start_idx:].split("\n", 1)[1]
                    if "\n" in content[start_idx:]
                    else ""
                )
                end_idx = content.rfind("```")
                if end_idx != -1:
                    content = content[:end_idx].strip()

        result_json = json_repair.loads(content)
        selected_urls = result_json.get("selected_urls", [])

        selected_urls_set = set(selected_urls)
        url_title_list = []
        for result in search_results:
            url = result.get("url")
            if url in selected_urls_set:
                url_title_list.append(
                    {
                        "url": url,
                        "title": result.get("title", "N/A"),
                        "snippet": result.get("snippet", "N/A"),
                    }
                )

        return url_title_list
    except Exception as e:
        logging.warning(
            f"Failed to parse LLM response for URL selection: {e}. Using all search results."
        )
        logging.warning(f"LLM response: {response.content}")
        return [
            {
                "url": result.get("url", ""),
                "title": result.get("title", "N/A"),
                "snippet": result.get("snippet", "N/A"),
            }
            for result in search_results
        ]


def _summarize_page_by_llm(
    page_content,
    root_query,
    title,
    llm_model,
    language="en",
    report_id: int = None,
    usage_file: str = None,
    search_goal: str = None,
):
    with open(PROMPT_LIB_DIR / "summarize_evidence.yaml", "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    system_prompt = yaml_data["system_jina"]

    def build_user_prompt(content):
        if search_goal:
            return f"""## Input
        ### root query
        {root_query}
        ### search goal
        {search_goal}
        ### web page title
        {title}
        ### web page content
        {content}

        language: {language}
        """
        else:
            return f"""## Input
        ### root query
        {root_query}
        ### web page title
        {title}
        ### web page content
        {content}

        language: {language}
        """

    # Default empty JSON result structure (consistent with normal return format)
    def get_empty_json_result():
        return {"is_useful": False, "summary": "", "evidence": "", "rational": ""}

    current_content = page_content
    max_retries = 2  # 1 initial attempt + 1 retry

    for attempt in range(max_retries):
        try:
            user_prompt = build_user_prompt(current_content)
            summarize_by_llm_v2_start_time = time.time()
            response = call_llm_model(
                llm_model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
            )
            summarize_by_llm_v2_end_time = time.time()
            # print(f"response.content:\n {response.content}")
            json_data = json_repair.loads(response.content)
            # print(f"json_data:\n {json_data}")
            # Only record usage when a valid summary result is parsed, to avoid double-counting on retries
            if (
                report_id is not None
                and usage_file is not None
                and isinstance(json_data, dict)
                and ("summary" in json_data or "evidence" in json_data)
            ):
                update_llm_usage(
                    response,
                    "summarize_by_llm",
                    report_id,
                    usage_file,
                    elapsed_time=getattr(
                        response,
                        "_call_elapsed_time",
                        summarize_by_llm_v2_end_time - summarize_by_llm_v2_start_time,
                    ),
                )
            return json_data

        except Exception as e:
            error_str = str(e)
            is_context_error = (
                "context_length_exceeded" in error_str.lower()
                or (
                    "context length" in error_str.lower()
                    and "maximum" in error_str.lower()
                )
                or "maximum context length" in error_str.lower()
                or (
                    "invalid_request_error" in error_str.lower()
                    and "context" in error_str.lower()
                    and "tokens" in error_str.lower()
                )
            )

            if is_context_error and attempt < max_retries - 1:
                import re

                max_context_match = re.search(
                    r"maximum context length is (\d+) tokens", error_str, re.IGNORECASE
                )
                if not max_context_match:
                    max_context_match = re.search(r"(\d+) tokens", error_str)

                if max_context_match:
                    max_context_tokens = int(max_context_match.group(1))
                    logging.warning(
                        f"Context length exceeded, max context: {max_context_tokens} tokens, attempting to truncate content (retry {attempt + 1}/{max_retries})"
                    )
                else:
                    max_context_tokens = 100000  # conservative default
                    logging.warning(
                        f"Context length exceeded but could not extract max context length, using default {max_context_tokens} tokens (retry {attempt + 1}/{max_retries})"
                    )

                try:
                    import tiktoken

                    model_name = (
                        getattr(llm_model, "model_name", None)
                        or getattr(llm_model, "llm_model_name", None)
                        or ""
                    )
                    if "gpt-5" in model_name.lower() or "gpt-4" in model_name.lower():
                        encoding = tiktoken.get_encoding("cl100k_base")
                    else:
                        encoding = tiktoken.get_encoding("cl100k_base")

                    system_tokens = len(encoding.encode(system_prompt))
                    template_without_content = build_user_prompt("")
                    template_tokens = len(encoding.encode(template_without_content))
                    reserved_tokens = system_tokens + template_tokens

                    available_tokens = (
                        max_context_tokens - reserved_tokens - 10000
                    )  # safety margin

                    if available_tokens <= 0:
                        logging.error(
                            f"Available tokens ({available_tokens}) too small to truncate, returning empty result"
                        )
                        return get_empty_json_result()

                    content_encoding = encoding.encode(current_content)
                    if len(content_encoding) > available_tokens:
                        truncated_encoding = content_encoding[:available_tokens]
                        current_content = encoding.decode(truncated_encoding)
                        logging.info(
                            f"Truncated page_content from {len(content_encoding)} tokens to {len(truncated_encoding)} tokens"
                        )
                    else:
                        # Content is already below available tokens but still errored; reduce further (by 20%)
                        new_available = int(available_tokens * 0.8)
                        truncated_encoding = content_encoding[:new_available]
                        current_content = encoding.decode(truncated_encoding)
                        logging.info(
                            f"Further truncated page_content to {len(truncated_encoding)} tokens (80% of available)"
                        )

                except ImportError:
                    # tiktoken not installed; estimate using character count (~1 token per 4 chars)
                    logging.warning("tiktoken not installed, using character count estimation for truncation")
                    estimated_max_chars = (
                        max_context_tokens - 5000
                    ) * 4  # safety margin of 5000 tokens
                    if len(current_content) > estimated_max_chars:
                        current_content = current_content[:estimated_max_chars]
                        logging.info(
                            f"Truncated page_content from {len(page_content)} chars to {len(current_content)} chars (rough estimate)"
                        )
                    else:
                        current_content = current_content[
                            : int(len(current_content) * 0.8)
                        ]
                        logging.info(
                            f"Further truncated page_content to {len(current_content)} chars (80%)"
                        )

                except Exception as truncate_error:
                    logging.error(f"Error while truncating content: {truncate_error}")
                    current_content = current_content[: len(current_content) // 2]
                    logging.warning(
                        f"Using simple character truncation, keeping first 50%: {len(current_content)} chars"
                    )

                continue
            else:
                if is_context_error:
                    logging.warning(
                        f"Context length exceeded, still failing after {max_retries} retries, returning empty result: {e}"
                    )
                else:
                    logging.warning(f"LLM call failed, returning empty result: {e}")
                return get_empty_json_result()

    logging.warning(f"Still failing after {max_retries} attempts, returning empty result")
    return get_empty_json_result()


def search_with_filtering_visited_urls(
    root_query: str = Field(description="The root query of the deep research"),
    query: str = Field(
        description="The search query string which needs to match the search goal which hasn't been searched"
    ),
    llm_model: any = Field(description="The LLM model to use for summarization"),
    bingsearch_num_results: int = Field(
        default=5, description="Number of search results to return (1-10, default: 5)"
    ),
    safe_search: bool = Field(
        default=True, description="Whether to enable safe search filtering"
    ),
    language: str = Field(
        default="en",
        description="Language code for search results (e.g., 'en', 'es', 'fr')",
    ),
    country: str = Field(
        default="us",
        description="Country code for search results (e.g., 'us', 'uk', 'ca')",
    ),
    visited_urls: set = None,
    report_id: int = None,
    usage_file: str = None,
    max_url_num_after_first_filter: int = 5,
    search_provider: str = "serper",
) -> List[Dict[str, Any]]:
    """
    Main search function: search the web using the configured search provider.

    Provides comprehensive web search capabilities including:
    - Search API integration (Bing / Serper)
    - Web page summarization
    - Knowledge graph updates

    Args:
        root_query: Root query of the deep research, providing overall context.
        query: Search query string matching a not-yet-searched search goal.
        llm_model: LLM model used for summarization.
        bingsearch_num_results: Number of search results to return (1-10, default 5).
        safe_search: Whether to enable safe search filtering.
        language: Language code for search results (e.g. 'en', 'es', 'fr').
        country: Country code for search results (e.g. 'us', 'uk', 'ca').
        visited_urls: Set of already-visited URLs for deduplication.

    Returns:
        List[Dict[str, Any]]: Evidence list, each element contains content, source_url, source_title.
    """

    if language.lower() in ["english", "en"]:
        language = "en"
        country = "us"
    elif language.lower() in ["chinese", "中文", "zh", "cn"]:
        language = "zh"
        country = "CN"
    else:
        language = "en"
        country = "us"
    # Important: do NOT use `if not visited_urls` which would discard an externally passed empty set reference
    if visited_urls is None:
        visited_urls = set()

    base_delay = 1.0   # base delay in seconds
    max_delay = 10.0    # max delay cap
    jitter_ratio = 0.5  # random jitter ratio: 0-50%

    _search_fn = {
        "bing": _search_bing,
        "serper": _search_serper,
    }.get(search_provider, _search_bing)

    for num_retry in range(3):
        try:
            message_content = _search_fn(
                query=query,
                num_results=bingsearch_num_results,
                safe_search=safe_search,
                language=language,
                country=country,
                visited_urls=visited_urls,
            )
            break
        except Exception as e:
            message_content = {"results": []}
            logging.warning(f"{search_provider} search failed on attempt {num_retry+1}/3: {e}")

            if num_retry < 2:
                exponential_delay = base_delay * (2**num_retry)
                exponential_delay = min(exponential_delay, max_delay)
                jitter = random.uniform(0, exponential_delay * jitter_ratio)
                total_delay = exponential_delay + jitter

                logging.info(
                    f"Waiting {total_delay:.2f}s before retry (exponential delay: {exponential_delay:.2f}s, jitter: {jitter:.2f}s)"
                )
                time.sleep(total_delay)

    search_results = message_content["results"]

    num_before_filter = len(search_results)

    url_to_visit_list = _select_urls_to_visit(
        search_results=search_results,
        root_query=root_query,
        query=query,
        llm_model=llm_model,
        language=language,
        report_id=report_id,
        usage_file=usage_file,
    )
    if len(url_to_visit_list) > max_url_num_after_first_filter:
        url_to_visit_list = url_to_visit_list[:max_url_num_after_first_filter]

    # # print("Search results before filtering:")
    # # for idx, sr in enumerate(search_results):
    # #     print(f"[{idx}] {sr['title']}")
    # # print("URLs to visit after filtering:")
    # # kept_indices = []
    # # for url_info in url_to_visit_list:
    # #     try:
    # #         idx = next(i for i, item in enumerate(search_results) if item['url'] == url_info['url'])
    # #     except StopIteration:
    # #         idx = -1
    # #     kept_indices.append(idx)
    # #     print(f"[{idx}] {url_info['title']}")
    # # all_indices = set(range(len(search_results)))
    # # kept_indices_set = set(kept_indices)
    # # filtered_indices = sorted(list(all_indices - kept_indices_set))
    # # if filtered_indices:
    # #     print("Filtered URL indices:", filtered_indices)
    num_after_filter = len(url_to_visit_list)
    filter_ratio = (
        (num_after_filter / num_before_filter * 100) if num_before_filter > 0 else 0.0
    )
    logging.info(
        f"URL first-pass filter stats - before: {num_before_filter}, after: {num_after_filter}, valid ratio: {filter_ratio:.2f}%"
    )

    if report_id is not None and usage_file is not None:
        update_filter_stats(
            report_id, usage_file, "first", num_before_filter, num_after_filter
        )

    evidence_list = []
    num_before_second_filter = len(url_to_visit_list)

    def process_url(url_info):
        """Process a single URL; return an evidence dict or None."""
        url = url_info["url"]
        source_title = url_info["title"]
        source_snippet = url_info.get("snippet", "")

        # Stagger requests to avoid all threads hitting the server simultaneously
        time.sleep(0.1 * random.random())

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 0.5 * (2**attempt)
                    logging.warning(
                        f"Retrying URL {url} (attempt {attempt + 1}/{max_retries}) after {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)

                crawl_result = _crawl_summarize_with_filter(
                    url,
                    root_query,
                    query,
                    llm_model,
                    title=source_title,
                    snippet=source_snippet,
                    language=language,
                    report_id=report_id,
                    usage_file=usage_file,
                )

                if not isinstance(crawl_result, dict):
                    error_msg = f"Unexpected return type from _crawl_summarize_with_filter for URL {url}: expected dict, got {type(crawl_result).__name__}. Value: {crawl_result}"
                    logging.warning(
                        f"{error_msg} (attempt {attempt + 1}/{max_retries})"
                    )
                    raise ValueError(error_msg)

                is_useful = crawl_result.get("is_useful", True)
                if is_useful:
                    summary = crawl_result.get("summary", "")
                    evidence = crawl_result.get("evidence", "")
                    rational = crawl_result.get("rational", "")
                    # TODO decide whether to include summary and rational based on requirements
                    # content = f"**Summary**: {summary}\n**Evidence**: {evidence}\n**Rational**: {rational}"
                    content = f"\n**Summary**: {summary}\n**Evidence**: {evidence}"

                    return {
                        "content": content,
                        "source_url": url,
                        "source_title": source_title,
                    }
                else:
                    return None
            except Exception as e:
                error_msg = str(e)

                # Page navigation errors should not be retried
                is_page_navigation_error = (
                    "Unable to retrieve content because the page is navigating"
                    in error_msg
                    or "page is navigating and changing the content" in error_msg
                    or "Page.content: Unable to retrieve content" in error_msg
                )

                if is_page_navigation_error:
                    logging.warning(
                        f"Page navigation error for URL {url}, skipping (no retry): {e}"
                    )
                    return None

                # Context length errors should not be retried
                is_context_length_error = (
                    "context_length_exceeded" in error_msg.lower()
                    or (
                        "context length" in error_msg.lower()
                        and "maximum" in error_msg.lower()
                    )
                    or "maximum context length" in error_msg.lower()
                    or (
                        "invalid_request_error" in error_msg.lower()
                        and "context" in error_msg.lower()
                        and "tokens" in error_msg.lower()
                    )
                )

                if is_context_length_error:
                    logging.warning(
                        f"Context length exceeded for URL {url}, skipping (no retry): {e}"
                    )
                    return None

                # Connection closed errors: retry at most once (2 total attempts)
                is_connection_closed_error = (
                    "ERR_CONNECTION_CLOSED" in error_msg
                    or "net::ERR_CONNECTION_CLOSED" in error_msg
                    or (
                        "Failed on navigating" in error_msg
                        and "ERR_CONNECTION_CLOSED" in error_msg
                    )
                )

                if is_connection_closed_error:
                    if attempt < 1:
                        wait_time = 1.2 + random.uniform(1, 4.2)
                        logging.warning(
                            f"Connection closed error for URL {url} (attempt {attempt + 1}/2): {e}. Retrying in {wait_time:.2f}s..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logging.warning(
                            f"Connection closed error for URL {url} after 1 retry, skipping: {e}"
                        )
                        return None

                is_connection_error = (
                    "10048" in error_msg
                    or "Connection error" in error_msg
                    or "ConnectError" in error_msg
                    or "APIConnectionError" in error_msg
                    or "WinError 10048" in error_msg
                )

                is_type_error = (
                    isinstance(e, ValueError) and "Unexpected return type" in error_msg
                )

                if attempt < max_retries - 1:
                    if is_connection_error:
                        wait_time = 2.0 * (2**attempt)  # 2s, 4s, 8s
                        logging.warning(
                            f"Connection error for URL {url} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.2f}s..."
                        )
                        time.sleep(wait_time)
                        continue
                    elif is_type_error:
                        wait_time = 0.5 * (2**attempt)  # 0.5s, 1s, 2s
                        logging.warning(
                            f"Type error for URL {url} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.2f}s..."
                        )
                        time.sleep(wait_time)
                        continue

                if isinstance(e, ValueError):
                    logging.warning(
                        f"Skipping URL due to error after {max_retries} attempts: {e}"
                    )
                else:
                    logging.warning(
                        f"Skipping URL {url} due to error after {max_retries} attempts: {e}"
                    )
                return None

        return None

    executor = URL_EXECUTOR

    # Stagger task submissions to avoid creating too many connections simultaneously
    future_to_url = {}
    for idx, url_info in enumerate(url_to_visit_list):
        if idx > 0:
            time.sleep(0.2)
        future = executor.submit(process_url, url_info)
        future_to_url[future] = url_info

    for future in concurrent.futures.as_completed(future_to_url):
        try:
            result = future.result()
            if result is not None:
                evidence_list.append(result)
                url = result.get("source_url", "")
                if url:
                    visited_urls.add(url)
        except Exception as e:
            error_msg = str(e)
            if "10048" in error_msg or "Connection error" in error_msg:
                logging.error(f"Connection error in thread pool: {e}")
            else:
                logging.error(f"Error processing URL in thread pool: {e}")

    num_after_second_filter = len(evidence_list)
    second_filter_ratio = (
        (num_after_second_filter / num_before_second_filter * 100)
        if num_before_second_filter > 0
        else 0.0
    )
    logging.info(
        f"URL second-pass filter stats - before: {num_before_second_filter}, after: {num_after_second_filter}, valid ratio: {second_filter_ratio:.2f}%"
    )
    # # remained_titles = set(result["source_title"] for result in evidence_list)
    # # kept = [(idx, url_info.get('title', '')) for idx, url_info in enumerate(url_to_visit_list) if url_info.get('title', '') in remained_titles]
    # # logging.info("Retained idx-title after second-pass filter:")
    # # for idx, title in kept:
    # #     logging.info(f"idx={idx}, title={title}")
    if report_id is not None and usage_file is not None:
        update_filter_stats(
            report_id,
            usage_file,
            "second",
            num_before_second_filter,
            num_after_second_filter,
        )

    return evidence_list


def _readpage_jina(url: str) -> str:
    """
    Read webpage content using Jina service.

    Args:
        url: The URL to read

    Returns:
        str: The webpage content or error message
    """
    max_retries = 3
    timeout = 50

    url_clean = url.strip()
    if not url_clean.startswith("http://") and not url_clean.startswith("https://"):
        if url_clean.startswith("//"):
            url_clean = f"https:{url_clean}"
        else:
            url_clean = f"https://{url_clean}"
    READER_URL = os.getenv("READER_URL", "https://r.jina.ai/")
    JINA_API_KEYS = os.getenv("JINA_API_KEYS", None)

    session = _get_thread_local_session()

    for attempt in range(max_retries):
        if JINA_API_KEYS is None:
            headers = {}
        else:
            headers = {
                "Authorization": f"Bearer {JINA_API_KEYS}",
            }
        try:
            response = session.get(
                f"{READER_URL}/{url_clean}", headers=headers, timeout=timeout
            )
            print(f"---jina readpage response.status_code: {response.status_code}---")
            if response.status_code == 200:
                webpage_content = response.text
                return webpage_content
            else:
                print(response.text)
                raise ValueError("jina readpage error")
        except Exception as e:
            time.sleep(0.5 + random.uniform(0, 4.0))
            if attempt == max_retries - 1:
                print(f"jina readpage error: {e}")
                return "[visit] Failed to read page."
    print(f"[visit] Failed to read page jina readpage error: {e}")
    raise ValueError("jina readpage error")


def _readpage_crawl4ai(url: str) -> str:
    """
    Read webpage content using Crawl4AI service.

    Args:
        url: The URL to read

    Returns:
        str: The webpage content (markdown) or error message
    """
    max_retries = 2
    timeout = 50

    url_clean = url.strip()
    if not url_clean.startswith("http://") and not url_clean.startswith("https://"):
        if url_clean.startswith("//"):
            url_clean = f"https:{url_clean}"
        else:
            url_clean = f"https://{url_clean}"

    CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", "http://localhost:11235")

    session = _get_thread_local_session()

    for attempt in range(max_retries):
        try:
            response = session.post(
                f"{CRAWL4AI_URL}/crawl",
                json={"urls": [url_clean], "priority": 10},
                timeout=timeout,
            )
            print(
                f"---crawl4ai readpage response.status_code: {response.status_code}---"
            )

            if response.status_code != 200:
                error_text = response.text
                print(error_text)
                raise ValueError(
                    f"crawl4ai readpage error: failed to submit crawl job. Response: {error_text}"
                )

            response_data = response.json()

            if "results" in response_data:
                results = response_data["results"]
                if results and len(results) > 0:
                    result = results[0]
                    if "markdown" in result and "raw_markdown" in result["markdown"]:
                        webpage_content = result["markdown"]["raw_markdown"]
                        return webpage_content
                    else:
                        raise ValueError(
                            "crawl4ai readpage error: no markdown content in results"
                        )
                else:
                    raise ValueError("crawl4ai readpage error: empty results")
            else:
                raise ValueError("crawl4ai readpage error: no results in response")

        except Exception as e:
            time.sleep(0.5 + random.uniform(0, 4.0))
            if attempt == max_retries - 1:
                print(f"crawl4ai readpage error: {e}")
                raise ValueError(
                    f"crawl4ai readpage error after {max_retries} attempts: {e}"
                )

    raise ValueError("crawl4ai readpage error: all retries exhausted")


def _readpage_firecrawl(url: str) -> str:
    """
    Read webpage content using the Firecrawl API (via FIRECRAWL_API_URL).
    Requires FIRECRAWL_API_URL in the environment; optionally FIRECRAWL_API_KEY for Bearer auth.

    Args:
        url: The URL to scrape.

    Returns:
        str: Page markdown content, or "[visit] Failed to read page." on failure.
    """
    max_retries = 2
    timeout = 60

    url_clean = url.strip()
    if not url_clean.startswith("http://") and not url_clean.startswith("https://"):
        if url_clean.startswith("//"):
            url_clean = f"https:{url_clean}"
        else:
            url_clean = f"https://{url_clean}"

    api_url = os.getenv("FIRECRAWL_API_URL", "")
    if not api_url:
        logging.warning("FIRECRAWL_API_URL not set, firecrawl readpage disabled")
        return "[visit] Failed to read page."

    scrape_url = api_url + "/v2/scrape"
    api_key = os.getenv("FIRECRAWL_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    session = _get_thread_local_session()

    for attempt in range(max_retries):
        try:
            response = session.post(
                scrape_url,
                headers=headers,
                json={"url": url_clean, "formats": ["markdown"]},
                timeout=timeout,
            )
            logging.debug(
                f"firecrawl readpage response.status_code: {response.status_code}"
            )
            if response.status_code != 200:
                response.raise_for_status()

            data = response.json()
            inner = data.get("data", {})
            page_markdown = inner.get("markdown") or data.get("markdown") or ""
            return page_markdown if page_markdown else "[visit] Failed to read page."
        except Exception as e:
            time.sleep(0.5 + random.uniform(0, 4.0))
            if attempt == max_retries - 1:
                logging.warning(f"firecrawl readpage error: {e}")
                return "[visit] Failed to read page."
    return "[visit] Failed to read page."


# Example usage and entry point
if __name__ == "__main__":
    pass