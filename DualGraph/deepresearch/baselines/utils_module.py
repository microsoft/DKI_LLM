import re
import json
import os
import threading
import json_repair
# Shared lock across all modules for concurrency safety
_file_lock = threading.Lock()


def clear_usage_data_if_exists(report_id: int, usage_file: str):
    """
    Clear data for a given report_id if it exists.
    Used to handle residual data from previous abnormal terminations.

    Args:
        report_id: Report ID.
        usage_file: Path to the usage file.
    """
    with _file_lock:
        try:
            if os.path.exists(usage_file):
                with open(usage_file, "r", encoding="utf-8") as f:
                    run_config = json.load(f)
            else:
                return

            llm_usage = run_config.get("llm_usage", {})
            filter_stats = run_config.get("filter_stats", {})
            module_elapsed_time = run_config.get("module_elapsed_time", {})
            action_elapsed_time = run_config.get("action_elapsed_time", {})
            readpage_stats = run_config.get("readpage_stats", {})
            report_id_str = str(report_id)

            need_save = False

            if report_id_str in llm_usage and llm_usage[report_id_str]:
                llm_usage[report_id_str] = {}
                run_config["llm_usage"] = llm_usage
                need_save = True
                print(f"[cleanup] Cleared residual usage data for report {report_id}")

            if report_id_str in filter_stats:
                del filter_stats[report_id_str]
                run_config["filter_stats"] = filter_stats
                need_save = True
                print(f"[cleanup] Cleared residual filter_stats data for report {report_id}")

            if report_id_str in module_elapsed_time:
                del module_elapsed_time[report_id_str]
                run_config["module_elapsed_time"] = module_elapsed_time
                need_save = True
                print(f"[cleanup] Cleared residual module_elapsed_time data for report {report_id}")

            if report_id_str in action_elapsed_time:
                del action_elapsed_time[report_id_str]
                run_config["action_elapsed_time"] = action_elapsed_time
                need_save = True
                print(f"[cleanup] Cleared residual action_elapsed_time data for report {report_id}")

            if report_id_str in readpage_stats:
                del readpage_stats[report_id_str]
                run_config["readpage_stats"] = readpage_stats
                need_save = True
                print(f"[cleanup] Cleared residual readpage_stats data for report {report_id}")

            if need_save:
                with open(usage_file, "w", encoding="utf-8") as f:
                    json.dump(run_config, f, indent=4)
        except Exception as e:
            print(f"[warning] Error clearing usage data for report {report_id}: {e}")


# ============================================================================
# Filter stats update helper
# ============================================================================
def update_filter_stats(report_id: int, usage_file: str, filter_type: str, before_count: int, after_count: int):
    """
    Update filter statistics for a given report_id, accumulating before/after counts
    and computing the average retention rate.

    Supports two filter types:
    - "first": First filter (URL selection)
    - "second": Second filter (content validity)

    Args:
        report_id: Report ID.
        usage_file: Path to the usage file.
        filter_type: Filter type, 'first' or 'second'.
        before_count: Count before filtering.
        after_count: Count after filtering.
    """
    with _file_lock:
        try:
            if os.path.exists(usage_file):
                try:
                    with open(usage_file, "r", encoding="utf-8") as f:
                        run_config = json.load(f)
                except (json.JSONDecodeError, IOError):
                    run_config = {"llm_usage": {}, "filter_stats": {}}
            else:
                run_config = {"llm_usage": {}, "filter_stats": {}}

            filter_stats = run_config.get("filter_stats", {})
            report_id_str = str(report_id)

            if report_id_str not in filter_stats:
                filter_stats[report_id_str] = {
                    "first_filter": {
                        "total_before": 0,
                        "total_after": 0,
                        "count": 0
                    },
                    "second_filter": {
                        "total_before": 0,
                        "total_after": 0,
                        "count": 0
                    }
                }

            filter_key = "first_filter" if filter_type == "first" else "second_filter"

            filter_stats[report_id_str][filter_key]["total_before"] += before_count
            filter_stats[report_id_str][filter_key]["total_after"] += after_count
            filter_stats[report_id_str][filter_key]["count"] += 1

            total_before = filter_stats[report_id_str][filter_key]["total_before"]
            total_after = filter_stats[report_id_str][filter_key]["total_after"]
            if total_before > 0:
                average_retention_rate = total_after / total_before
            else:
                average_retention_rate = 0.0

            filter_stats[report_id_str][filter_key]["average_retention_rate"] = average_retention_rate

            run_config["filter_stats"] = filter_stats

            with open(usage_file, "w", encoding="utf-8") as f:
                json.dump(run_config, f, indent=4)

            print(f"[update] Report {report_id} {filter_type} filter stats - this round: {before_count}->{after_count}, cumulative: {total_before}->{total_after}, avg retention: {average_retention_rate:.2%}")
        except Exception as e:
            print(f"[warning] Error updating filter stats for report {report_id}: {e}")


# ============================================================================
# Elapsed time update helper
# ============================================================================
def update_elapsed_time(report_id: int, usage_file: str, elapsed_time: float):
    """
    Update elapsed time data for a given report_id.

    Args:
        report_id: Report ID.
        usage_file: Path to the usage file.
        elapsed_time: Elapsed time in seconds.
    """
    with _file_lock:
        try:
            if os.path.exists(usage_file):
                with open(usage_file, "r", encoding="utf-8") as f:
                    run_config = json.load(f)
            else:
                run_config = {"llm_usage": {}}

            llm_usage = run_config.get("llm_usage", {})
            report_id_str = str(report_id)

            if report_id_str not in llm_usage:
                llm_usage[report_id_str] = {}

            llm_usage[report_id_str]["elapsed_time"] = elapsed_time

            run_config["llm_usage"] = llm_usage

            with open(usage_file, "w", encoding="utf-8") as f:
                json.dump(run_config, f, indent=4)
        except Exception as e:
            print(f"[warning] Error updating elapsed time for report {report_id}: {e}")

def update_action_elapsed_time(action_name, report_id, usage_file: str, elapsed_time: float = None):
    """
    Update action elapsed time data for a given report_id and action_name.

    Accumulates action elapsed time in the config file, grouped by report ID and action name.
    Thread-safe via file lock.

    Args:
        action_name: Action name for distinguishing different action time stats.
        report_id: Report ID for distinguishing different reports.
        usage_file: Path to the usage file.
        elapsed_time: Optional elapsed time in seconds. If provided, accumulated into the action's total.
    """
    if elapsed_time is None:
        return

    with _file_lock:
        try:
            if os.path.exists(usage_file):
                try:
                    with open(usage_file, "r", encoding="utf-8") as f:
                        run_config = json.load(f)
                except (json.JSONDecodeError, IOError):
                    run_config = {"action_elapsed_time": {}}
            else:
                run_config = {"action_elapsed_time": {}}

            action_elapsed_time = run_config.get("action_elapsed_time", {})

            if str(report_id) not in action_elapsed_time:
                action_elapsed_time[str(report_id)] = {}

            if action_name not in action_elapsed_time[str(report_id)]:
                action_elapsed_time[str(report_id)][action_name] = {
                    "elapsed_time": 0.0,
                }

            action_elapsed_time[str(report_id)][action_name]["elapsed_time"] += elapsed_time

            run_config["action_elapsed_time"] = action_elapsed_time

            with open(usage_file, "w", encoding="utf-8") as f:
                json.dump(run_config, f, indent=4)
        except Exception as e:
            print(f"[warning] Error updating action {action_name} elapsed time for report {report_id}: {e}")

def update_llm_usage(llm_response, function_name, report_id, usage_file: str, elapsed_time: float = None):
    """
    Update LLM usage statistics in the config file.

    Accumulates token usage from each LLM call, grouped by report ID and function name.
    Thread-safe via file lock.

    Args:
        llm_response: LLM response object containing usage field (prompt_tokens, completion_tokens).
        function_name: Name of the calling function, for distinguishing token usage by feature.
        report_id: Report ID for distinguishing different reports.
        usage_file: Path to the usage file.
        elapsed_time: Optional elapsed time in seconds. If provided, accumulated into the function's total.
    """
    with _file_lock:
        if os.path.exists(usage_file):
            try:
                with open(usage_file, "r", encoding="utf-8") as f:
                    run_config = json.load(f)
            except (json.JSONDecodeError, IOError):
                run_config = {"llm_usage": {}}
        else:
            run_config = {"llm_usage": {}}

        llm_usage = run_config.get("llm_usage", {})

        if str(report_id) not in llm_usage:
            llm_usage[str(report_id)] = {}

        if function_name not in llm_usage[str(report_id)]:
            llm_usage[str(report_id)][function_name] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "call_count": 0,
            }

        if "call_count" not in llm_usage[str(report_id)][function_name]:
            llm_usage[str(report_id)][function_name]["call_count"] = 0

        llm_usage[str(report_id)][function_name]["prompt_tokens"] += llm_response.usage['prompt_tokens']
        llm_usage[str(report_id)][function_name]["completion_tokens"] += llm_response.usage['completion_tokens']
        llm_usage[str(report_id)][function_name]["call_count"] += 1

        if elapsed_time is not None:
            if "elapsed_time" not in llm_usage[str(report_id)][function_name]:
                llm_usage[str(report_id)][function_name]["elapsed_time"] = 0.0
            llm_usage[str(report_id)][function_name]["elapsed_time"] += elapsed_time

        run_config["llm_usage"] = llm_usage
        time_info = f", elapsed: {elapsed_time:.2f}s" if elapsed_time is not None else ""
        print(f"  --[update] Report {report_id} ---{function_name} LLM usage updated{time_info}--")
        with open(usage_file, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=4)


def merge_shared_init_usage_into_file(source_usage_file: str, target_usage_file: str, report_id: int):
    """
    Merge shared init usage data (e.g., shared_init_usage_case{report_id}_iter0.json)
    for the given report_id into the caller's target_usage_file, so this run's usage
    includes the shared init's tokens/time.

    Args:
        source_usage_file: Path to the shared init usage JSON.
        target_usage_file: Path to this run's usage file.
        report_id: Report ID.
    """
    if not source_usage_file or not os.path.exists(source_usage_file):
        return
    with _file_lock:
        try:
            with open(source_usage_file, "r", encoding="utf-8") as f:
                source_config = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[warning] Failed to read shared init usage file: {source_usage_file}, {e}")
            return
    rid = str(report_id)
    llm_s = source_config.get("llm_usage", {}).get(rid, {})
    action_s = source_config.get("action_elapsed_time", {}).get(rid, {})
    filter_s = source_config.get("filter_stats", {}).get(rid, {})
    module_s = source_config.get("module_elapsed_time", {}).get(rid, {})
    readpage_s = source_config.get("readpage_stats", {}).get(rid, {})

    with _file_lock:
        try:
            if os.path.exists(target_usage_file):
                try:
                    with open(target_usage_file, "r", encoding="utf-8") as f:
                        run_config = json.load(f)
                except (json.JSONDecodeError, IOError):
                    run_config = {}
            else:
                run_config = {}
            llm_usage = run_config.get("llm_usage", {})
            action_elapsed_time = run_config.get("action_elapsed_time", {})
            filter_stats = run_config.get("filter_stats", {})
            module_elapsed_time = run_config.get("module_elapsed_time", {})
            readpage_stats = run_config.get("readpage_stats", {})

            if rid not in llm_usage:
                llm_usage[rid] = {}
            for fn, data in llm_s.items():
                if fn not in llm_usage[rid]:
                    llm_usage[rid][fn] = {"prompt_tokens": 0, "completion_tokens": 0, "call_count": 0}
                if "elapsed_time" not in llm_usage[rid][fn]:
                    llm_usage[rid][fn]["elapsed_time"] = 0.0
                for k in ("prompt_tokens", "completion_tokens", "call_count"):
                    llm_usage[rid][fn][k] = llm_usage[rid][fn].get(k, 0) + data.get(k, 0)
                llm_usage[rid][fn]["elapsed_time"] += data.get("elapsed_time", 0.0)

            if rid not in action_elapsed_time:
                action_elapsed_time[rid] = {}
            for an, data in action_s.items():
                if an not in action_elapsed_time[rid]:
                    action_elapsed_time[rid][an] = {"elapsed_time": 0.0}
                action_elapsed_time[rid][an]["elapsed_time"] = action_elapsed_time[rid][an].get("elapsed_time", 0.0) + data.get("elapsed_time", 0.0)

            if filter_s:
                if rid not in filter_stats:
                    filter_stats[rid] = {"first_filter": {"total_before": 0, "total_after": 0, "count": 0}, "second_filter": {"total_before": 0, "total_after": 0, "count": 0}}
                for fk in ("first_filter", "second_filter"):
                    if fk not in filter_s:
                        continue
                    sf = filter_s[fk]
                    tf = filter_stats[rid].get(fk, {"total_before": 0, "total_after": 0, "count": 0})
                    filter_stats[rid][fk] = {
                        "total_before": tf.get("total_before", 0) + sf.get("total_before", 0),
                        "total_after": tf.get("total_after", 0) + sf.get("total_after", 0),
                        "count": tf.get("count", 0) + sf.get("count", 0),
                    }
                    tb, ta = filter_stats[rid][fk]["total_before"], filter_stats[rid][fk]["total_after"]
                    filter_stats[rid][fk]["average_retention_rate"] = (ta / tb) if tb > 0 else 0.0

            if module_s:
                if rid not in module_elapsed_time:
                    module_elapsed_time[rid] = {}
                for mk, mv in module_s.items():
                    if isinstance(mv, (int, float)):
                        module_elapsed_time[rid][mk] = module_elapsed_time[rid].get(mk, 0.0) + mv
                    elif isinstance(mv, dict) and "elapsed_time" in mv:
                        if mk not in module_elapsed_time[rid]:
                            module_elapsed_time[rid][mk] = {"elapsed_time": 0.0}
                        module_elapsed_time[rid][mk]["elapsed_time"] = module_elapsed_time[rid][mk].get("elapsed_time", 0.0) + mv.get("elapsed_time", 0.0)

            if readpage_s:
                if rid not in readpage_stats:
                    readpage_stats[rid] = {"primary_success_count": 0, "jina_fallback_success_count": 0, "fail_count": 0, "fail_rate": 0.0}
                for k in ("primary_success_count", "jina_fallback_success_count", "fail_count"):
                    readpage_stats[rid][k] = readpage_stats[rid].get(k, 0) + readpage_s.get(k, 0)
                total = readpage_stats[rid]["primary_success_count"] + readpage_stats[rid]["jina_fallback_success_count"] + readpage_stats[rid]["fail_count"]
                readpage_stats[rid]["fail_rate"] = (readpage_stats[rid]["fail_count"] / total) if total > 0 else 0.0

            run_config["llm_usage"] = llm_usage
            run_config["action_elapsed_time"] = action_elapsed_time
            run_config["filter_stats"] = filter_stats
            run_config["module_elapsed_time"] = module_elapsed_time
            run_config["readpage_stats"] = readpage_stats
            with open(target_usage_file, "w", encoding="utf-8") as f:
                json.dump(run_config, f, indent=4)
            print(f"[update] Merged shared init usage into {target_usage_file} (report_id={report_id})")
        except Exception as e:
            print(f"[warning] Error merging shared init usage into {target_usage_file}: {e}")


def update_readpage_stats(report_id: int, usage_file: str, result_type: str):
    """
    Update readpage success/failure statistics for a given report_id.

    Called after each _readpage invocation in _crawl_summarize_with_filter,
    tracking primary method success, Jina fallback success, and failure counts.

    Args:
        report_id: Report ID.
        usage_file: Path to the usage file.
        result_type: Result type:
            - "primary_success": Primary method (crawl4ai/jina/firecrawl) succeeded directly
            - "jina_fallback_success": Primary method failed, Jina fallback succeeded
            - "fail": All methods failed
    """
    if report_id is None or usage_file is None:
        return

    with _file_lock:
        try:
            if os.path.exists(usage_file):
                try:
                    with open(usage_file, "r", encoding="utf-8") as f:
                        run_config = json.load(f)
                except (json.JSONDecodeError, IOError):
                    run_config = {}
            else:
                run_config = {}

            readpage_stats = run_config.get("readpage_stats", {})
            report_id_str = str(report_id)

            if report_id_str not in readpage_stats:
                readpage_stats[report_id_str] = {
                    "primary_success_count": 0,
                    "jina_fallback_success_count": 0,
                    "fail_count": 0,
                    "fail_rate": 0.0
                }

            # Backward compatibility: migrate old field names
            stats = readpage_stats[report_id_str]
            if "success_count" in stats and "primary_success_count" not in stats:
                stats["primary_success_count"] = stats.pop("success_count")
            if "jina_fallback_success_count" not in stats:
                stats["jina_fallback_success_count"] = 0

            if result_type == "primary_success":
                stats["primary_success_count"] += 1
            elif result_type == "jina_fallback_success":
                stats["jina_fallback_success_count"] += 1
            else:
                stats["fail_count"] += 1

            total = stats["primary_success_count"] + stats["jina_fallback_success_count"] + stats["fail_count"]
            if total > 0:
                stats["fail_rate"] = stats["fail_count"] / total
            else:
                stats["fail_rate"] = 0.0

            run_config["readpage_stats"] = readpage_stats

            with open(usage_file, "w", encoding="utf-8") as f:
                json.dump(run_config, f, indent=4)

            print(f"[update] Report {report_id} readpage stats - primary success: {stats['primary_success_count']}, jina fallback success: {stats['jina_fallback_success_count']}, fail: {stats['fail_count']}, fail rate: {stats['fail_rate']:.2%}")
        except Exception as e:
            print(f"[warning] Error updating readpage stats for report {report_id}: {e}")


def clean_json_string(raw_output: str) -> str:
    """
    Clean LLM output string: extract and fix the JSON portion.
    1. Remove Markdown code block markers (```json or ```)
    2. Extract content between first '{' or '[' and last '}' or ']'
    """
    if not raw_output:
        return ""

    code_block_pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(code_block_pattern, raw_output, re.DOTALL)

    if match:
        content = match.group(1)
    else:
        content = raw_output.strip()

    start_curly = content.find('{')
    start_bracket = content.find('[')

    if start_curly == -1 and start_bracket == -1:
        return content

    start_pos = start_curly if (start_curly != -1 and (start_bracket == -1 or start_curly < start_bracket)) else start_bracket

    end_curly = content.rfind('}')
    end_bracket = content.rfind(']')
    end_pos = max(end_curly, end_bracket)

    if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
        return content[start_pos:end_pos + 1]

    return content

def safe_json_loads(raw_output: str):
    """Safe JSON parsing wrapper."""
    cleaned = clean_json_string(raw_output)
    try:
        return json_repair.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON parse failed: {e}")
        return None


# ============================================================================
# Gym dataset ID conversion utilities
# ============================================================================
def load_gym_id_mapping(mapping_file_path: str) -> dict:
    """
    Load gym ID mapping file.

    Args:
        mapping_file_path: Path to the mapping file (typically report_dir/gym_id_mapping.json).

    Returns:
        Dict containing mapping data with keys:
        - sequential_to_gym_id: sequential ID -> actual gym ID
        - gym_id_to_sequential: actual gym ID -> sequential ID
    """
    if not os.path.exists(mapping_file_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file_path}")

    with open(mapping_file_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    return mapping_data


def sequential_id_to_gym_id(sequential_id: int, mapping_file_path: str) -> int:
    """
    Convert a sequential ID to the actual gym dataset ID.

    Args:
        sequential_id: Sequential ID (starting from 1).
        mapping_file_path: Path to the mapping file.

    Returns:
        Actual gym dataset ID.

    Raises:
        KeyError: If sequential_id is not in the mapping range.
    """
    mapping_data = load_gym_id_mapping(mapping_file_path)
    sequential_to_gym_id_raw = mapping_data.get("sequential_to_gym_id", {})

    # JSON keys are strings after loading, convert to integers
    sequential_to_gym_id = {int(k): int(v) for k, v in sequential_to_gym_id_raw.items()}

    if sequential_id not in sequential_to_gym_id:
        raise KeyError(f"Sequential ID {sequential_id} not in mapping range (max: {max(sequential_to_gym_id.keys()) if sequential_to_gym_id else 0})")

    return sequential_to_gym_id[sequential_id]


def gym_id_to_sequential_id(gym_id: int, mapping_file_path: str) -> int:
    """
    Convert an actual gym dataset ID to a sequential ID.

    Args:
        gym_id: Actual gym dataset ID.
        mapping_file_path: Path to the mapping file.

    Returns:
        Sequential ID (starting from 1).

    Raises:
        KeyError: If gym_id is not in the mapping.
    """
    mapping_data = load_gym_id_mapping(mapping_file_path)
    gym_id_to_sequential_raw = mapping_data.get("gym_id_to_sequential", {})

    # JSON keys are strings after loading, convert to integers
    gym_id_to_sequential = {int(k): int(v) for k, v in gym_id_to_sequential_raw.items()}

    if gym_id not in gym_id_to_sequential:
        raise KeyError(f"Gym ID {gym_id} not in mapping")

    return gym_id_to_sequential[gym_id]
