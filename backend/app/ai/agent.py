import json
import logging
import re
import time
from typing import Any

from app.ai.action_manager import prepare_confirmation
from app.ai.config import get_config
from app.ai.conversation import load_conversation, save_turn
from app.ai.llm_client import LLMError, get_llm_client
from app.ai.prompts import build_system_prompt
from app.ai.rag.retriever import retrieve
from app.ai.tools.dataset_tools import get_duplicates, get_missing_values
from app.ai.tools.registry import READ_TOOLS, execute_read_tool
from app.services.dataset_service import dataset_info, get_dataset


logger = logging.getLogger(__name__)
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_MOJIBAKE_PATTERN = re.compile(r"(?:Ã.|â€.|ï¿½)+")


def _repair_mojibake(text: str) -> str:
    def repair(match: re.Match[str]) -> str:
        fragment = match.group(0)
        try:
            return fragment.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return fragment

    return _MOJIBAKE_PATTERN.sub(repair, text)


def _retrieved_context(question: str) -> list[dict[str, Any]]:
    try:
        return retrieve(question)
    except Exception as exc:
        logger.info("Knowledge retrieval unavailable: %s", type(exc).__name__)
        return []


def _knowledge_prompt(results: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{index}] {item['title']} ({item['source']})\n{item['text']}"
        for index, item in enumerate(results, start=1)
    )


def _citations(answer: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexes = {int(value) for value in _CITATION_PATTERN.findall(answer)}
    return [
        {
            "title": item["title"],
            "source": item["source"],
            "category": item["category"],
            "chunk_id": item["chunk_id"],
            "score": item["score"],
        }
        for index, item in enumerate(results, start=1)
        if index in indexes
    ]


def _requires_dataset(message: str) -> bool:
    lowered = message.casefold()
    if re.search(r"\b(dataset|my data|my file|my dataset|this data|this file)\b", lowered):
        return True
    if re.search(r"\b(fill|impute|remove|convert|change|rename|delete|drop|create)\b", lowered):
        return True
    if re.search(r"\b(how many|count|show|list|preview|profile)\b", lowered) and re.search(
        r"\b(missing|duplicate|outlier|row|column|statistics|salary|chart|visualization|visualisation|history)\w*\b",
        lowered,
    ):
        return True
    return bool(re.search(r"\b(health score|data quality score|statistics for|average of|mean of|median of|outliers in|duplicates in|missing values in|clean first)\b", lowered))


def _find_column(text: str, columns: list[str]) -> str | None:
    folded = text.casefold()
    matches = [column for column in columns if column.casefold() in folded]
    return max(matches, key=len) if matches else None


def _write_intent(message: str, dataset_id: str) -> tuple[str, dict[str, Any]] | None:
    lowered = message.casefold()
    if re.search(r"\b(remove|delete|drop)\b.{0,30}\bduplicates?\b|\bduplicates?\b.{0,30}\b(remove|delete)\b", lowered):
        return "remove_duplicates", {}

    if not any(word in lowered for word in ("fill", "impute", "remove", "convert", "change", "rename", "delete", "create")):
        return None
    try:
        info = dataset_info(dataset_id)
    except (LookupError, ValueError):
        return None
    column = _find_column(message, info["column_names"])

    if any(word in lowered for word in ("fill", "impute")) and any(word in lowered for word in ("missing", "null", "nan")):
        method_match = re.search(r"\b(mean|median|mode|ffill|bfill|remove_rows|custom)\b", lowered)
        if method_match:
            method = method_match.group(1)
            parameters: dict[str, Any] = {"method": method}
            if column:
                parameters["column"] = column
            if method == "custom":
                value_match = re.search(r"\b(?:value|with)\s+(?:of\s+)?(.+?)\s*$", message, re.IGNORECASE)
                if not value_match:
                    return None
                raw_value = value_match.group(1).strip().strip("\"'")
                try:
                    numeric_value = float(raw_value)
                    parameters["value"] = int(numeric_value) if numeric_value.is_integer() else numeric_value
                except ValueError:
                    if raw_value.casefold() in {"true", "false"}:
                        parameters["value"] = raw_value.casefold() == "true"
                    else:
                        parameters["value"] = raw_value
            return "fill_missing_values", parameters

    dtype_match = re.search(r"\b(integer|float|string|boolean|datetime)\b", lowered)
    if column and dtype_match and any(word in lowered for word in ("convert", "change", "type", "dtype")):
        return "convert_dtype", {"column": column, "dtype": dtype_match.group(1)}

    rename_match = re.search(r"\brename\s+(?:column\s+)?['\"]?(.+?)['\"]?\s+to\s+['\"]?([^'\"]+?)['\"]?\s*$", message, re.IGNORECASE)
    if rename_match:
        old_name = _find_column(rename_match.group(1), info["column_names"])
        if old_name:
            return "rename_column", {"old_name": old_name, "new_name": rename_match.group(2).strip()}

    if column and re.search(r"\b(delete|drop|remove)\b", lowered) and "column" in lowered:
        return "delete_column", {"column": column}

    create_match = re.search(
        r"\bcreate(?:\s+a)?(?:\s+calculated)?\s+column(?:\s+(?:named|called))?\s+"
        r"['\"]?([A-Za-z][A-Za-z0-9_]*)['\"]?\s*(?:=|\bas\b|\bfrom\b)\s*(.+?)\s*[.!?]*$",
        message,
        re.IGNORECASE,
    )
    if create_match:
        return "create_column", {"name": create_match.group(1), "expression": create_match.group(2).strip()}
    return None


def _fallback_tool_names(message: str) -> list[str]:
    lowered = message.casefold()
    if "clean first" in lowered or "what should i clean" in lowered or "priorit" in lowered:
        return ["get_dataset_profile", "get_missing_values", "get_duplicates", "get_outliers", "get_data_health"]
    if any(term in lowered for term in ("missing", "null", "nan")):
        return ["get_missing_values"]
    if "duplicate" in lowered:
        return ["get_duplicates"]
    if any(term in lowered for term in ("health", "quality", "score")):
        return ["get_data_health"]
    if "outlier" in lowered:
        return ["get_outliers"]
    if any(term in lowered for term in ("statistic", "average", "mean", "median", "maximum", "minimum")):
        return ["get_statistics"]
    if any(term in lowered for term in ("chart", "visualization", "visualisation", "plot", "graph")):
        return ["get_visualization_recommendations"]
    if "history" in lowered:
        return ["get_history"]
    if any(term in lowered for term in ("preview", "sample row")):
        return ["get_dataset_preview"]
    if any(term in lowered for term in ("overview", "summary", "shape", "column", "row", "dataset", "my data", "my file")):
        return ["get_dataset_profile"]
    return []


def _fallback_answer(message: str, tool_results: list[tuple[str, Any]], knowledge: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    by_tool = dict(tool_results)
    if len(tool_results) > 1:
        profile = by_tool.get("get_dataset_profile", {})
        missing = by_tool.get("get_missing_values", {})
        duplicates = by_tool.get("get_duplicates", {})
        outliers = by_tool.get("get_outliers", {})
        health = by_tool.get("get_data_health", {})
        outlier_total = sum(report["outlier_count"] for report in outliers.values())
        issues = []
        if missing.get("total", 0):
            issues.append(f"{missing['total']} missing values")
        if duplicates.get("duplicate_rows", 0):
            issues.append(f"{duplicates['duplicate_rows']} duplicate rows")
        if outlier_total:
            issues.append(f"{outlier_total} IQR outliers")
        focus = ", ".join(issues) if issues else "no missing values, duplicate rows, or IQR outliers"
        if health:
            focus += f". Overall health is {health['overall_score']}/100"
        return f"Based on live checks, first review {focus}. The dataset has {profile.get('rows', 'unknown')} rows.", []
    if "get_missing_values" in by_tool:
        result = by_tool["get_missing_values"]
        details = "; ".join(f"{key}: {value}" for key, value in result["by_column"].items() if value)
        answer = f"Your dataset contains {result['total']} missing values ({result['percentage']}% of cells)."
        if details:
            answer += f" By column: {details}."
        return answer, []
    if "get_duplicates" in by_tool:
        result = by_tool["get_duplicates"]
        return f"Your dataset contains {result['duplicate_rows']} duplicate rows ({result['percentage']}%).", []
    if "get_data_health" in by_tool:
        result = by_tool["get_data_health"]
        return (
            f"Data health is {result['overall_score']}/100: completeness {result['completeness']}%, "
            f"consistency {result['consistency']}%, validity {result['validity']}%."
        ), []
    if "get_dataset_profile" in by_tool and len(tool_results) == 1:
        result = by_tool["get_dataset_profile"]
        columns = ", ".join(result["data_types"])
        return f"The dataset has {result['rows']} rows and {result['columns']} columns: {columns}.", []
    if "get_statistics" in by_tool:
        result = by_tool["get_statistics"]
        columns = result.get("numerical", {})
        if columns:
            values = "; ".join(
                f"{name}: count {stats['count']}, mean {stats['mean']}, median {stats['median']}, "
                f"std {stats['standard_deviation']}, min {stats['minimum']}, max {stats['maximum']}, "
                f"Q1 {stats['q1']}, Q3 {stats['q3']}"
                for name, stats in columns.items()
            )
            return "Live numerical statistics: " + values + ".", []
        return "No numerical statistics are available for this dataset.", []
    if "get_visualization_recommendations" in by_tool:
        items = by_tool["get_visualization_recommendations"]["recommendations"]
        samples = "; ".join(f"{item['chart_type']} for {item['x_column']}" for item in items[:5])
        return (f"The backend returned {len(items)} visualization recommendations." + (f" Examples: {samples}." if samples else "")), []
    if "get_outliers" in by_tool:
        values = by_tool["get_outliers"]
        summary = "; ".join(f"{column}: {report['outlier_count']}" for column, report in values.items())
        return "IQR-based outlier counts: " + (summary or "no numeric columns were found") + ".", []
    if "get_dataset_preview" in by_tool:
        preview = by_tool["get_dataset_preview"]
        return f"Here are {len(preview['rows'])} preview rows for columns: {', '.join(preview['columns'])}.", []
    if "get_history" in by_tool:
        return f"The dataset has {len(by_tool['get_history'])} recorded processing operations.", []
    if knowledge:
        top = knowledge[0]
        citation = {"title": top["title"], "source": top["source"], "category": top["category"], "chunk_id": top["chunk_id"], "score": top["score"]}
        excerpt = top["text"].strip()
        return f"The LLM is not configured, but this relevant knowledge-base excerpt may help: {excerpt} [1]", [citation]
    return "The LLM is not configured, and the knowledge base did not provide enough information to answer that reliably.", []


def _clean_citations(answer: str, knowledge: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    answer = _repair_mojibake(answer)
    valid_indexes = set(range(1, len(knowledge) + 1))
    answer = _CITATION_PATTERN.sub(
        lambda match: match.group(0) if int(match.group(1)) in valid_indexes else "",
        answer,
    )
    return answer.strip(), _citations(answer, knowledge)


async def _llm_answer(
    message: str,
    dataset_id: str | None,
    history: list[dict[str, str]],
    knowledge: list[dict[str, Any]],
    request_id: str,
    observations: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    config = get_config()
    system = build_system_prompt(dataset_id is not None, _knowledge_prompt(knowledge), config.max_tool_calls)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages.extend(history[-config.conversation_turn_limit * 2:])
    messages.append({"role": "user", "content": message})
    call_records: list[dict[str, str]] = []
    calls_used = 0
    while True:
        response = await get_llm_client().chat(
            messages,
            READ_TOOLS if dataset_id and calls_used < config.max_tool_calls else None,
        )
        tool_calls = response.get("tool_calls") or []
        if not tool_calls:
            content = response.get("content")
            if not isinstance(content, str) or not content.strip():
                raise LLMError("The AI service returned an empty response.")
            return content.strip(), call_records
        if calls_used >= config.max_tool_calls:
            return "I reached the dataset inspection limit for this request. Please ask a more focused question.", call_records
        messages.append(response)
        for tool_call in tool_calls:
            if calls_used >= config.max_tool_calls:
                messages.append({"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": "Tool-call limit reached."})
                continue
            calls_used += 1
            function = tool_call.get("function", {})
            name = function.get("name", "")
            started = time.perf_counter()
            try:
                result = execute_read_tool(name, function.get("arguments", "{}"), dataset_id or "") if dataset_id else None
                status = "completed"
                result_json = json.dumps(result, ensure_ascii=True, default=str)
                observations.append({"tool": name, "result": result})
            except (LookupError, ValueError, TypeError, json.JSONDecodeError):
                status = "failed"
                result_json = json.dumps({"error": "Tool request could not be completed."})
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            call_records.append({"tool": name, "status": status, "duration_ms": elapsed_ms})
            logger.info(
                "Agent tool request_id=%s name=%s status=%s duration_ms=%s dataset_id=%s",
                request_id, name, status, elapsed_ms, dataset_id or "none",
            )
            messages.append({"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": result_json})


async def answer_chat(
    message: str,
    dataset_id: str | None,
    conversation_id: str | None = None,
    submitted_history: list[dict[str, str]] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    request_id = request_id or "unknown"
    config = get_config()
    if len(message) > config.max_message_length:
        raise ValueError(f"Message must be at most {config.max_message_length} characters")
    message = message.strip()
    if not message:
        raise ValueError("Message cannot be empty")
    if dataset_id:
        get_dataset(dataset_id)

    identifier, history = load_conversation(conversation_id, submitted_history or [])
    knowledge = _retrieved_context(message)
    citations: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    action = None
    requires_confirmation = False

    if dataset_id and (write := _write_intent(message, dataset_id)):
        action_type, parameters = write
        if action_type == "remove_duplicates":
            duplicate_report = get_duplicates(dataset_id)
            observations.append({"tool": "get_duplicates", "result": duplicate_report})
            tool_calls.append({"tool": "get_duplicates", "status": "completed"})
            if duplicate_report["duplicate_rows"] == 0:
                answer = "There are no duplicate rows to remove. No changes were made."
            else:
                action = prepare_confirmation(dataset_id, action_type, parameters, request_id)
                answer = (
                    f"I found {duplicate_report['duplicate_rows']} duplicate rows. Removing them will modify "
                    "the processed dataset. Would you like me to proceed?"
                )
                requires_confirmation = True
        elif action_type == "fill_missing_values":
            missing_report = get_missing_values(dataset_id)
            observations.append({"tool": "get_missing_values", "result": missing_report})
            tool_calls.append({"tool": "get_missing_values", "status": "completed"})
            if missing_report["total"] == 0:
                answer = "There are no missing values to process. No changes were made."
            else:
                action = prepare_confirmation(dataset_id, action_type, parameters, request_id)
                scope = f" in column {parameters['column']}" if parameters.get("column") else " across supported columns"
                answer = (
                    f"I found {missing_report['total']} missing values. I can apply {parameters['method']}"
                    f"{scope}. This will modify the processed dataset. Would you like me to proceed?"
                )
                requires_confirmation = True
        else:
            action = prepare_confirmation(dataset_id, action_type, parameters, request_id)
            answer = f"This will {action_type.replace('_', ' ')} and modify the processed dataset. Would you like me to proceed?"
            requires_confirmation = True
    elif not dataset_id and _requires_dataset(message):
        answer = "No dataset is currently selected. Select a dataset to ask about its contents or request a data operation."
    elif not get_config().llm_api_key:
        tool_names = _fallback_tool_names(message) if dataset_id else []
        tool_results = []
        for name in tool_names[:config.max_tool_calls]:
            started = time.perf_counter()
            arguments: dict[str, Any] = {}
            if name == "get_statistics" and dataset_id:
                columns = dataset_info(dataset_id)["column_names"]
                selected_column = _find_column(message, columns)
                if selected_column:
                    arguments["column"] = selected_column
            result = execute_read_tool(name, arguments, dataset_id or "")
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            tool_results.append((name, result))
            observations.append({"tool": name, "result": result})
            tool_calls.append({"tool": name, "status": "completed", "duration_ms": elapsed_ms})
            logger.info(
                "Agent tool request_id=%s name=%s status=completed duration_ms=%s dataset_id=%s",
                request_id, name, elapsed_ms, dataset_id or "none",
            )
        answer, citations = _fallback_answer(message, tool_results, knowledge)
    else:
        try:
            answer, tool_calls = await _llm_answer(message, dataset_id, history, knowledge, request_id, observations)
            answer, citations = _clean_citations(answer, knowledge)
        except LLMError as exc:
            logger.info("Agent LLM request unavailable: %s", str(exc))
            answer = str(exc)
        except Exception as exc:
            logger.exception(
                "Agent request failed request_id=%s dataset_id=%s error_type=%s",
                request_id, dataset_id or "none", type(exc).__name__,
            )
            answer = "I couldn't complete that request safely. Please try again."

    save_turn(identifier, history, message, answer, observations)
    legacy_tool_names = {
        "get_dataset_profile": "dataset_info",
        "get_missing_values": "dataset_profile",
        "get_duplicates": "dataset_profile",
        "get_data_health": "dataset_health",
        "get_statistics": "calculate_statistics",
        "get_visualization_recommendations": "visualization_recommendations",
    }
    return {
        "answer": answer,
        "citations": citations,
        "tool_calls": tool_calls,
        "requires_confirmation": requires_confirmation,
        "action": action,
        "conversation_id": identifier,
        "dataset_id": dataset_id,
        "error": answer.startswith(("The AI service ", "OpenRouter ", "I couldn't complete", "The LLM is not configured")),
        "message": answer,
        "tools_used": [legacy_tool_names.get(item["tool"], item["tool"]) for item in tool_calls],
        "operations": [],
        "pending_confirmation": action,
    }