import logging
import secrets
import time
from threading import Lock
from typing import Any

from app.ai.tools.dataset_tools import get_dataset_profile, get_history
from app.ai.tools.preprocessing_tools import execute_write_action, propose_write_action
from app.services.dataset_service import get_dataset


logger = logging.getLogger(__name__)
_pending: dict[str, tuple[float, dict[str, Any]]] = {}
_lock = Lock()
_TOKEN_TTL_SECONDS = 600


def prepare_confirmation(
    dataset_id: str,
    action_type: str,
    parameters: dict[str, Any],
    request_id: str | None = None,
) -> dict[str, Any]:
    get_dataset(dataset_id)
    action = propose_write_action(dataset_id, action_type, parameters)
    token = secrets.token_urlsafe(24)
    with _lock:
        now = time.monotonic()
        for key, (expiry, _) in list(_pending.items()):
            if expiry <= now:
                del _pending[key]
        _pending[token] = (now + _TOKEN_TTL_SECONDS, action)
    logger.info("Agent action staged request_id=%s type=%s dataset_id=%s", request_id or "unknown", action_type, dataset_id)
    return {"confirmation_id": token, "type": action_type, "dataset_id": dataset_id, "parameters": parameters}


def cancel_confirmation(token: str) -> bool:
    with _lock:
        return _pending.pop(token, None) is not None


def _verify(action: dict[str, Any], result: dict[str, Any], before: dict[str, Any], after: dict[str, Any]) -> bool:
    action_type = action["type"]
    params = action["parameters"]
    if not result.get("success", action_type == "remove_duplicates"):
        return False
    if action_type == "remove_duplicates":
        return after["duplicate_rows"] == 0 and after["rows"] <= before["rows"]
    if action_type == "fill_missing_values":
        column = params.get("column")
        before_values = before["missing_values"]["by_column"]
        after_values = after["missing_values"]["by_column"]
        targets = [column] if column else list(before_values)
        return all(after_values.get(name, 0) <= before_values.get(name, 0) for name in targets)
    if action_type == "convert_dtype":
        dtype = after["data_types"].get(params["column"], "").casefold()
        expected = {
            "integer": dtype.startswith(("int", "uint")),
            "float": dtype.startswith("float"),
            "string": dtype in {"string", "object"},
            "boolean": dtype in {"boolean", "bool"},
            "datetime": dtype.startswith("datetime64"),
        }
        return expected.get(params["dtype"], False)
    if action_type == "rename_column":
        return params["new_name"] in after["data_types"] and params["old_name"] not in after["data_types"]
    if action_type == "delete_column":
        return params["column"] not in after["data_types"]
    if action_type == "create_column":
        return params["name"] in after["data_types"]
    return False


def execute_confirmation(token: str, request_id: str | None = None) -> dict[str, Any]:
    with _lock:
        stored = _pending.pop(token, None)
    if stored is None or stored[0] <= time.monotonic():
        return {"success": False, "verified": False, "message": "This confirmation is invalid or expired. Please ask the assistant again."}

    action = stored[1]
    dataset_id = action["dataset_id"]
    try:
        before = get_dataset_profile(dataset_id)
        history_before = get_history(dataset_id)
        result = execute_write_action(action)
        after = get_dataset_profile(dataset_id)
        verified = _verify(action, result, before, after)
        history = get_history(dataset_id)
        history_verified = len(history) > len(history_before)
        if action["type"] == "remove_duplicates" and result.get("duplicate_count", 0) == 0:
            history_verified = True
        verified = verified and history_verified
        logger.info(
            "Agent action completed request_id=%s type=%s dataset_id=%s verified=%s",
            request_id or "unknown", action["type"], dataset_id, verified,
        )
        if not verified:
            return {"success": False, "verified": False, "message": "Operation could not be verified."}
        return {
            "success": True,
            "verified": True,
            "message": result.get("message", f"{action['type']} completed and verified."),
            "action": {"type": action["type"], "dataset_id": dataset_id},
            "before": before,
            "after": after,
            "result": result,
        }
    except (LookupError, ValueError) as exc:
        logger.info("Agent action rejected request_id=%s type=%s dataset_id=%s", request_id or "unknown", action["type"], dataset_id)
        return {"success": False, "verified": False, "message": str(exc)}
    except Exception:
        logger.exception("Agent action failed request_id=%s type=%s dataset_id=%s", request_id or "unknown", action["type"], dataset_id)
        return {"success": False, "verified": False, "message": "The requested operation failed and was not verified."}