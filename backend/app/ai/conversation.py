from collections import OrderedDict
import json
from threading import Lock
from typing import Any
from uuid import uuid4

from app.ai.config import get_config


_conversations: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
_tool_observations: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
_lock = Lock()
_MAX_CONVERSATIONS = 500


def load_conversation(conversation_id: str | None, submitted: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    identifier = conversation_id or str(uuid4())
    limit = get_config().conversation_turn_limit * 2
    with _lock:
        history = list(_conversations.get(identifier, submitted))[-limit:]
        _conversations[identifier] = history
        _conversations.move_to_end(identifier)
        while len(_conversations) > _MAX_CONVERSATIONS:
            expired_id, _ = _conversations.popitem(last=False)
            _tool_observations.pop(expired_id, None)
    return identifier, history


def save_turn(
    conversation_id: str,
    history: list[dict[str, Any]],
    user_message: str,
    assistant_message: str,
    tool_results: list[dict[str, Any]] | None = None,
) -> None:
    limit = get_config().conversation_turn_limit * 2
    safe_history = [
        {"role": item["role"], "content": str(item["content"])}
        for item in history
        if item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str)
    ]
    safe_history.extend((
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ))
    with _lock:
        _conversations[conversation_id] = safe_history[-limit:]
        _conversations.move_to_end(conversation_id)
        if tool_results:
            previous = _tool_observations.get(conversation_id, [])
            current = [
                {
                    "tool": str(item.get("tool", "")),
                    "result": json.dumps(item.get("result"), default=str)[:8000],
                }
                for item in tool_results[-5:]
            ]
            _tool_observations[conversation_id] = (previous + current)[-5:]
            _tool_observations.move_to_end(conversation_id)


def reset_conversations() -> None:
    with _lock:
        _conversations.clear()
        _tool_observations.clear()