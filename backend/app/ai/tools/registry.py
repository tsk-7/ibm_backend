import json
from typing import Any

from app.ai.tools.dataset_tools import (
    get_dataset_preview,
    get_dataset_profile,
    get_duplicates,
    get_history,
    get_missing_values,
)
from app.ai.tools.health_tools import get_data_health
from app.ai.tools.statistics_tools import get_statistics
from app.ai.tools.visualization_tools import get_visualization_recommendations
from app.services.outliers_service import get_outliers


def _schema(name: str, description: str, properties: dict[str, Any] | None = None, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


READ_TOOLS = [
    _schema("get_dataset_profile", "Inspect dataset shape, column groups, types, missing values, and duplicates."),
    _schema("get_dataset_preview", "Return only a small safe sample of current dataset rows.", {"limit": {"type": "integer", "minimum": 1, "maximum": 8}}),
    _schema("get_missing_values", "Get missing counts by column and overall percentage."),
    _schema("get_duplicates", "Get duplicate row count and percentage."),
    _schema("get_statistics", "Get descriptive statistics, optionally for one column.", {"column": {"type": "string"}}),
    _schema("get_outliers", "Get existing IQR-based outlier report."),
    _schema("get_data_health", "Get data-health score and component metrics."),
    _schema("get_visualization_recommendations", "Get existing chart recommendations."),
    _schema("get_history", "Get recorded preprocessing history."),
]

_HANDLERS = {
    "get_dataset_profile": lambda dataset_id, _arguments: get_dataset_profile(dataset_id),
    "get_dataset_preview": lambda dataset_id, arguments: get_dataset_preview(dataset_id, arguments.get("limit", 5)),
    "get_missing_values": lambda dataset_id, _arguments: get_missing_values(dataset_id),
    "get_duplicates": lambda dataset_id, _arguments: get_duplicates(dataset_id),
    "get_statistics": lambda dataset_id, arguments: get_statistics(dataset_id, arguments.get("column")),
    "get_outliers": lambda dataset_id, _arguments: get_outliers(dataset_id),
    "get_data_health": lambda dataset_id, _arguments: get_data_health(dataset_id),
    "get_visualization_recommendations": lambda dataset_id, _arguments: get_visualization_recommendations(dataset_id),
    "get_history": lambda dataset_id, _arguments: get_history(dataset_id),
}


def execute_read_tool(name: str, arguments: str | dict[str, Any], dataset_id: str) -> Any:
    if name not in _HANDLERS:
        raise ValueError("Tool is not allowed")
    parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must be an object")
    allowed_arguments = {
        "get_dataset_preview": {"limit"},
        "get_statistics": {"column"},
    }
    if set(parsed) - allowed_arguments.get(name, set()):
        raise ValueError("Unexpected tool arguments")
    return _HANDLERS[name](dataset_id, parsed)