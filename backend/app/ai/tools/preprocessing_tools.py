from typing import Any

from app.services.preprocessing_service import (
    convert_dtype,
    create_column,
    delete_column,
    process_duplicates,
    process_missing_values,
    rename_column,
)


def propose_write_action(dataset_id: str, action_type: str, parameters: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "fill_missing_values": ({"column", "method", "value"}, {"method"}),
        "remove_duplicates": (set(), set()),
        "convert_dtype": ({"column", "dtype"}, {"column", "dtype"}),
        "rename_column": ({"old_name", "new_name"}, {"old_name", "new_name"}),
        "delete_column": ({"column"}, {"column"}),
        "create_column": ({"name", "expression"}, {"name", "expression"}),
    }
    if action_type not in fields:
        raise ValueError("Unsupported action")
    allowed, required = fields[action_type]
    if set(parameters) - allowed or not required.issubset(parameters):
        raise ValueError("Unexpected action parameters")
    if action_type == "fill_missing_values" and parameters["method"] not in {"remove_rows", "mean", "median", "mode", "custom", "ffill", "bfill"}:
        raise ValueError("Unsupported missing-value method")
    if action_type == "convert_dtype" and parameters["dtype"] not in {"integer", "float", "string", "boolean", "datetime"}:
        raise ValueError("Unsupported data type")
    return {"type": action_type, "dataset_id": dataset_id, "parameters": parameters}


def execute_write_action(action: dict[str, Any]) -> dict[str, Any]:
    dataset_id = action["dataset_id"]
    parameters = action["parameters"]
    action_type = action["type"]
    if action_type == "fill_missing_values":
        return process_missing_values(
            dataset_id, parameters.get("column"), parameters["method"], parameters.get("value")
        )
    if action_type == "remove_duplicates":
        return process_duplicates(dataset_id, "remove")
    if action_type == "convert_dtype":
        return convert_dtype(dataset_id, parameters["column"], parameters["dtype"])
    if action_type == "rename_column":
        return rename_column(dataset_id, parameters["old_name"], parameters["new_name"])
    if action_type == "delete_column":
        return delete_column(dataset_id, parameters["column"])
    if action_type == "create_column":
        return create_column(dataset_id, parameters["name"], parameters["expression"])
    raise ValueError("Unsupported action")