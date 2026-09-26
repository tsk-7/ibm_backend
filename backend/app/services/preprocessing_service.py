import ast
import operator
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.database.database import get_connection
from app.services.dataset_service import load_current
from app.utils.dataframe_utils import missing_count
from app.utils.file_utils import data_directory, save_dataframe_csv, utc_now


def _commit(dataset_id: str, frame: pd.DataFrame, operation: str, column_name: str | None,
            rows_before: int, details: str) -> dict:
    dataset, _ = load_current(dataset_id)
    current = Path(dataset.processed_path)
    backup_dir = data_directory("backups")
    backup_path = backup_dir / f"{dataset_id}_{utc_now().replace(':', '').replace('+', '_')}_{uuid.uuid4().hex[:8]}.csv"
    shutil.copy2(current, backup_path)
    save_dataframe_csv(frame, current)
    now = utc_now()
    with get_connection() as connection:
        connection.execute(
            "UPDATE datasets SET `rows` = ?, `columns` = ?, updated_at = ? WHERE dataset_id = ?",
            (len(frame), len(frame.columns), now, dataset_id),
        )
        connection.execute(
            "INSERT INTO history (dataset_id, operation, column_name, rows_before, rows_after, details, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (dataset_id, operation, column_name, rows_before, len(frame), details, now),
        )
    return {
        "success": True,
        "rows_before": rows_before,
        "rows_after": len(frame),
        "message": details,
    }


def _require_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        raise ValueError(f"Column does not exist: {column}")


def process_missing_values(dataset_id: str, column: str | None, method: str, value: Any = None) -> dict:
    _, frame = load_current(dataset_id)
    if column is not None:
        _require_column(frame, column)
    before_rows = len(frame)
    before_missing = missing_count(frame, column)
    if method == "remove_rows":
        frame = frame.dropna(subset=[column] if column else None)
    elif method == "mean":
        targets = [column] if column else list(frame.select_dtypes(include=["number"]).columns)
        if not targets:
            raise ValueError("Mean imputation requires at least one numerical column")
        for target in targets:
            if not pd.api.types.is_numeric_dtype(frame[target]):
                raise ValueError(f"Mean imputation requires a numerical column: {target}")
            mean = frame[target].mean()
            if pd.isna(mean) and frame[target].isna().any():
                raise ValueError(f"Cannot calculate mean for entirely missing column: {target}")
            frame[target] = frame[target].fillna(mean)
    elif method == "median":
        targets = [column] if column else list(frame.select_dtypes(include=["number"]).columns)
        if not targets:
            raise ValueError("Median imputation requires at least one numerical column")
        for target in targets:
            if not pd.api.types.is_numeric_dtype(frame[target]):
                raise ValueError(f"Median imputation requires a numerical column: {target}")
            median = frame[target].median()
            if pd.isna(median) and frame[target].isna().any():
                raise ValueError(f"Cannot calculate median for entirely missing column: {target}")
            frame[target] = frame[target].fillna(median)
    elif method == "mode":
        targets = [column] if column else list(frame.columns)
        for target in targets:
            modes = frame[target].mode(dropna=True)
            if modes.empty and frame[target].isna().any():
                raise ValueError(f"Cannot calculate mode for entirely missing column: {target}")
            if not modes.empty:
                frame[target] = frame[target].fillna(modes.iloc[0])
    elif method == "custom":
        if value is None:
            raise ValueError("A non-null value is required for custom imputation")
        if column:
            frame[column] = frame[column].fillna(value)
        else:
            frame = frame.fillna(value)
    elif method in {"ffill", "bfill"}:
        frame = frame.ffill() if method == "ffill" else frame.bfill()
    else:
        raise ValueError("Invalid preprocessing method")
    after_missing = missing_count(frame, column)
    details = f"Missing values processed using {method}; {before_missing - after_missing} missing values resolved"
    result = _commit(dataset_id, frame, "Process missing values", column, before_rows, details)
    result.update({"missing_before": before_missing, "missing_after": after_missing})
    return result


def process_duplicates(dataset_id: str, action: str) -> dict:
    _, frame = load_current(dataset_id)
    rows_before = len(frame)
    duplicate_count = int(frame.duplicated().sum())
    if action == "detect":
        return {"duplicate_count": duplicate_count, "rows_before": rows_before, "rows_after": rows_before}
    if action != "remove":
        raise ValueError("Invalid duplicate action")
    if duplicate_count == 0:
        return {"duplicate_count": 0, "rows_before": rows_before, "rows_after": rows_before, "success": True}
    frame = frame.drop_duplicates()
    _commit(dataset_id, frame, "Remove duplicates", None, rows_before, f"{duplicate_count} duplicate rows removed")
    return {"duplicate_count": duplicate_count, "rows_before": rows_before, "rows_after": len(frame), "success": True}


def _convert_dtype(series: pd.Series, dtype: str) -> pd.Series:
    if dtype == "integer":
        numbers = pd.to_numeric(series, errors="raise")
        non_null = numbers.dropna()
        if not np.equal(np.mod(non_null, 1), 0).all():
            raise ValueError("Values contain non-integral numbers")
        return numbers.astype("Int64" if numbers.isna().any() else "int64")
    if dtype == "float":
        return pd.to_numeric(series, errors="raise").astype(float)
    if dtype == "string":
        return series.astype("string")
    if dtype == "boolean":
        mapping = {"true": True, "false": False, "1": True, "0": False,
                   "yes": True, "no": False, "y": True, "n": False}

        def convert(value: Any) -> Any:
            if pd.isna(value):
                return pd.NA
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            key = str(value).strip().lower()
            if key not in mapping:
                raise ValueError(f"Cannot convert value to boolean: {value}")
            return mapping[key]

        return series.map(convert).astype("boolean")
    if dtype == "datetime":
        converted = pd.to_datetime(series, errors="coerce")
        invalid = series.notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Cannot convert value to datetime: {series[invalid].iloc[0]}")
        return converted
    raise ValueError("Invalid data type")


def convert_dtype(dataset_id: str, column: str, dtype: str) -> dict:
    dataset, frame = load_current(dataset_id)
    _require_column(frame, column)
    rows_before = len(frame)
    try:
        frame[column] = _convert_dtype(frame[column], dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Cannot safely convert column '{column}' to {dtype}: {exc}") from exc
    return _commit(dataset_id, frame, "Convert data type", column, rows_before,
                   f"Column '{column}' converted to {dtype}")


def rename_column(dataset_id: str, old_name: str, new_name: str) -> dict:
    dataset, frame = load_current(dataset_id)
    _require_column(frame, old_name)
    if not new_name.strip():
        raise ValueError("New column name cannot be empty")
    if new_name in frame.columns and new_name != old_name:
        raise ValueError(f"Column already exists: {new_name}")
    rows_before = len(frame)
    frame = frame.rename(columns={old_name: new_name})
    return _commit(dataset_id, frame, "Rename column", old_name, rows_before,
                   f"Column '{old_name}' renamed to '{new_name}'")


def delete_column(dataset_id: str, column: str) -> dict:
    dataset, frame = load_current(dataset_id)
    _require_column(frame, column)
    if len(frame.columns) == 1:
        raise ValueError("Cannot delete the last column in a dataset")
    rows_before = len(frame)
    frame = frame.drop(columns=[column])
    return _commit(dataset_id, frame, "Delete column", column, rows_before, f"Column '{column}' deleted")


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _evaluate_expression(node: ast.AST, frame: pd.DataFrame):
    if isinstance(node, ast.Expression):
        return _evaluate_expression(node.body, frame)
    if isinstance(node, ast.Name):
        if node.id not in frame.columns:
            raise ValueError(f"Unknown column in operation: {node.id}")
        return frame[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _evaluate_expression(node.operand, frame)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_expression(node.left, frame)
        right = _evaluate_expression(node.right, frame)
        try:
            return _BINARY_OPERATORS[type(node.op)](left, right)
        except (TypeError, ZeroDivisionError) as exc:
            raise ValueError(f"Invalid calculated-column operation: {exc}") from exc
    raise ValueError("Operation supports only column names, numeric constants, and +, -, *, /")


def create_column(dataset_id: str, name: str, expression: str) -> dict:
    dataset, frame = load_current(dataset_id)
    if name in frame.columns:
        raise ValueError(f"Column already exists: {name}")
    if not name.strip():
        raise ValueError("Column name cannot be empty")
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid calculated-column expression") from exc
    frame[name] = _evaluate_expression(parsed, frame)
    return _commit(dataset_id, frame, "Create calculated column", name, len(frame),
                   f"Calculated column '{name}' created from a safe arithmetic expression")
