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
from app.utils.dataframe_utils import column_metadata, dataframe_records, json_value, missing_count
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
        raise ValueError(f"Column '{column}' does not exist in this dataset.")


def _series_stats(series: pd.Series) -> dict[str, Any]:
    non_missing = series.dropna()
    stats: dict[str, Any] = {
        "count": int(non_missing.count()),
        "missing_count": int(series.isna().sum()),
    }
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        stats.update({
            "mean": json_value(float(non_missing.mean())) if not non_missing.empty else None,
            "median": json_value(float(non_missing.median())) if not non_missing.empty else None,
            "mode": json_value(non_missing.mode(dropna=True).iloc[0]) if not non_missing.empty and not non_missing.mode(dropna=True).empty else None,
            "std": json_value(float(non_missing.std(ddof=1))) if len(non_missing) > 1 else 0.0,
            "variance": json_value(float(non_missing.var(ddof=1))) if len(non_missing) > 1 else 0.0,
            "min": json_value(float(non_missing.min())) if not non_missing.empty else None,
            "max": json_value(float(non_missing.max())) if not non_missing.empty else None,
            "q1": json_value(float(non_missing.quantile(0.25))) if not non_missing.empty else None,
            "q3": json_value(float(non_missing.quantile(0.75))) if not non_missing.empty else None,
            "iqr": json_value(float(non_missing.quantile(0.75) - non_missing.quantile(0.25))) if not non_missing.empty else None,
        })
    elif pd.api.types.is_datetime64_any_dtype(series.dtype):
        stats.update({
            "min": json_value(non_missing.min()) if not non_missing.empty else None,
            "max": json_value(non_missing.max()) if not non_missing.empty else None,
            "unique_count": int(non_missing.nunique()),
        })
    else:
        counts = non_missing.value_counts()
        stats.update({
            "unique_count": int(non_missing.nunique()),
            "mode": json_value(counts.index[0]) if not counts.empty else None,
            "top_values": [
                {"value": json_value(value), "count": int(count), "percentage": round(100 * count / len(non_missing), 2) if len(non_missing) else 0.0}
                for value, count in counts.head(5).items()
            ],
        })
    return stats


def preprocessing_summary(dataset_id: str) -> dict:
    dataset, frame = load_current(dataset_id)
    columns = column_metadata(frame)
    total_cells = len(frame) * len(frame.columns)
    missing_total = int(frame.isna().sum().sum())
    affected = [column for column in columns if column["missing_count"]]
    highest_count = max(columns, key=lambda item: item["missing_count"], default=None)
    highest_percentage = max(
        columns,
        key=lambda item: item["missing_count"] / len(frame) if len(frame) else 0,
        default=None,
    )
    return {
        "dataset_id": dataset.dataset_id,
        "rows": int(len(frame)),
        "columns_count": int(len(frame.columns)),
        "missing": {
            "total_cells": missing_total,
            "percentage": round(100 * missing_total / total_cells, 2) if total_cells else 0.0,
            "columns_affected": len(affected),
            "columns": [
                {
                    "column": item["name"],
                    "dtype": item["dtype"],
                    "kind": item["kind"],
                    "missing_count": item["missing_count"],
                    "missing_percentage": round(100 * item["missing_count"] / len(frame), 2) if len(frame) else 0.0,
                    "non_missing_count": len(frame) - item["missing_count"],
                }
                for item in columns
            ],
            "highest_count_column": highest_count["name"] if highest_count and highest_count["missing_count"] else None,
            "highest_percentage_column": highest_percentage["name"] if highest_percentage and highest_percentage["missing_count"] else None,
            "supported_methods": ["remove_rows", "mean", "median", "mode", "custom", "ffill", "bfill"],
        },
        "duplicates": {
            "count": int(frame.duplicated().sum()),
            "percentage": round(100 * int(frame.duplicated().sum()) / len(frame), 2) if len(frame) else 0.0,
            "rows": int(len(frame)),
        },
        "columns": columns,
    }


def _validate_missing_values(frame: pd.DataFrame, column: str | None, method: str, value: Any) -> list[str]:
    supported = {"remove_rows", "mean", "median", "mode", "custom", "ffill", "bfill"}
    if method not in supported:
        raise ValueError("Unsupported missing-value method.")
    if column is not None:
        _require_column(frame, column)
    targets = [column] if column is not None else list(frame.columns)
    if not targets or missing_count(frame, column) == 0:
        label = f"Column '{column}'" if column is not None else "Dataset"
        raise ValueError(f"{label} contains no missing values.")
    if method in {"mean", "median"}:
        numeric = [target for target in targets if pd.api.types.is_numeric_dtype(frame[target])]
        if column is not None and not numeric:
            label = "Mean" if method == "mean" else "Median"
            raise ValueError(f"{label} imputation requires a numerical column.")
        if not numeric:
            label = "Mean" if method == "mean" else "Median"
            raise ValueError(f"{label} imputation requires at least one numerical column.")
        targets = numeric
    if method == "custom" and value is None:
        raise ValueError("A non-null value is required for custom imputation.")
    if method in {"mean", "median", "mode"}:
        for target in targets:
            values = frame[target]
            if values.isna().any() and values.dropna().empty:
                label = "mean" if method == "mean" else "median" if method == "median" else "mode"
                raise ValueError(f"Cannot calculate {label} for entirely missing column '{target}'.")
    return targets


def preview_missing_values(dataset_id: str, column: str | None, method: str, value: Any = None) -> dict:
    _, frame = load_current(dataset_id)
    targets = _validate_missing_values(frame, column, method, value)
    before = missing_count(frame, column)
    replacement: Any = None
    if method in {"mean", "median", "mode"}:
        replacements = {}
        for target in targets:
            series = frame[target]
            if method == "mean":
                selected = series.mean()
            elif method == "median":
                selected = series.median()
            else:
                modes = series.mode(dropna=True)
                selected = modes.iloc[0] if not modes.empty else None
            replacements[target] = json_value(selected)
        replacement = replacements.get(column) if column is not None else replacements
    elif method == "custom":
        replacement = json_value(value)
    preview = frame.copy()
    if method == "remove_rows":
        preview = preview.dropna(subset=[column] if column else None)
    elif method in {"mean", "median", "mode"}:
        for target in targets:
            series = preview[target]
            if method == "mean":
                selected = series.mean()
            elif method == "median":
                selected = series.median()
            else:
                modes = series.mode(dropna=True)
                selected = modes.iloc[0] if not modes.empty else None
            if selected is not None:
                preview[target] = series.fillna(selected)
    elif method == "custom":
        if column is None:
            preview = preview.fillna(value)
        else:
            preview[column] = preview[column].fillna(value)
    elif method == "ffill":
        if column is None:
            preview = preview.ffill()
        else:
            preview[column] = preview[column].ffill()
    elif method == "bfill":
        if column is None:
            preview = preview.bfill()
        else:
            preview[column] = preview[column].bfill()

    affected = int(frame[column].isna().sum()) if column else int(frame.isna().sum().sum())
    before_stats = _series_stats(frame[column]) if column else {"all_columns": {name: _series_stats(frame[name]) for name in frame.columns}}
    after_stats = _series_stats(preview[column]) if column else {"all_columns": {name: _series_stats(preview[name]) for name in preview.columns}}
    impact: dict[str, Any] = {"rows_affected": affected}
    if column is not None and pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column]):
        before_mean = frame[column].mean()
        after_mean = preview[column].mean()
        before_std = frame[column].std(ddof=1) if frame[column].dropna().count() > 1 else 0.0
        after_std = preview[column].std(ddof=1) if preview[column].dropna().count() > 1 else 0.0
        impact.update({
            "mean_change": json_value(after_mean - before_mean),
            "std_change": json_value(after_std - before_std),
            "variance_change": json_value((preview[column].var(ddof=1) if preview[column].dropna().count() > 1 else 0.0) - (frame[column].var(ddof=1) if frame[column].dropna().count() > 1 else 0.0)),
        })
    return {
        "dataset_id": dataset_id,
        "column": column,
        "method": method,
        "missing_before": before,
        "estimated_missing_after": missing_count(preview, column),
        "replacement_value": replacement,
        "rows_affected": affected,
        "before": before_stats,
        "after": after_stats,
        "impact": impact,
    }


def preview_standardization(dataset_id: str, column: str) -> dict:
    _, frame = load_current(dataset_id)
    _require_column(frame, column)
    series = frame[column].dropna()
    if series.empty:
        raise ValueError(f"Column '{column}' contains no non-missing values.")
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    if std == 0:
        standardized = pd.Series(0.0, index=series.index)
    else:
        standardized = (frame[column] - mean) / std
    before = _series_stats(frame[column])
    after = _series_stats(standardized)
    return {
        "dataset_id": dataset_id,
        "column": column,
        "method": "standardization",
        "before": before,
        "after": after,
        "mean_before": json_value(mean),
        "std_before": json_value(std),
        "mean_after": json_value(float(standardized.mean())),
        "std_after": json_value(float(standardized.std(ddof=1))) if len(standardized.dropna()) > 1 else 0.0,
    }


def preview_normalization(dataset_id: str, column: str) -> dict:
    _, frame = load_current(dataset_id)
    _require_column(frame, column)
    values = frame[column].dropna()
    if values.empty:
        raise ValueError(f"Column '{column}' contains no non-missing values.")
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum == minimum:
        normalized = pd.Series(0.0, index=frame.index)
    else:
        normalized = (frame[column] - minimum) / (maximum - minimum)
    before = _series_stats(frame[column])
    after = _series_stats(normalized)
    return {
        "dataset_id": dataset_id,
        "column": column,
        "method": "normalization",
        "before": before,
        "after": after,
        "min_before": json_value(minimum),
        "max_before": json_value(maximum),
        "rows_affected": int(frame[column].isna().sum()),
        "min_after": json_value(float(normalized.min())),
        "max_after": json_value(float(normalized.max())),
    }


def preview_iqr_analysis(dataset_id: str, column: str) -> dict:
    _, frame = load_current(dataset_id)
    _require_column(frame, column)
    series = frame[column].dropna()
    if series.empty:
        raise ValueError(f"Column '{column}' contains no non-missing values.")
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    outlier_count = int(outlier_mask.sum())
    outlier_values = [json_value(value) for value in series[outlier_mask].head(10).tolist()]
    return {
        "dataset_id": dataset_id,
        "column": column,
        "method": "iqr_outlier_analysis",
        "q1": json_value(q1),
        "q3": json_value(q3),
        "iqr": json_value(iqr),
        "lower_bound": json_value(lower_bound),
        "upper_bound": json_value(upper_bound),
        "outlier_count": outlier_count,
        "outlier_percentage": round(100.0 * outlier_count / len(series), 2) if len(series) else 0.0,
        "rows_affected": outlier_count,
        "sample_outlier_values": outlier_values,
        "before": _series_stats(frame[column]),
    }


def process_missing_values(dataset_id: str, column: str | None, method: str, value: Any = None) -> dict:
    _, frame = load_current(dataset_id)
    targets = _validate_missing_values(frame, column, method, value)
    before_rows = len(frame)
    before_missing = missing_count(frame, column)
    if method == "remove_rows":
        frame = frame.dropna(subset=[column] if column else None)
    elif method == "mean":
        for target in targets:
            mean = frame[target].mean()
            if pd.isna(mean) and frame[target].isna().any():
                raise ValueError(f"Cannot calculate mean for entirely missing column: {target}")
            frame[target] = frame[target].fillna(mean)
    elif method == "median":
        for target in targets:
            median = frame[target].median()
            if pd.isna(median) and frame[target].isna().any():
                raise ValueError(f"Cannot calculate median for entirely missing column: {target}")
            frame[target] = frame[target].fillna(median)
    elif method == "mode":
        for target in targets:
            modes = frame[target].mode(dropna=True)
            if modes.empty and frame[target].isna().any():
                raise ValueError(f"Cannot calculate mode for entirely missing column: {target}")
            if not modes.empty:
                frame[target] = frame[target].fillna(modes.iloc[0])
    elif method == "custom":
        if column:
            frame[column] = frame[column].fillna(value)
        else:
            frame = frame.fillna(value)
    elif method in {"ffill", "bfill"}:
        if column is None:
            frame = frame.ffill() if method == "ffill" else frame.bfill()
        else:
            frame[column] = frame[column].ffill() if method == "ffill" else frame[column].bfill()
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
        return {
            "duplicate_count": duplicate_count,
            "count": duplicate_count,
            "percentage": round(100 * duplicate_count / rows_before, 2) if rows_before else 0.0,
            "rows_before": rows_before,
            "rows_after": rows_before,
        }
    if action != "remove":
        raise ValueError("Invalid duplicate action")
    if duplicate_count == 0:
        return {"duplicate_count": 0, "count": 0, "percentage": 0.0, "rows_before": rows_before,
            "rows_after": rows_before, "duplicates_removed": 0, "success": True}
    frame = frame.drop_duplicates()
    _commit(dataset_id, frame, "Remove duplicates", None, rows_before, f"{duplicate_count} duplicate rows removed")
    return {"duplicate_count": duplicate_count, "count": duplicate_count,
            "percentage": round(100 * duplicate_count / rows_before, 2) if rows_before else 0.0,
            "rows_before": rows_before, "rows_after": len(frame),
            "duplicates_removed": rows_before - len(frame), "success": True}


def preview_duplicates(dataset_id: str, sample_limit: int = 10) -> dict:
    _, frame = load_current(dataset_id)
    duplicate_mask = frame.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    samples = dataframe_records(frame.loc[duplicate_mask].head(sample_limit))
    return {
        "dataset_id": dataset_id,
        "duplicate_count": duplicate_count,
        "percentage": round(100 * duplicate_count / len(frame), 2) if len(frame) else 0.0,
        "rows_before": int(len(frame)),
        "estimated_rows_after_removal": int(len(frame) - duplicate_count),
        "sample_columns": [str(column) for column in frame.columns],
        "sample_duplicate_rows": samples,
    }


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
