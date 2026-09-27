import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def json_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def dataframe_records(frame: pd.DataFrame) -> list[list[Any]]:
    return [[json_value(value) for value in row] for row in frame.itertuples(index=False, name=None)]


def missing_count(frame: pd.DataFrame, column: str | None = None) -> int:
    target = frame[column] if column is not None else frame
    return int(target.isna().sum().sum() if isinstance(target, pd.DataFrame) else target.isna().sum())


def datetime_column_names(frame: pd.DataFrame) -> list[Any]:
    columns = list(frame.select_dtypes(include=["datetime", "datetimetz"]).columns)
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        values = frame[column].dropna()
        if values.empty:
            continue
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.8:
            columns.append(column)
    return columns


def logical_column_kind(frame: pd.DataFrame, column: Any) -> str:
    series = frame[column]
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series.dtype) or column in datetime_column_names(frame):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numeric"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"
    non_missing = series.dropna()
    unique = int(non_missing.nunique())
    if unique > max(50, int(len(non_missing) * 0.5)):
        return "text"
    return "categorical"


def column_metadata(frame: pd.DataFrame) -> list[dict[str, Any]]:
    metadata = []
    for column in frame.columns:
        series = frame[column]
        metadata.append({
            "name": str(column),
            "dtype": str(series.dtype),
            "kind": logical_column_kind(frame, column),
            "nullable": bool(series.isna().any()),
            "unique_count": int(series.nunique(dropna=True)),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": round(100 * int(series.isna().sum()) / len(frame), 2) if len(frame) else 0.0,
            "non_missing_count": int(series.notna().sum()),
        })
    return metadata
