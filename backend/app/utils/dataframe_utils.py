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
