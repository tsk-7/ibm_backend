from __future__ import annotations

from typing import Any

import pandas as pd

from app.utils.dataframe_utils import datetime_column_names, json_value, logical_column_kind


def _percentage(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def _numeric_column_profile(series: pd.Series) -> dict[str, Any]:
    non_missing = series.dropna()
    if non_missing.empty:
        return {
            "count": 0,
            "missing": 0,
            "mean": None,
            "median": None,
            "mode": None,
            "std": None,
            "variance": None,
            "min": None,
            "max": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "range": None,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
        }

    q1 = float(non_missing.quantile(0.25))
    q3 = float(non_missing.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((non_missing < lower_bound) | (non_missing > upper_bound)).sum())
    std = float(non_missing.std(ddof=1)) if len(non_missing) > 1 else 0.0
    variance = float(non_missing.var(ddof=1)) if len(non_missing) > 1 else 0.0
    mode = non_missing.mode(dropna=True)
    return {
        "count": int(non_missing.count()),
        "missing": int(series.isna().sum()),
        "mean": json_value(float(non_missing.mean())) if not non_missing.empty else None,
        "median": json_value(float(non_missing.median())) if not non_missing.empty else None,
        "mode": json_value(mode.iloc[0]) if not mode.empty else None,
        "std": json_value(std),
        "variance": json_value(variance),
        "min": json_value(float(non_missing.min())),
        "max": json_value(float(non_missing.max())),
        "q1": json_value(q1),
        "q3": json_value(q3),
        "iqr": json_value(iqr),
        "range": json_value(float(non_missing.max() - non_missing.min())),
        "outlier_count": outlier_count,
        "outlier_percentage": _percentage(outlier_count, len(non_missing)),
    }


def _categorical_column_profile(series: pd.Series, frame: pd.DataFrame) -> dict[str, Any]:
    non_missing = series.dropna()
    counts = non_missing.value_counts(dropna=True)
    top_values = [
        {
            "value": json_value(value),
            "count": int(count),
            "percentage": _percentage(int(count), int(len(non_missing))) if len(non_missing) else 0.0,
        }
        for value, count in counts.head(10).items()
    ]
    return {
        "count": int(non_missing.count()),
        "missing": int(series.isna().sum()),
        "unique_count": int(non_missing.nunique()),
        "unique_percentage": _percentage(int(non_missing.nunique()), len(series)) if len(series) else 0.0,
        "top_values": top_values,
        "mode": json_value(counts.index[0]) if not counts.empty else None,
        "mode_frequency": int(counts.iloc[0]) if not counts.empty else 0,
        "mode_percentage": _percentage(int(counts.iloc[0]), len(non_missing)) if len(non_missing) else 0.0,
    }


def _datetime_column_profile(series: pd.Series) -> dict[str, Any]:
    non_missing = series.dropna()
    unique_values = non_missing.sort_values().drop_duplicates()
    return {
        "count": int(non_missing.count()),
        "missing": int(series.isna().sum()),
        "unique_count": int(unique_values.nunique()),
        "minimum": json_value(non_missing.min()) if not non_missing.empty else None,
        "maximum": json_value(non_missing.max()) if not non_missing.empty else None,
        "date_range_days": None,
    }


def column_profile(frame: pd.DataFrame, column: Any) -> dict[str, Any]:
    series = frame[column]
    base = {
        "name": str(column),
        "dtype": str(series.dtype),
        "kind": logical_column_kind(frame, column),
        "null_count": int(series.isna().sum()),
        "null_percentage": _percentage(int(series.isna().sum()), len(frame)) if len(frame) else 0.0,
        "unique_count": int(series.nunique(dropna=True)),
        "unique_percentage": _percentage(int(series.nunique(dropna=True)), len(frame)) if len(frame) else 0.0,
        "minimum": None,
        "maximum": None,
    }
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        numeric = _numeric_column_profile(series)
        base.update(numeric)
    elif pd.api.types.is_datetime64_any_dtype(series.dtype) or column in datetime_column_names(frame):
        datetime_stats = _datetime_column_profile(series)
        base.update(datetime_stats)
    else:
        categorical = _categorical_column_profile(series, frame)
        base.update(categorical)
    return base


def profile_dataset(dataset_id: str) -> dict[str, Any]:
    from app.services.dataset_service import load_current

    dataset, frame = load_current(dataset_id)
    missing_total = int(frame.isna().sum().sum())
    total_cells = int(len(frame) * len(frame.columns)) if len(frame) and len(frame.columns) else 0
    duplicate_rows = int(frame.duplicated().sum())
    numeric_columns = [str(column) for column in frame.select_dtypes(include=["number"]).columns]
    datetime_columns = [str(column) for column in datetime_column_names(frame)]
    categorical_columns = [str(column) for column in frame.columns if str(column) not in numeric_columns + datetime_columns]
    column_profiles = [column_profile(frame, column) for column in frame.columns]
    return {
        "dataset_id": dataset.dataset_id,
        "general": {
            "row_count": int(len(frame)),
            "column_count": int(len(frame.columns)),
            "total_cells": total_cells,
            "duplicate_row_count": duplicate_rows,
            "duplicate_percentage": _percentage(duplicate_rows, len(frame)) if len(frame) else 0.0,
            "missing_cell_count": missing_total,
            "missing_percentage": _percentage(missing_total, total_cells) if total_cells else 0.0,
            "memory_usage_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
        },
        "columns": column_profiles,
        "numerical_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "summary": {
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "missing_total": missing_total,
            "duplicates": duplicate_rows,
        },
    }
