import pandas as pd
import numpy as np

from app.services.dataset_service import load_current


def _percentage(numerator: int, denominator: int) -> float:
    return round((100.0 * numerator / denominator), 2) if denominator else 100.0


def dataset_health(dataset_id: str) -> dict:
    _, frame = load_current(dataset_id)
    cells = len(frame) * len(frame.columns)
    missing = int(frame.isna().sum().sum())
    duplicate_rows = int(frame.duplicated().sum())
    duplicate_percentage = round(100.0 * duplicate_rows / len(frame), 2) if len(frame) else 0.0
    missing_percentage = round(100.0 * missing / cells, 2) if cells else 0.0
    numeric_columns = list(frame.select_dtypes(include=["number"]).columns)
    numeric_cells = int(frame[numeric_columns].notna().sum().sum()) if numeric_columns else 0
    outlier_cells = 0
    for column in numeric_columns:
        values = frame[column].dropna()
        finite_values = values[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))]
        outlier_cells += len(values) - len(finite_values)
        if finite_values.empty:
            continue
        q1, q3 = finite_values.quantile([0.25, 0.75])
        spread = q3 - q1
        outlier_cells += int(((finite_values < q1 - 1.5 * spread) | (finite_values > q3 + 1.5 * spread)).sum())
    outlier_percentage = round(100.0 * outlier_cells / numeric_cells, 2) if numeric_cells else 0.0
    completeness = round(100.0 - missing_percentage, 2)
    consistency = round(100.0 - duplicate_percentage, 2)
    validity = round(100.0 - outlier_percentage, 2)
    overall = round((completeness + consistency + validity) / 3, 2)
    return {
        "overall_score": overall,
        "completeness": completeness,
        "consistency": consistency,
        "validity": validity,
        "duplicate_percentage": duplicate_percentage,
        "missing_percentage": missing_percentage,
        "outlier_percentage": outlier_percentage,
        "duplicates": duplicate_percentage,
        "outliers": outlier_percentage,
        "scoring": {
            "completeness": "100 - missing cells / all cells * 100",
            "consistency": "100 - duplicate rows / all rows * 100",
            "validity": "100 - numeric outlier cells / non-missing numeric cells * 100; outliers use 1.5*IQR bounds",
            "overall_score": "Arithmetic mean of completeness, consistency, and validity",
            "empty_denominator": "A metric with no applicable values scores 100; its reported percentage is 0",
        },
    }