import pandas as pd
import numpy as np

from app.services.dataset_service import load_current


def _percentage(numerator: int, denominator: int) -> float:
    return round((100.0 * numerator / denominator), 2) if denominator else 100.0


def dataset_health(dataset_id: str) -> dict:
    _, frame = load_current(dataset_id)
    from app.services.dataset_profiler import profile_dataset

    profile = profile_dataset(dataset_id)
    columns = profile["columns"]
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

    column_health = []
    for column in columns:
        status = "good"
        reasons = []
        if column["null_count"] > 0:
            status = "warning"
            reasons.append(f"{column['null_count']} missing values")
        if column.get("outlier_count", 0) > 0:
            status = "critical" if status == "warning" else "warning"
            reasons.append(f"{column['outlier_count']} outliers")
        if column.get("kind") in {"text", "categorical"} and column["unique_count"] > 50:
            status = "warning"
            reasons.append("high cardinality")
        column_health.append({
            "column": column["name"],
            "status": status,
            "missing": column["null_count"],
            "missing_percentage": column["null_percentage"],
            "duplicates": None,
            "outliers": column.get("outlier_count"),
            "dtype_concerns": None,
            "reasons": reasons,
        })

    recommendations = []
    if missing > 0:
        recommendations.append(f"{missing} missing values are present across the dataset; review affected columns before analysis.")
    if duplicate_rows > 0:
        recommendations.append(f"{duplicate_rows} duplicate rows were detected and may distort counts or visual summaries.")
    if outlier_cells > 0:
        recommendations.append(f"{outlier_cells} numeric outlier values were flagged using the 1.5×IQR rule.")
    if not recommendations:
        recommendations.append("No structural quality issues were detected in the current dataset profile.")

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
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_health": column_health,
        "recommendations": recommendations,
        "scoring": {
            "completeness": "100 - missing cells / all cells * 100",
            "consistency": "100 - duplicate rows / all rows * 100",
            "validity": "100 - numeric outlier cells / non-missing numeric cells * 100; outliers use 1.5*IQR bounds",
            "overall_score": "Arithmetic mean of completeness, consistency, and validity",
            "empty_denominator": "A metric with no applicable values scores 100; its reported percentage is 0",
        },
    }