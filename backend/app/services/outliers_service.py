import numpy as np

from app.services.dataset_service import load_current
from app.utils.dataframe_utils import json_value


def get_outliers(dataset_id: str) -> dict:
    _, frame = load_current(dataset_id)
    result = {}
    for column in frame.select_dtypes(include=["number"]).columns:
        values = frame[column].dropna()
        finite_values = values[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))]
        non_finite_count = len(values) - len(finite_values)
        if finite_values.empty:
            result[str(column)] = {
                "q1": None, "q3": None, "iqr": None, "lower_bound": None,
                "upper_bound": None, "outlier_count": non_finite_count,
                "outlier_percentage": round(100.0 * non_finite_count / len(values), 2) if len(values) else 0.0,
            }
            continue
        q1 = float(finite_values.quantile(0.25))
        q3 = float(finite_values.quantile(0.75))
        spread = q3 - q1
        lower = q1 - 1.5 * spread
        upper = q3 + 1.5 * spread
        count = non_finite_count + int(((finite_values < lower) | (finite_values > upper)).sum())
        result[str(column)] = {
            "q1": json_value(q1),
            "q3": json_value(q3),
            "iqr": json_value(spread),
            "lower_bound": json_value(lower),
            "upper_bound": json_value(upper),
            "outlier_count": count,
            "outlier_percentage": round(100.0 * count / len(values), 2) if len(values) else 0.0,
        }
    return result