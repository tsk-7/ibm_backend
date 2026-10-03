import pandas as pd

from app.services.dataset_profiler import profile_dataset
from app.services.dataset_service import load_current
from app.utils.dataframe_utils import json_value


def calculate_statistics(dataset_id: str) -> dict:
    _, frame = load_current(dataset_id)
    profile = profile_dataset(dataset_id)
    numerical = {}
    categorical = {}
    for column in profile["columns"]:
        if column["kind"] == "numeric":
            values = frame[column["name"]].dropna()
            numerical[str(column["name"])] = {
                "count": int(values.count()),
                "missing": int(column["null_count"]),
                "mean": json_value(column.get("mean")),
                "median": json_value(column.get("median")),
                "mode": json_value(column.get("mode")),
                "standard_deviation": json_value(column.get("std")),
                "variance": json_value(column.get("variance")),
                "minimum": json_value(column.get("min")),
                "maximum": json_value(column.get("max")),
                "q1": json_value(column.get("q1")),
                "q3": json_value(column.get("q3")),
                "iqr": json_value(column.get("iqr")),
                "range": json_value(column.get("range")),
                "outlier_count": int(column.get("outlier_count", 0)),
                "outlier_percentage": column.get("outlier_percentage", 0.0),
            }
        elif column["kind"] in {"categorical", "boolean", "text"}:
            values = frame[column["name"]].dropna()
            counts = values.value_counts()
            categorical[str(column["name"])] = {
                "count": int(values.count()),
                "missing": int(column["null_count"]),
                "unique_values": int(column["unique_count"]),
                "most_frequent_value": json_value(column.get("mode")),
                "most_frequent_count": int(column.get("mode_frequency", 0)),
                "top_values": column.get("top_values", []),
            }
    return {"numerical": numerical, "categorical": categorical}