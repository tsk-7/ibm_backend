import pandas as pd

from app.services.dataset_service import load_current
from app.utils.dataframe_utils import json_value


def calculate_statistics(dataset_id: str) -> dict:
    _, frame = load_current(dataset_id)
    numerical = {}
    categorical = {}
    for column in frame.select_dtypes(include=["number"]).columns:
        values = frame[column].dropna()
        numerical[str(column)] = {
            "count": int(values.count()),
            "mean": json_value(values.mean()),
            "median": json_value(values.median()),
            "standard_deviation": json_value(values.std()),
            "minimum": json_value(values.min()) if not values.empty else None,
            "maximum": json_value(values.max()) if not values.empty else None,
            "q1": json_value(values.quantile(0.25)),
            "q3": json_value(values.quantile(0.75)),
        }
    categorical_columns = [column for column in frame.columns if column not in frame.select_dtypes(include=["number"]).columns]
    for column in categorical_columns:
        values = frame[column].dropna()
        counts = values.value_counts()
        categorical[str(column)] = {
            "count": int(values.count()),
            "unique_values": int(values.nunique()),
            "most_frequent_value": json_value(counts.index[0]) if not counts.empty else None,
            "most_frequent_count": int(counts.iloc[0]) if not counts.empty else 0,
        }
    return {"numerical": numerical, "categorical": categorical}