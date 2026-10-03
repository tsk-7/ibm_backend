import itertools
from typing import Any

import numpy as np
import pandas as pd

from app.services.dataset_service import load_current
from app.utils.dataframe_utils import datetime_column_names, json_value, logical_column_kind


_AGGREGATIONS = {"count", "sum", "mean", "median", "mode", "std"}


def _require_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        raise ValueError(f"Column '{column}' does not exist in this dataset.")


def _numeric(frame: pd.DataFrame, column: str) -> bool:
    return pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column])


def _aggregate(values: pd.Series, aggregation: str, column: str) -> Any:
    if aggregation not in _AGGREGATIONS:
        raise ValueError(f"Unsupported aggregation: {aggregation}.")
    if aggregation in {"sum", "mean", "median", "std"} and not pd.api.types.is_numeric_dtype(values):
        label = "Mean" if aggregation == "mean" else aggregation.title()
        raise ValueError(f"{label} aggregation requires a numerical column.")
    if aggregation == "count":
        return int(values.count())
    if aggregation == "sum":
        return values.sum()
    if aggregation == "mean":
        return values.mean()
    if aggregation == "median":
        return values.median()
    if aggregation == "std":
        return values.std()
    modes = values.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else None


def _aggregate_name(value: str) -> str:
    return "mean" if value == "average" else value


def _title_aggregate(aggregation: str) -> str:
    return {"mean": "Average", "average": "Average", "median": "Median", "mode": "Mode",
            "sum": "Sum", "std": "Standard Deviation", "count": "Count"}.get(aggregation, "")


def prepare_visualization(dataset_id: str, chart_type: str, column_a: str, column_b: str | None = None,
                          aggregation: str = "none", bins: int = 10, limit: int = 1000) -> dict:
    _, frame = load_current(dataset_id)
    _require_column(frame, column_a)
    if column_b is not None:
        _require_column(frame, column_b)
    aggregation = _aggregate_name(aggregation)
    if aggregation != "none" and aggregation not in _AGGREGATIONS:
        raise ValueError(f"Unsupported aggregation: {aggregation}.")

    result: dict[str, Any] = {
        "chart_type": chart_type,
        "dataset_id": dataset_id,
        "columns": {"x": column_a},
        "aggregation": aggregation if aggregation != "none" else None,
        "metadata": {"rows_in_dataset": int(len(frame))},
    }
    if chart_type == "histogram":
        if not _numeric(frame, column_a):
            raise ValueError("Histogram requires one numerical column.")
        values = frame[column_a].dropna().to_numpy(dtype=float)
        counts, edges = np.histogram(values, bins=bins)
        result.update({
            "title": f"{column_a} Distribution",
            "columns": {"value": column_a},
            "bins": [float(edge) for edge in edges],
            "counts": [int(count) for count in counts],
            "metadata": {"rows_in_dataset": int(len(frame)), "non_missing_values": int(len(values)), "bin_count": bins},
        })
        return result

    if chart_type == "scatter":
        if column_b is None or not _numeric(frame, column_a) or not _numeric(frame, column_b):
            raise ValueError("Scatter plots require two numerical columns.")
        if aggregation != "none":
            raise ValueError("Scatter plots display individual rows and do not support aggregation.")
        points = frame[[column_a, column_b]].dropna().head(limit)
        result.update({
            "title": f"{column_a} vs {column_b}",
            "columns": {"x": column_a, "y": column_b},
            "aggregation": None,
            "x": [json_value(value) for value in points[column_a]],
            "y": [json_value(value) for value in points[column_b]],
            "metadata": {"rows_in_dataset": int(len(frame)), "points_returned": int(len(points)),
                         "preview_limit": limit, "truncated": len(frame[[column_a, column_b]].dropna()) > len(points)},
        })
        return result

    if chart_type == "line":
        if column_b is None or not (_numeric(frame, column_a) or column_a in datetime_column_names(frame)) or not _numeric(frame, column_b):
            raise ValueError("Line charts require an ordered or datetime X column and a numerical Y column.")
        if aggregation not in {"none", "count", "sum", "mean", "median", "mode", "std"}:
            raise ValueError("Unsupported line-chart aggregation.")
        data = frame[[column_a, column_b]].dropna().sort_values(column_a, kind="stable")
        if aggregation == "none":
            data = data.head(limit)
            x_values = [json_value(value) for value in data[column_a]]
            y_values = [json_value(value) for value in data[column_b]]
        else:
            if aggregation in {"sum", "mean", "median", "std"} and not _numeric(frame, column_b):
                raise ValueError(f"{aggregation.title()} aggregation requires a numerical column.")
            grouped = data.groupby(column_a, sort=True, dropna=True)[column_b]
            if aggregation == "count":
                aggregated = grouped.count()
            elif aggregation == "sum":
                aggregated = grouped.sum()
            elif aggregation == "mean":
                aggregated = grouped.mean()
            elif aggregation == "median":
                aggregated = grouped.median()
            elif aggregation == "std":
                aggregated = grouped.std()
            else:
                aggregated = grouped.agg(lambda values: _aggregate(values, "mode", column_b))
            x_values = [json_value(value) for value in aggregated.index]
            y_values = [json_value(value) for value in aggregated.tolist()]
        aggregate_text = _title_aggregate(aggregation)
        result.update({"title": f"{aggregate_text} {column_b} by {column_a}" if aggregate_text else f"{column_b} over {column_a}",
                       "columns": {"x": column_a, "y": column_b}, "x": x_values, "y": y_values})
        return result

    if chart_type in {"bar", "pie"}:
        if column_b is None:
            if aggregation not in {"none", "count"}:
                raise ValueError("A value column is required for this aggregation.")
            counts = frame[column_a].dropna().value_counts()
            labels = [json_value(value) for value in counts.index]
            values = [int(value) for value in counts.tolist()]
            aggregate_text = "Count"
        else:
            if logical_column_kind(frame, column_a) not in {"categorical", "text", "boolean"}:
                raise ValueError(f"{chart_type.title()} charts require a categorical category column.")
            if aggregation == "none":
                aggregation = "count"
            if aggregation != "count" and aggregation in {"sum", "mean", "median", "std"} and not _numeric(frame, column_b):
                raise ValueError(f"{aggregation.title()} aggregation requires a numerical column.")
            grouped = frame.dropna(subset=[column_a]).groupby(column_a, sort=False, observed=True)[column_b]
            if aggregation == "count":
                aggregated = grouped.count()
            else:
                aggregated = grouped.agg(lambda values: _aggregate(values, aggregation, column_b))
            labels = [json_value(value) for value in aggregated.index]
            values = [json_value(value) for value in aggregated.tolist()]
            aggregate_text = _title_aggregate(aggregation)
        if chart_type == "pie" and len(labels) > 50:
            raise ValueError("Pie charts require 50 or fewer categories.")
        result.update({
            "title": f"{aggregate_text} {column_b or column_a} by {column_a}" if column_b else f"{column_a} Distribution",
            "columns": {"category": column_a, **({"value": column_b} if column_b else {})},
            "aggregation": aggregation,
            "labels": labels,
            "values": values,
        })
        return result

    if chart_type == "box":
        if column_b is None:
            if not _numeric(frame, column_a):
                raise ValueError("Box plots require a numerical column, optionally grouped by a categorical column.")
            grouping = frame[[column_a]].dropna()[column_a]
            groups = {column_a: grouping.tolist()}
            result["columns"] = {"value": column_a}
            title = f"{column_a} Distribution"
        else:
            if logical_column_kind(frame, column_a) not in {"categorical", "text", "boolean"} or not _numeric(frame, column_b):
                raise ValueError("Grouped box plots require a categorical grouping column and a numerical value column.")
            groups = {
                str(label): values.dropna().tolist()
                for label, values in frame.dropna(subset=[column_a]).groupby(column_a, sort=False, observed=True)[column_b]
            }
            result["columns"] = {"category": column_a, "value": column_b}
            title = f"{column_b} Distribution by {column_a}"
        result.update({"title": title, "groups": [{"label": label, "values": [json_value(value) for value in values]}
                                                      for label, values in groups.items()]})
        return result

    raise ValueError(f"Unsupported chart type: {chart_type}.")


def recommendations(dataset_id: str) -> dict:
    _, frame = load_current(dataset_id)
    numeric = list(frame.select_dtypes(include=["number"]).columns)
    datetime = datetime_column_names(frame)
    categorical = [column for column in frame.columns if column not in numeric + datetime]
    charts = []

    for x_column, y_column in itertools.islice(itertools.product(numeric, numeric), 0, 50):
        if x_column != y_column:
            pair = frame[[x_column, y_column]].dropna()
            correlation = None
            if len(pair) > 1 and pair[x_column].nunique(dropna=True) > 1 and pair[y_column].nunique(dropna=True) > 1:
                correlation = float(pair[[x_column, y_column]].corr().iloc[0, 1])
            charts.append({
                "chart_type": "scatter",
                "x_column": str(x_column),
                "y_column": str(y_column),
                "reason": "Both columns are numeric and contain enough non-null observations to compare their relationship.",
                "purpose": f"Explore the relationship between {x_column} and {y_column}.",
                "statistics": {"correlation": json_value(correlation), "sample_size": int(len(pair))},
            })

    for x_column, y_column in itertools.islice(itertools.product(categorical, numeric), 0, 50):
        charts.append({
            "chart_type": "bar",
            "x_column": str(x_column),
            "y_column": str(y_column),
            "reason": "A categorical field is grouped against a numeric measure.",
            "purpose": f"Compare the distribution of {y_column} across {x_column}.",
            "statistics": {"rows_used": int(frame.dropna(subset=[x_column, y_column]).shape[0])},
        })

    for x_column, y_column in itertools.islice(itertools.product(datetime, numeric), 0, 50):
        charts.append({
            "chart_type": "line",
            "x_column": str(x_column),
            "y_column": str(y_column),
            "reason": "The date column provides an ordered timeline for the numeric measure.",
            "purpose": f"Track how {y_column} changes across {x_column}.",
            "statistics": {"rows_used": int(frame.dropna(subset=[x_column, y_column]).shape[0])},
        })

    for column in numeric:
        charts.append({
            "chart_type": "histogram",
            "x_column": str(column),
            "y_column": None,
            "reason": "The column is numeric and a distribution view helps identify skew and spread.",
            "purpose": f"Understand the distribution of {column}.",
            "statistics": {"sample_size": int(frame[column].dropna().shape[0])},
        })
        charts.append({
            "chart_type": "box",
            "x_column": str(column),
            "y_column": None,
            "reason": "The column is numeric and this highlights spread and outliers.",
            "purpose": f"Inspect the spread and outlier behavior of {column}.",
            "statistics": {"sample_size": int(frame[column].dropna().shape[0])},
        })

    for column in categorical:
        if frame[column].nunique(dropna=True) <= 50:
            charts.append({
                "chart_type": "bar",
                "x_column": str(column),
                "y_column": None,
                "reason": "The field is categorical and its category frequencies are meaningful.",
                "purpose": f"Review the frequency distribution of {column}.",
                "statistics": {"distinct_categories": int(frame[column].nunique(dropna=True))},
            })

    for x_column, y_column in itertools.islice(itertools.product(categorical, numeric), 0, 50):
        if frame[x_column].nunique(dropna=True) <= 50:
            charts.append({
                "chart_type": "pie",
                "x_column": str(x_column),
                "y_column": str(y_column),
                "reason": "A low-cardinality category can be shown as a share of the numeric total without overwhelming the viewer.",
                "purpose": f"See how {y_column} is distributed across the categories in {x_column}.",
                "statistics": {"distinct_categories": int(frame[x_column].nunique(dropna=True))},
            })
            charts.append({
                "chart_type": "box",
                "x_column": str(x_column),
                "y_column": str(y_column),
                "reason": "This compares the numeric distribution within each category.",
                "purpose": f"Inspect the distribution of {y_column} by {x_column}.",
                "statistics": {"distinct_categories": int(frame[x_column].nunique(dropna=True))},
            })

    return {"recommendations": charts[:250]}