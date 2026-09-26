import itertools

import pandas as pd

from app.services.dataset_service import load_current
from app.utils.dataframe_utils import datetime_column_names


def recommendations(dataset_id: str) -> dict:
    _, frame = load_current(dataset_id)
    numeric = list(frame.select_dtypes(include=["number"]).columns)
    datetime = datetime_column_names(frame)
    categorical = [column for column in frame.columns if column not in numeric + datetime]
    charts = []
    for x_column, y_column in itertools.islice(itertools.product(numeric, numeric), 0, 100):
        if x_column != y_column:
            charts.append({"chart_type": "scatter", "x_column": str(x_column), "y_column": str(y_column),
                           "reason": "Relationship between two numerical columns"})
    for x_column, y_column in itertools.islice(itertools.product(categorical, numeric), 0, 100):
        charts.append({"chart_type": "bar", "x_column": str(x_column), "y_column": str(y_column),
                       "reason": "Categorical column compared with numerical values"})
    for x_column, y_column in itertools.islice(itertools.product(datetime, numeric), 0, 100):
        charts.append({"chart_type": "line", "x_column": str(x_column), "y_column": str(y_column),
                       "reason": "Numerical values over time"})
    for column in numeric:
        charts.append({"chart_type": "histogram", "x_column": str(column), "y_column": None,
                       "reason": "Distribution of a numerical column"})
    for column in categorical:
        charts.append({"chart_type": "bar", "x_column": str(column), "y_column": None,
                       "reason": "Frequency of categories"})
    return {"recommendations": charts[:250]}