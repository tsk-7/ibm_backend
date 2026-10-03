import json
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import pandas as pd

from app.database.database import get_connection
from app.models.dataset import Dataset
from app.utils.dataframe_utils import datetime_column_names
from app.utils.file_utils import data_directory, read_dataframe, save_dataframe_csv, save_upload, utc_now


def _dataset(row: Mapping[str, Any]) -> Dataset:
    return Dataset(**dict(row))


def upload_dataset(upload: BinaryIO, filename: str) -> tuple[Dataset, pd.DataFrame]:
    dataset_id, original_path, clean_name = save_upload(upload, filename)
    try:
        frame = read_dataframe(original_path)
        processed_path = data_directory("processed") / f"{dataset_id}.csv"
        save_dataframe_csv(frame, processed_path)
        now = utc_now()
        with get_connection() as connection:
            connection.execute(
                "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (dataset_id, clean_name, original_path,
                 str(processed_path), len(frame), len(frame.columns), now, now),
            )
            row = connection.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
        return _dataset(row), frame
    except Exception:
        Path(original_path).unlink(missing_ok=True)
        raise


def get_dataset(dataset_id: str) -> Dataset:
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
    if row is None:
        raise LookupError("Dataset not found")
    return _dataset(row)


def list_datasets() -> list[Dataset]:
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM datasets ORDER BY uploaded_at DESC").fetchall()
    return [_dataset(row) for row in rows]


def load_current(dataset_id: str) -> tuple[Dataset, pd.DataFrame]:
    dataset = get_dataset(dataset_id)
    try:
        return dataset, read_dataframe(dataset.processed_path)
    except FileNotFoundError as exc:
        raise LookupError("Dataset file not found") from exc


def dataset_info(dataset_id: str) -> dict:
    dataset, frame = load_current(dataset_id)
    return {
        "dataset_id": dataset.dataset_id,
        "filename": dataset.filename,
        "rows": dataset.rows,
        "columns": dataset.columns,
        "uploaded_at": dataset.uploaded_at,
        "column_names": [str(column) for column in frame.columns],
        "data_types": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "processing_status": "processed" if dataset.updated_at != dataset.uploaded_at else "uploaded",
    }


def dataset_profile(dataset_id: str) -> dict:
    from app.services.dataset_profiler import profile_dataset

    dataset, frame = load_current(dataset_id)
    profile = profile_dataset(dataset_id)
    missing = {str(column): int(count) for column, count in frame.isna().sum().items()}
    unique = {str(column): int(frame[column].nunique(dropna=True)) for column in frame.columns}
    payload = {
        "dataset_id": dataset.dataset_id,
        "total_rows": int(len(frame)),
        "total_columns": int(len(frame.columns)),
        "numerical_columns": [str(column) for column in profile["numerical_columns"]],
        "categorical_columns": [str(column) for column in profile["categorical_columns"]],
        "datetime_columns": [str(column) for column in profile["datetime_columns"]],
        "missing_values": {"by_column": missing, "total": int(frame.isna().sum().sum())},
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_percentage": profile["general"]["duplicate_percentage"],
        "unique_values": unique,
        "memory_usage_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
        "data_types": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "general": profile["general"],
        "column_profiles": profile["columns"],
        "summary": profile["summary"],
    }
    return payload


def dataset_preview(dataset_id: str, limit: int, offset: int) -> dict:
    _, frame = load_current(dataset_id)
    sample = frame.iloc[offset:offset + limit]
    from app.utils.dataframe_utils import dataframe_records

    return {"columns": [str(column) for column in frame.columns], "rows": dataframe_records(sample), "total_rows": len(frame)}
