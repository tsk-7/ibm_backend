from typing import Any

from pydantic import BaseModel


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    rows: int
    columns: int
    message: str


class DatasetSummary(BaseModel):
    dataset_id: str
    filename: str
    rows: int
    columns: int
    uploaded_at: str


class DatasetInfo(DatasetSummary):
    column_names: list[str]
    data_types: dict[str, str]
    processing_status: str


class DatasetPreview(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    total_rows: int
