from fastapi import APIRouter, Body

from app.schemas.preprocessing import (
    CreateColumnRequest,
    DuplicateRequest,
    DtypeRequest,
    MissingValuesRequest,
    RenameColumnRequest,
)
from app.services.preprocessing_service import (
    convert_dtype,
    create_column,
    delete_column,
    preview_iqr_analysis,
    preview_missing_values,
    preview_normalization,
    preview_standardization,
    process_duplicates,
    process_missing_values,
    preprocessing_summary,
    preview_duplicates,
    rename_column,
)


router = APIRouter(prefix="/datasets", tags=["preprocessing"])


@router.get("/{dataset_id}/preprocessing/summary")
def preprocessing_summary_route(dataset_id: str):
    return preprocessing_summary(dataset_id)


@router.post("/{dataset_id}/preprocess/missing-values/preview")
def missing_values_preview(dataset_id: str, request: MissingValuesRequest):
    return preview_missing_values(dataset_id, request.column, request.method, request.value)


@router.post("/{dataset_id}/preprocess/missing-values")
def missing_values(dataset_id: str, request: MissingValuesRequest):
    return process_missing_values(dataset_id, request.column, request.method, request.value)


@router.post("/{dataset_id}/preprocess/duplicates")
def duplicates(dataset_id: str, request: DuplicateRequest = Body(default=DuplicateRequest())):
    return process_duplicates(dataset_id, request.action)


@router.post("/{dataset_id}/preprocess/duplicates/preview")
def duplicates_preview(dataset_id: str):
    return preview_duplicates(dataset_id)


@router.post("/{dataset_id}/preprocess/standardization/preview")
def standardization_preview(dataset_id: str, request: dict):
    return preview_standardization(dataset_id, request["column"])


@router.post("/{dataset_id}/preprocess/normalization/preview")
def normalization_preview(dataset_id: str, request: dict):
    return preview_normalization(dataset_id, request["column"])


@router.post("/{dataset_id}/preprocess/iqr/preview")
def iqr_preview(dataset_id: str, request: dict):
    return preview_iqr_analysis(dataset_id, request["column"])


@router.post("/{dataset_id}/preprocess/dtype")
def dtype(dataset_id: str, request: DtypeRequest):
    return convert_dtype(dataset_id, request.column, request.dtype)


@router.post("/{dataset_id}/columns/rename")
def rename(dataset_id: str, request: RenameColumnRequest):
    return rename_column(dataset_id, request.old_name, request.new_name)


@router.delete("/{dataset_id}/columns/{column_name}")
def delete(dataset_id: str, column_name: str):
    return delete_column(dataset_id, column_name)


@router.post("/{dataset_id}/columns/create")
def create(dataset_id: str, request: CreateColumnRequest):
    return create_column(dataset_id, request.name, request.operation)
