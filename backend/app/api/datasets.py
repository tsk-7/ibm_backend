from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.dataset import DatasetInfo, DatasetPreview, DatasetSummary, DatasetUploadResponse
from app.services.dataset_service import (
    dataset_info,
    dataset_preview,
    dataset_profile,
    list_datasets,
    upload_dataset,
)


router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetUploadResponse, status_code=201)
def upload_dataset_route(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required")
    try:
        dataset, frame = upload_dataset(file.file, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read uploaded dataset: {exc}") from exc
    return {
        "dataset_id": dataset.dataset_id,
        "filename": dataset.filename,
        "rows": len(frame),
        "columns": len(frame.columns),
        "message": "Dataset uploaded successfully",
    }


@router.get("", response_model=list[DatasetSummary])
def get_datasets():
    return list_datasets()


@router.get("/{dataset_id}", response_model=DatasetInfo)
def get_dataset_info(dataset_id: str):
    return dataset_info(dataset_id)


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
def get_dataset_preview(
    dataset_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    return dataset_preview(dataset_id, limit, offset)


@router.get("/{dataset_id}/profile")
def get_dataset_profile(dataset_id: str):
    return dataset_profile(dataset_id)
