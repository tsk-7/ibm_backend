from fastapi import APIRouter

from app.services.history_service import list_history


router = APIRouter(prefix="/datasets", tags=["history"])


@router.get("/{dataset_id}/history")
def processing_history(dataset_id: str):
    return list_history(dataset_id)
