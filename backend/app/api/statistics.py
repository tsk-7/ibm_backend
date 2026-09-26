from fastapi import APIRouter

from app.services.statistics_service import calculate_statistics


router = APIRouter(prefix="/datasets", tags=["statistics"])


@router.get("/{dataset_id}/statistics")
def statistics(dataset_id: str):
    return calculate_statistics(dataset_id)
