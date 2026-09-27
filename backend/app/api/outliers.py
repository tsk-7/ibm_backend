from fastapi import APIRouter

from app.services.outliers_service import get_outliers


router = APIRouter(prefix="/datasets", tags=["analysis"])


@router.get("/{dataset_id}/outliers")
def outliers(dataset_id: str):
    return get_outliers(dataset_id)