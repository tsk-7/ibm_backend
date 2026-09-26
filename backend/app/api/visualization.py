from fastapi import APIRouter

from app.services.visualization_service import recommendations


router = APIRouter(prefix="/datasets", tags=["visualization"])


@router.get("/{dataset_id}/visualization/recommendations")
def chart_recommendations(dataset_id: str):
    return recommendations(dataset_id)
