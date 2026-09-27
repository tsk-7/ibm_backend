from fastapi import APIRouter

from app.schemas.visualization import VisualizationPreviewRequest, VisualizationPreviewResponse
from app.services.visualization_service import prepare_visualization, recommendations


router = APIRouter(prefix="/datasets", tags=["visualization"])


@router.get("/{dataset_id}/visualization/recommendations")
def chart_recommendations(dataset_id: str):
    return recommendations(dataset_id)


@router.post("/{dataset_id}/visualizations/preview", response_model=VisualizationPreviewResponse)
def visualization_preview(dataset_id: str, request: VisualizationPreviewRequest):
    return prepare_visualization(
        dataset_id,
        request.chart_type,
        request.column_a,
        request.column_b,
        request.aggregation,
        request.bins,
        request.limit,
    )
