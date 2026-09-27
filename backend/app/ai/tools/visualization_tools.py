from app.services.visualization_service import recommendations


def get_visualization_recommendations(dataset_id: str) -> dict:
    return recommendations(dataset_id)