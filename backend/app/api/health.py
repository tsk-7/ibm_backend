from fastapi import APIRouter

from app.database.database import get_connection
from app.services.health_service import dataset_health


router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    try:
        with get_connection() as connection:
            connection.execute("SELECT 1").fetchone()
        return {"status": "ok", "database": "ok"}
    except Exception:
        return {"status": "degraded", "database": "unavailable"}


@router.get("/datasets/{dataset_id}/health")
def dataset_health_report(dataset_id: str):
    return dataset_health(dataset_id)
