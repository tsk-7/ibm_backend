from fastapi import APIRouter

from app.ai.rag.manager import agent_health
from app.database.database import get_connection
from app.services.health_service import dataset_health


router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    try:
        with get_connection() as connection:
            connection.execute("SELECT 1").fetchone()
        database_status = "ok"
    except Exception:
        database_status = "unavailable"
    ai_status = agent_health()
    return {
        "status": "ok" if database_status == "ok" else "degraded",
        "database": database_status,
        **ai_status,
    }


@router.get("/datasets/{dataset_id}/health")
def dataset_health_report(dataset_id: str):
    return dataset_health(dataset_id)
