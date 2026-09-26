from app.database.database import get_connection
from app.services.dataset_service import get_dataset


def list_history(dataset_id: str) -> list[dict]:
    get_dataset(dataset_id)
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, operation, column_name, rows_before, rows_after, details, timestamp "
            "FROM history WHERE dataset_id = ? ORDER BY id DESC",
            (dataset_id,),
        ).fetchall()
    return [dict(row) for row in rows]