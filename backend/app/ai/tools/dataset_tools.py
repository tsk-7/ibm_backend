from app.ai.config import get_config
from app.services.dataset_service import dataset_info, dataset_preview, dataset_profile
from app.services.history_service import list_history


def get_dataset_profile(dataset_id: str) -> dict:
    profile = dataset_profile(dataset_id)
    info = dataset_info(dataset_id)
    return {
        "dataset_id": dataset_id,
        "filename": info["filename"],
        "rows": profile["total_rows"],
        "columns": profile["total_columns"],
        "numerical_columns": profile["numerical_columns"],
        "categorical_columns": profile["categorical_columns"],
        "datetime_columns": profile["datetime_columns"],
        "missing_values": profile["missing_values"],
        "duplicate_rows": profile["duplicate_rows"],
        "data_types": profile["data_types"],
    }


def get_dataset_preview(dataset_id: str, limit: int = 5) -> dict:
    safe_limit = max(1, min(limit, get_config().preview_row_limit))
    return dataset_preview(dataset_id, safe_limit, 0)


def get_missing_values(dataset_id: str) -> dict:
    info = dataset_info(dataset_id)
    profile = dataset_profile(dataset_id)
    total_cells = info["rows"] * info["columns"]
    missing = profile["missing_values"]
    return {
        "total": missing["total"],
        "by_column": missing["by_column"],
        "percentage": round(100 * missing["total"] / total_cells, 2) if total_cells else 0.0,
    }


def get_duplicates(dataset_id: str) -> dict:
    profile = dataset_profile(dataset_id)
    rows = profile["total_rows"]
    duplicates = profile["duplicate_rows"]
    return {
        "duplicate_rows": duplicates,
        "rows": rows,
        "percentage": round(100 * duplicates / rows, 2) if rows else 0.0,
    }


def get_history(dataset_id: str) -> list[dict]:
    return list_history(dataset_id)