from app.services.statistics_service import calculate_statistics


def get_statistics(dataset_id: str, column: str | None = None) -> dict:
    result = calculate_statistics(dataset_id)
    if column is None:
        return result
    for group in ("numerical", "categorical"):
        if column in result[group]:
            return {group: {column: result[group][column]}}
    raise ValueError(f"Column does not exist: {column}")