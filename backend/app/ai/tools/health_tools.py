from app.services.health_service import dataset_health


def get_data_health(dataset_id: str) -> dict:
    return dataset_health(dataset_id)