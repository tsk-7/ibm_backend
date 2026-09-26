from dataclasses import dataclass


@dataclass
class Dataset:
    dataset_id: str
    filename: str
    original_path: str
    processed_path: str
    rows: int
    columns: int
    uploaded_at: str
    updated_at: str
