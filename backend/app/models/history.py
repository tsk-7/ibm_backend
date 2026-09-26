from dataclasses import dataclass


@dataclass
class HistoryEntry:
    id: int
    dataset_id: str
    operation: str
    column_name: str | None
    rows_before: int
    rows_after: int
    details: str
    timestamp: str
