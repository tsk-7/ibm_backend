import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from app.utils.dataframe_utils import normalize_missing_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def data_directory(name: str) -> Path:
    configured = os.getenv("DATAINSIGHT_DATA_DIR")
    root = Path(configured) if configured else PROJECT_ROOT / "data"
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(filename: str) -> str:
    return Path(filename.replace("\\", "/")).name


def read_dataframe(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    extension = source.suffix.lower()
    if extension == ".csv":
        frame = pd.read_csv(source, keep_default_na=True)
    elif extension in {".xlsx", ".xls"}:
        frame = pd.read_excel(source)
    else:
        raise ValueError("Unsupported dataset format")
    if frame.empty and len(frame.columns) == 0:
        raise ValueError("Dataset is empty or has no columns")
    if frame.columns.duplicated().any():
        duplicates = [str(column) for column in frame.columns[frame.columns.duplicated()]]
        raise ValueError(f"Dataset contains duplicate column names: {', '.join(duplicates)}")
    return normalize_missing_values(frame)


def save_dataframe_csv(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_upload(upload: BinaryIO, filename: str) -> tuple[str, str, str]:
    clean_name = safe_filename(filename)
    extension = Path(clean_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file format. Upload a CSV, XLSX, or XLS file.")
    dataset_id = f"DS-{uuid.uuid4().hex[:12].upper()}"
    destination = data_directory("uploads") / f"{dataset_id}{extension}"
    with destination.open("xb") as target:
        shutil.copyfileobj(upload, target)
    return dataset_id, str(destination), clean_name
