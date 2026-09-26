from io import BytesIO
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.services.dataset_service import load_current


router = APIRouter(prefix="/datasets", tags=["download"])


@router.get("/{dataset_id}/download")
def download_dataset(dataset_id: str, format: Literal["csv", "xlsx"] = Query(default="csv")):
    dataset, frame = load_current(dataset_id)
    output = BytesIO()
    if format == "csv":
        output.write(frame.to_csv(index=False).encode("utf-8-sig"))
        media_type = "text/csv; charset=utf-8"
        extension = ".csv"
    else:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name="Data")
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        extension = ".xlsx"
    output.seek(0)
    filename = f"{Path(dataset.filename).stem}_processed{extension}"
    return StreamingResponse(
        output,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
