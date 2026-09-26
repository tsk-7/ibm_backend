from typing import Any

from pydantic import BaseModel


class StatisticsResponse(BaseModel):
    numerical: dict[str, dict[str, Any]]
    categorical: dict[str, dict[str, Any]]
