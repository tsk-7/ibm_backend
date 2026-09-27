from typing import Any, Literal

from pydantic import BaseModel, Field


class VisualizationPreviewRequest(BaseModel):
    chart_type: Literal["bar", "pie", "scatter", "line", "box", "histogram"]
    column_a: str = Field(min_length=1)
    column_b: str | None = None
    aggregation: Literal["none", "count", "sum", "mean", "average", "median", "mode", "std"] = "none"
    bins: int = Field(default=10, ge=1, le=100)
    limit: int = Field(default=1000, ge=1, le=10000)


class VisualizationPreviewResponse(BaseModel):
    chart_type: str
    dataset_id: str
    title: str
    columns: dict[str, str]
    aggregation: str | None = None
    labels: list[Any] | None = None
    values: list[Any] | None = None
    x: list[Any] | None = None
    y: list[Any] | None = None
    groups: list[dict[str, Any]] | None = None
    bins: list[float] | None = None
    counts: list[int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)