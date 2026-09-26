from typing import Any, Literal

from pydantic import BaseModel, Field


class MissingValuesRequest(BaseModel):
    column: str | None = None
    method: Literal["remove_rows", "mean", "median", "mode", "custom", "ffill", "bfill"]
    value: Any = None


class DuplicateRequest(BaseModel):
    action: Literal["detect", "remove"] = "detect"


class DtypeRequest(BaseModel):
    column: str
    dtype: Literal["integer", "float", "string", "boolean", "datetime"]


class RenameColumnRequest(BaseModel):
    old_name: str
    new_name: str = Field(min_length=1)


class CreateColumnRequest(BaseModel):
    name: str = Field(min_length=1)
    operation: str = Field(min_length=1)
