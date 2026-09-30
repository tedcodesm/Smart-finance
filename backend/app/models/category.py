"""Pydantic models for the `categories` collection."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.constants import TransactionType
from app.models.base import MongoModel

HEX_COLOR = Annotated[str, Field(pattern=r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")]


class CategoryBase(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: Annotated[str, Field(min_length=1, max_length=60)]
    type: TransactionType
    icon: Annotated[str | None, Field(default=None, max_length=40)] = None
    color: HEX_COLOR | None = None
    monthly_typical_amount: float | None = Field(default=None, ge=0)

    @field_validator("name")
    @classmethod
    def _normalise(cls, value: str) -> str:
        return " ".join(value.split()).title()


class CategoryCreate(CategoryBase):
    pass


class CategoryUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: Annotated[str | None, Field(default=None, min_length=1, max_length=60)] = None
    icon: Annotated[str | None, Field(default=None, max_length=40)] = None
    color: HEX_COLOR | None = None
    monthly_typical_amount: float | None = Field(default=None, ge=0)
    is_archived: bool | None = None

    @field_validator("name")
    @classmethod
    def _normalise(cls, value: str | None) -> str | None:
        return " ".join(value.split()).title() if value else value


class CategoryOut(MongoModel):
    name: str
    type: TransactionType
    icon: str | None = None
    color: str | None = None
    is_system: bool = False
    is_archived: bool = False
    monthly_typical_amount: float | None = None
    transaction_count: int | None = Field(
        default=None, description="Populated by list endpoints that aggregate usage"
    )
    created_at: datetime | None = None
    updated_at: datetime | None = None
