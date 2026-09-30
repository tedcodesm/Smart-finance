"""Pydantic models for the `transactions` collection."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.constants import TransactionType
from app.core.money import round_money
from app.core.time_utils import strip_tz, utc_now
from app.models.base import MongoModel

Amount = Annotated[float, Field(gt=0, description="Always a positive magnitude; direction comes from `type`")]


def _coerce_amount(value: object) -> float:
    try:
        return round_money(Decimal(str(value)))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid amount: {value!r}") from exc


class TransactionCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    amount: Amount
    type: TransactionType
    category_id: str
    date: datetime | None = Field(default=None, description="Defaults to now (UTC)")
    description: Annotated[str | None, Field(default=None, max_length=280)] = None
    notes: Annotated[str | None, Field(default=None, max_length=2000)] = None
    is_recurring: bool = False

    @field_validator("amount", mode="before")
    @classmethod
    def _amount(cls, value: object) -> float:
        return _coerce_amount(value)

    @field_validator("date")
    @classmethod
    def _date(cls, value: datetime | None) -> datetime:
        return strip_tz(value) if value else strip_tz(utc_now())


class TransactionUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    amount: float | None = Field(default=None, gt=0)
    type: TransactionType | None = None
    category_id: str | None = None
    date: datetime | None = None
    description: Annotated[str | None, Field(default=None, max_length=280)] = None
    notes: Annotated[str | None, Field(default=None, max_length=2000)] = None
    is_recurring: bool | None = None

    @field_validator("amount", mode="before")
    @classmethod
    def _amount(cls, value: object) -> float | None:
        return None if value is None else _coerce_amount(value)

    @field_validator("date")
    @classmethod
    def _date(cls, value: datetime | None) -> datetime | None:
        return strip_tz(value) if value else None


class TransactionOut(MongoModel):
    amount: float
    type: TransactionType
    category_id: str | None = None
    category_name: str | None = Field(default=None, description="Denormalised for list rendering")
    category_color: str | None = None
    category_icon: str | None = None
    date: datetime
    description: str | None = None
    notes: str | None = None
    is_recurring: bool = False
    year: int | None = None
    month: int | None = None
    day: int | None = None
    day_of_week: int | None = Field(default=None, description="ISO-8601: 0=Monday")
    day_of_month: int | None = None
    week_of_year: int | None = None
    period_key: str | None = None
    week_key: str | None = None
    signed_amount: float | None = Field(
        default=None, description="+amount for income, -amount for expense"
    )
    created_at: datetime | None = None
    updated_at: datetime | None = None


class TransactionBulkCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: Annotated[list[TransactionCreate], Field(min_length=1, max_length=1000)]


class TransactionSummary(BaseModel):
    income: float = 0.0
    expense: float = 0.0
    net: float = 0.0
    transaction_count: int = 0
    average_expense: float = 0.0
    largest_expense: float | None = None
