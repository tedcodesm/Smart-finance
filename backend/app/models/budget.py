"""Pydantic models for the `budgets` collection."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.constants import BudgetPeriod
from app.core.money import round_money
from app.core.time_utils import period_elapsed_fraction, strip_tz, time_buckets, utc_now
from app.models.base import MongoModel


def _coerce_amount(value: object) -> float:
    try:
        return round_money(Decimal(str(value)))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid amount: {value!r}") from exc


class BudgetCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    limit_amount: float = Field(gt=0)
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    category_id: str | None = Field(
        default=None, description="Omit for an all-categories (overall) budget"
    )
    name: Annotated[str | None, Field(default=None, max_length=120)] = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    alert_threshold_pct: float = Field(default=0.8, gt=0, le=2.0)
    rollover: bool = False
    is_active: bool = True

    @field_validator("limit_amount", mode="before")
    @classmethod
    def _limit(cls, value: object) -> float:
        return _coerce_amount(value)

    @field_validator("alert_threshold_pct", mode="before")
    @classmethod
    def _threshold(cls, value: object) -> float:
        pct = float(value)
        return pct / 100.0 if pct > 1.0 else pct

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _date(cls, value: object) -> object:
        return strip_tz(value) if isinstance(value, datetime) else value

    @model_validator(mode="after")
    def _range(self) -> "BudgetCreate":
        if self.start_date and self.end_date and self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self


class BudgetUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    limit_amount: float | None = Field(default=None, gt=0)
    period: BudgetPeriod | None = None
    category_id: str | None = None
    name: Annotated[str | None, Field(default=None, max_length=120)] = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    alert_threshold_pct: float | None = Field(default=None, gt=0, le=2.0)
    rollover: bool | None = None
    is_active: bool | None = None

    @field_validator("limit_amount", mode="before")
    @classmethod
    def _limit(cls, value: object) -> float | None:
        return None if value is None else _coerce_amount(value)

    @field_validator("alert_threshold_pct", mode="before")
    @classmethod
    def _threshold(cls, value: object) -> float | None:
        if value is None:
            return None
        pct = float(value)
        return pct / 100.0 if pct > 1.0 else pct

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _date(cls, value: object) -> object:
        return strip_tz(value) if isinstance(value, datetime) else value


class BudgetOut(MongoModel):
    name: str | None = None
    category_id: str | None = None
    category_name: str | None = None
    category_color: str | None = None
    limit_amount: float
    period: BudgetPeriod
    start_date: datetime
    end_date: datetime | None = None
    period_key: str | None = None
    alert_threshold_pct: float = 0.8
    rollover: bool = False
    is_active: bool = True
    spent_amount: float = 0.0
    remaining_amount: float = 0.0
    usage_pct: float = 0.0
    days_remaining: int = 0
    period_elapsed_pct: float = 0.0
    is_exceeded: bool = False
    is_threshold_crossed: bool = False
    projected_total: float | None = Field(
        default=None, description="Simple linear projection: spent / elapsed_fraction"
    )
    pace_status: str | None = Field(
        default=None, description="'on_track' | 'warning' | 'over_budget'"
    )
    last_alerted_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class BudgetProgress(BaseModel):
    budget_id: str
    limit_amount: float
    spent_amount: float
    remaining_amount: float
    usage_pct: float
    period: BudgetPeriod
    period_key: str | None = None
    days_remaining: int
    period_elapsed_pct: float
    pace_status: str
    is_exceeded: bool = False
    is_threshold_crossed: bool = False
    projected_total: float | None = None


def initial_buckets(period: BudgetPeriod, start_date: datetime) -> dict[str, str]:
    """Seed the denormalised `period_key` when a budget is created."""
    return {"period_key": time_buckets(start_date)["period_key"], "as_of": time_buckets(utc_now())["period_key"]}


def compute_pace_status(usage_pct: float, elapsed_pct: float, limit_amount: float, spent_amount: float) -> str:
    if limit_amount and spent_amount >= limit_amount:
        return "over_budget"
    if elapsed_pct <= 0:
        return "on_track"
    pace_ratio = usage_pct / (elapsed_pct * 100.0)
    if pace_ratio > 1.15:
        return "warning"
    return "on_track"


def budget_window_fields(period: BudgetPeriod, ref: datetime) -> dict[str, object]:
    return {
        "period_elapsed_pct": round(period_elapsed_fraction(period, ref) * 100.0, 2),
        "as_of": ref,
    }
