"""Pydantic models for the `savings_goals` collection."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.constants import GoalStatus
from app.core.money import round_money, safe_div
from app.core.time_utils import strip_tz, utc_now
from app.models.base import MongoModel


def _coerce_amount(value: object) -> float:
    try:
        return round_money(Decimal(str(value)))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid amount: {value!r}") from exc


class GoalCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: Annotated[str, Field(min_length=1, max_length=120)]
    target_amount: float = Field(gt=0)
    target_date: datetime | None = None
    current_amount: float = Field(default=0.0, ge=0)
    description: Annotated[str | None, Field(default=None, max_length=500)] = None
    color: Annotated[str | None, Field(default=None, pattern=r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")] = None
    linked_category_id: str | None = None

    @field_validator("target_amount", "current_amount", mode="before")
    @classmethod
    def _amount(cls, value: object) -> float:
        return _coerce_amount(value)

    @field_validator("target_date", mode="before")
    @classmethod
    def _date(cls, value: object) -> object:
        return strip_tz(value) if isinstance(value, datetime) else value

    @model_validator(mode="after")
    def _check(self) -> "GoalCreate":
        if self.current_amount > self.target_amount:
            raise ValueError("current_amount cannot exceed target_amount")
        return self


class GoalUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: Annotated[str | None, Field(default=None, min_length=1, max_length=120)] = None
    target_amount: float | None = Field(default=None, gt=0)
    target_date: datetime | None = None
    current_amount: float | None = Field(default=None, ge=0)
    description: Annotated[str | None, Field(default=None, max_length=500)] = None
    color: Annotated[str | None, Field(default=None, pattern=r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")] = None
    status: GoalStatus | None = None
    linked_category_id: str | None = None

    @field_validator("target_amount", "current_amount", mode="before")
    @classmethod
    def _amount(cls, value: object) -> float | None:
        return None if value is None else _coerce_amount(value)

    @field_validator("target_date", mode="before")
    @classmethod
    def _date(cls, value: object) -> object:
        return strip_tz(value) if isinstance(value, datetime) else value


class GoalContribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(gt=0, description="Positive to add, negative to withdraw")
    note: Annotated[str | None, Field(default=None, max_length=280)] = None

    @field_validator("amount", mode="before")
    @classmethod
    def _amount(cls, value: object) -> float:
        return _coerce_amount(value)


class GoalOut(MongoModel):
    name: str
    description: str | None = None
    target_amount: float
    current_amount: float
    remaining_amount: float = 0.0
    progress_pct: float = 0.0
    target_date: datetime | None = None
    status: GoalStatus = GoalStatus.ACTIVE
    color: str | None = None
    linked_category_id: str | None = None
    days_remaining: int | None = None
    required_monthly_contribution: float | None = None
    on_track: bool | None = None
    required_daily_contribution: float | None = None
    achieved_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class GoalProgressEvent(BaseModel):
    goal_id: str
    amount: float
    new_balance: float
    progress_pct: float
    goal_achieved: bool
    note: str | None = None
    created_at: datetime = Field(default_factory=lambda: utc_now())


class GoalSummaryOut(BaseModel):
    active_goals: int = 0
    achieved_goals: int = 0
    total_goals: int = 0
    total_target: float = 0.0
    total_saved: float = 0.0
    total_remaining: float = 0.0
    overall_progress_pct: float = 0.0


def goal_metrics(goal: dict) -> dict:
    """Denormalised progress fields recomputed on every write."""
    target = float(goal.get("target_amount") or 0.0)
    current = float(goal.get("current_amount") or 0.0)
    remaining = max(0.0, round_money(target - current))
    progress = round(safe_div(current, target) * 100.0, 2) if target else 0.0

    target_date = goal.get("target_date")
    if not target_date:
        return {
            "progress_pct": progress,
            "remaining_amount": round_money(remaining),
            "days_remaining": None,
            "required_monthly_contribution": None,
            "required_daily_contribution": None,
            "on_track": None,
        }

    today = utc_now().date()
    days_left = max(0, (strip_tz(target_date).date() - today).days)

    if days_left <= 0:
        # Deadline has passed: either the goal is met or it is now the shortfall.
        return {
            "progress_pct": progress,
            "remaining_amount": round_money(remaining),
            "days_remaining": 0,
            "required_monthly_contribution": 0.0 if current >= target else round_money(remaining),
            "required_daily_contribution": 0.0,
            "on_track": current >= target,
        }

    months_left = max(days_left / 30.4375, 1e-6)
    return {
        "progress_pct": progress,
        "remaining_amount": round_money(remaining),
        "days_remaining": days_left,
        "required_monthly_contribution": round_money(safe_div(remaining, months_left)),
        "required_daily_contribution": round_money(safe_div(remaining, days_left)),
        "on_track": progress >= _expected_progress_pct(goal, days_left) - 5.0,
    }


def _expected_progress_pct(goal: dict, days_left: int) -> float:
    """Linear expectation: elapsed share of the goal's creation -> deadline span.

    Without a target date at creation time we fall back to today's progress
    requirement, so a goal created today is considered trivially on-track.
    """
    created = strip_tz(goal.get("created_at") or utc_now()).date()
    target_date = goal.get("target_date")
    if not target_date:
        return 0.0
    total_span = max(1, (strip_tz(target_date).date() - created).days)
    elapsed = max(0, (utc_now().date() - created).days)
    return round(min(100.0, safe_div(elapsed, total_span) * 100.0), 2)
