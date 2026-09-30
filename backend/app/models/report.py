"""Pydantic models for the `reports` collection and report request payloads."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.constants import BudgetPeriod, ReportType
from app.core.money import round_money, safe_div
from app.core.time_utils import period_window, resolve_range, strip_tz
from app.models.base import MongoModel

ReportPeriod = Literal["daily", "weekly", "monthly", "yearly", "custom"]

_ONE_MICROSECOND = timedelta(microseconds=1)


def _anchor_date(period: str, period_key: str) -> datetime:
    """Map a bucket key back to a representative date inside that bucket."""
    if period == "yearly":
        return datetime(int(period_key), 6, 1)
    if period == "monthly":
        return datetime.strptime(period_key, "%Y-%m")
    if period == "weekly":
        year_s, week_s = period_key.split("-W")
        return datetime.combine(
            _date_from_iso_week(int(year_s), int(week_s)),
            datetime.min.time(),
        )
    if period == "daily":
        return datetime.strptime(period_key, "%Y-%m-%d")
    raise ValueError(f"Unsupported period {period!r} with period_key {period_key!r}")


def _date_from_iso_week(year: int, week: int):
    from datetime import date

    return date.fromisocalendar(year, week, 1)


class ReportRequest(BaseModel):
    """Query payload for generating an on-demand report."""

    model_config = ConfigDict(extra="forbid")

    report_type: ReportType
    period: ReportPeriod = "monthly"
    start_date: datetime | None = None
    end_date: datetime | None = None
    period_key: str | None = Field(
        default=None,
        description="Anchor for relative periods, e.g. '2026-09' or '2026-W38'",
    )
    category_ids: list[str] | None = None
    persist: bool = Field(default=False, description="Write the result into the `reports` collection")
    limit_top: int = Field(default=20, ge=1, le=200)

    @model_validator(mode="after")
    def _check(self) -> "ReportRequest":
        if self.period == "custom":
            if not self.start_date or not self.end_date:
                raise ValueError("period='custom' requires both start_date and end_date")
            if strip_tz(self.end_date) < strip_tz(self.start_date):
                raise ValueError("end_date must not precede start_date")
        if self.period_key and self.period == "custom":
            raise ValueError("period_key is not valid when period='custom'")
        return self

    def resolved_range(self) -> tuple[datetime, datetime, str]:
        """Concrete inclusive [start, end] datetimes plus the period bucket key."""
        if self.period == "custom":
            assert self.start_date is not None and self.end_date is not None
            start, end = strip_tz(self.start_date), strip_tz(self.end_date)
            span_days = (end.date() - start.date()).days
            key = "custom" if span_days > 45 else start.date().isoformat()
            return start, end, key

        if self.period_key:
            try:
                anchor = _anchor_date(self.period, self.period_key)
            except (ValueError, KeyError):
                anchor = None
            if anchor is not None:
                start, end, key = period_window(BudgetPeriod(self.period), anchor)
                return start, end - _ONE_MICROSECOND, key

        return resolve_range(self.period, None)  # type: ignore[arg-type]


class CategorySlice(BaseModel):
    category_id: str | None = None
    category_name: str
    category_color: str | None = None
    total: float
    count: int
    share_pct: float
    average: float
    previous_total: float | None = None
    change_pct: float | None = None
    trend: str | None = Field(default=None, description="up | down | flat")


class TimeBucket(BaseModel):
    period_key: str
    income: float = 0.0
    expense: float = 0.0
    net: float = 0.0
    transaction_count: int = 0
    cumulative_net: float = 0.0


class BudgetSlice(BaseModel):
    budget_id: str
    category_name: str | None = None
    limit_amount: float
    spent_amount: float
    usage_pct: float
    remaining_amount: float
    pace_status: str


class ReportTotals(BaseModel):
    income: float = 0.0
    expense: float = 0.0
    net: float = 0.0
    transaction_count: int = 0
    savings_rate_pct: float = 0.0
    average_transaction: float = 0.0
    largest_expense: float | None = None
    largest_expense_category: str | None = None


class ReportData(BaseModel):
    model_config = ConfigDict(extra="allow")

    report_type: ReportType
    period: ReportPeriod
    period_key: str | None = None
    start_date: datetime
    end_date: datetime
    totals: ReportTotals = Field(default_factory=ReportTotals)
    by_category: list[CategorySlice] = Field(default_factory=list)
    timeline: list[TimeBucket] = Field(default_factory=list)
    budgets: list[BudgetSlice] = Field(default_factory=list)
    goals: list[dict[str, Any]] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)


class ReportOut(MongoModel):
    report_type: ReportType
    period: ReportPeriod
    period_start: datetime
    period_end: datetime
    period_key: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    format: str = "json"
    generated_at: datetime
    expires_at: datetime | None = None


class ExportFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["csv", "pdf", "json"] = "csv"
    include_summary: bool = True
    max_rows: int = Field(default=5000, ge=1, le=100_000)


class ReportExportResponse(BaseModel):
    report_id: str | None = None
    report_type: ReportType
    period: ReportPeriod
    filename: str
    media_type: str
    download_url: str
    generated_at: datetime
    size_bytes: Annotated[int, Field(ge=0)]


def compute_savings_rate(income: float, expense: float) -> float:
    if income <= 0:
        return 0.0
    return round(safe_div(income - expense, income) * 100.0, 2)


def trend_label(change_pct: float | None, *, deadband: float = 5.0) -> str | None:
    """Coarse up/down/flat label for UI arrows."""
    if change_pct is None:
        return None
    if change_pct > deadband:
        return "up"
    if change_pct < -deadband:
        return "down"
    return "flat"


def money(value: float) -> float:
    return round_money(value)


__all__ = [
    "BudgetSlice",
    "CategorySlice",
    "ExportFormat",
    "ReportData",
    "ReportExportResponse",
    "ReportOut",
    "ReportRequest",
    "ReportTotals",
    "TimeBucket",
    "compute_savings_rate",
    "money",
    "trend_label",
]
