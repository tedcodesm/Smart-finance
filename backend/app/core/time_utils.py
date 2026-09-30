"""Date / period helpers.

Time buckets are denormalised onto every transaction document at write time
(`year`, `month`, `day`, `day_of_week`, `day_of_month`, `week_of_year`,
`period_key`) so that reporting can group by an indexed scalar field instead of
running date arithmetic in an aggregation pipeline.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal

from app.core.constants import BudgetPeriod

PeriodLike = Literal["daily", "weekly", "monthly", "yearly"]


def utc_now() -> datetime:
    """Timezone-aware current UTC time (BSON stores naive UTC ms)."""
    return datetime.now(timezone.utc)


def as_utc(value: datetime) -> datetime:
    """Normalise a datetime to timezone-aware UTC.

    MongoDB round-trips datetimes as naive; we always treat naive input as UTC.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def strip_tz(value: datetime) -> datetime:
    """Drop the tzinfo after normalising to UTC - matches BSON round-trip."""
    return as_utc(value).replace(tzinfo=None)


def start_of_day(value: datetime | date) -> datetime:
    d = value.date() if isinstance(value, datetime) else value
    return datetime.combine(d, time.min)


def end_of_day(value: datetime | date) -> datetime:
    d = value.date() if isinstance(value, datetime) else value
    return datetime.combine(d, time.max)


def month_key(value: datetime | date) -> str:
    d = value.date() if isinstance(value, datetime) else value
    return f"{d.year:04d}-{d.month:02d}"


def year_key(value: datetime | date) -> str:
    d = value.date() if isinstance(value, datetime) else value
    return f"{d.year:04d}"


def week_key(value: datetime | date) -> str:
    """ISO-8601 week bucket, e.g. `2026-W38`."""
    d = value.date() if isinstance(value, datetime) else value
    iso = d.isocalendar()
    return f"{iso.year:04d}-W{iso.week:02d}"


def day_key(value: datetime | date) -> str:
    d = value.date() if isinstance(value, datetime) else value
    return d.isoformat()


def period_key_for(period: str, value: datetime | date) -> str:
    """Bucket key for a given period granularity."""
    period = (period or "monthly").lower()
    if period == "daily":
        return day_key(value)
    if period == "weekly":
        return week_key(value)
    if period == "yearly":
        return year_key(value)
    return month_key(value)


def time_buckets(value: datetime | date) -> dict[str, int | str]:
    """Denormalised time fields stored on a transaction document.

    `day_of_week` is ISO-8601 (0 = Monday .. 6 = Sunday).
    """
    d = value.date() if isinstance(value, datetime) else value
    iso = d.isocalendar()
    return {
        "year": d.year,
        "month": d.month,
        "day": d.day,
        "day_of_week": d.weekday(),
        "day_of_month": d.day,
        "week_of_year": iso.week,
        "iso_year": iso.year,
        "period_key": month_key(d),
        "week_key": week_key(d),
    }


def period_window(period: BudgetPeriod | str, ref: datetime | date) -> tuple[datetime, datetime, str]:
    """Resolve the [start, end) window containing `ref` for a budget period.

    Returns `(start, end, period_key)` where `end` is exclusive and
    `period_key` is the bucket identifier stored on the budget.
    """
    d = ref.date() if isinstance(ref, datetime) else ref
    base = datetime.combine(d, time.min)

    if period == BudgetPeriod.WEEKLY:
        start_date = d - timedelta(days=d.weekday())  # Monday
        end_date = start_date + timedelta(days=7)
        return datetime.combine(start_date, time.min), datetime.combine(end_date, time.min), week_key(d)

    if period == BudgetPeriod.YEARLY:
        start_date = date(d.year, 1, 1)
        end_date = date(d.year + 1, 1, 1)
        return datetime.combine(start_date, time.min), datetime.combine(end_date, time.min), year_key(d)

    # default: monthly
    start_date = date(d.year, d.month, 1)
    last_day = calendar.monthrange(d.year, d.month)[1]
    end_date = date(d.year, d.month, last_day) + timedelta(days=1)
    return (
        datetime.combine(start_date, time.min),
        datetime.combine(end_date, time.min),
        month_key(d),
    )


def next_period(period: BudgetPeriod | str, period_key: str) -> str:
    """Advance a period bucket key by one step (used for forecasting)."""
    period = BudgetPeriod(period)
    if period is BudgetPeriod.YEARLY:
        return f"{int(period_key) + 1:04d}"
    if period is BudgetPeriod.WEEKLY:
        # `YYYY-Www` -> parse via the Monday of that ISO week, then add 7 days.
        year_s, week_s = period_key.split("-W")
        monday = date.fromisocalendar(int(year_s), int(week_s), 1)
        return week_key(monday + timedelta(days=7))
    year_s, month_s = period_key.split("-")
    year, month = int(year_s), int(month_s)
    return month_key(date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1))


def days_in_period(period: BudgetPeriod | str, period_key: str) -> int:
    """Number of days in the bucket identified by `period_key`."""
    period = BudgetPeriod(period)
    if period is BudgetPeriod.YEARLY:
        return 366 if int(period_key) % 4 == 0 else 365
    if period is BudgetPeriod.WEEKLY:
        return 7
    year_s, month_s = period_key.split("-")
    return calendar.monthrange(int(year_s), int(month_s))[1]


def period_elapsed_fraction(period: BudgetPeriod | str, ref: datetime | date) -> float:
    """Fraction of the current budget period already elapsed, in [0, 1]."""
    start, end, _ = period_window(period, ref)
    d = ref.date() if isinstance(ref, datetime) else ref
    total = (end - start).total_seconds()
    if total <= 0:
        return 1.0
    done = (datetime.combine(d, time.min) - start).total_seconds()
    return max(0.0, min(1.0, done / total))


def days_remaining(period: BudgetPeriod | str, ref: datetime | date) -> int:
    """Whole days left in the current budget period (>= 0)."""
    _, end, _ = period_window(period, ref)
    d = ref.date() if isinstance(ref, datetime) else ref
    return max(0, (end.date() - d).days)


def resolve_range(
    period: PeriodLike, ref: datetime | date | None = None
) -> tuple[datetime, datetime, str]:
    """Resolve a reporting range for `period` ending at `ref` (inclusive)."""
    ref = ref or utc_now()
    if period == "daily":
        start = start_of_day(ref)
        return start, end_of_day(ref), day_key(ref)
    start, end, key = period_window(BudgetPeriod(period), ref)
    return start, end - timedelta(microseconds=1), key


def month_start_end(ref: datetime | date) -> tuple[datetime, datetime]:
    start, end, _ = period_window(BudgetPeriod.MONTHLY, ref)
    return start, end - timedelta(microseconds=1)
