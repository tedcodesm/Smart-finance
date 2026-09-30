"""Period arithmetic for the ML pipeline.

Duplicated from `backend/app/core/time_utils.py` on purpose: `mercy_ml` is an
independently installable package (it can run as a standalone batch process) and
must not depend on the web application's import path. The semantics are kept
identical - ISO weeks, calendar months, calendar years.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from typing import Literal

PeriodLiteral = Literal["weekly", "monthly", "yearly"]


def parse_period_key(period: str, period_key: str) -> tuple[int, int]:
    """`('monthly', '2026-09')` -> `(2026, 9)`. Weeks collapse to their Monday."""
    if period == "yearly":
        return int(period_key), 1
    if period == "weekly":
        year_s, week_s = period_key.split("-W")
        return date.fromisocalendar(int(year_s), int(week_s), 1).year, date.fromisocalendar(
            int(year_s), int(week_s), 1
        ).month
    year_s, month_s = period_key.split("-")
    return int(year_s), int(month_s)


def period_start(period: str, period_key: str) -> date:
    if period == "yearly":
        return date(int(period_key), 1, 1)
    if period == "weekly":
        year_s, week_s = period_key.split("-W")
        return date.fromisocalendar(int(year_s), int(week_s), 1)
    year, month = int(period_key.split("-")[0]), int(period_key.split("-")[1])
    return date(year, month, 1)


def period_end(period: str, period_key: str) -> date:
    """Inclusive last day of the bucket."""
    if period == "yearly":
        return date(int(period_key), 12, 31)
    if period == "weekly":
        return period_start(period, period_key) + timedelta(days=6)
    year, month = int(period_key.split("-")[0]), int(period_key.split("-")[1])
    return date(year, month, calendar.monthrange(year, month)[1])


def period_length(period: str, period_key: str) -> int:
    start, end = period_start(period, period_key), period_end(period, period_key)
    return (end - start).days + 1


def next_period_key(period: str, period_key: str) -> str:
    """The bucket key immediately after `period_key`."""
    if period == "yearly":
        return f"{int(period_key) + 1:04d}"
    if period == "weekly":
        return week_key(period_start(period, period_key) + timedelta(days=7))
    year, month = int(period_key.split("-")[0]), int(period_key.split("-")[1])
    if month == 12:
        return f"{year + 1:04d}-01"
    return f"{year:04d}-{month + 1:02d}"


def previous_period_key(period: str, period_key: str) -> str:
    if period == "yearly":
        return f"{int(period_key) - 1:04d}"
    if period == "weekly":
        return week_key(period_start(period, period_key) - timedelta(days=7))
    year, month = int(period_key.split("-")[0]), int(period_key.split("-")[1])
    if month == 1:
        return f"{year - 1:04d}-12"
    return f"{year:04d}-{month - 1:02d}"


def month_key(value: date | datetime) -> str:
    d = value.date() if isinstance(value, datetime) else value
    return f"{d.year:04d}-{d.month:02d}"


def week_key(value: date | datetime) -> str:
    d = value.date() if isinstance(value, datetime) else value
    iso = d.isocalendar()
    return f"{iso.year:04d}-W{iso.week:02d}"


def key_for(period: str, value: date | datetime) -> str:
    if period == "weekly":
        return week_key(value)
    if period == "yearly":
        d = value.date() if isinstance(value, datetime) else value
        return f"{d.year:04d}"
    return month_key(value)


def keys_between(period: str, start: date, end: date) -> list[str]:
    """All bucket keys covering the inclusive date range."""
    keys: list[str] = []
    if period == "yearly":
        for year in range(start.year, end.year + 1):
            keys.append(f"{year:04d}")
        return keys
    if period == "weekly":
        cursor = period_start("weekly", week_key(start))
        while cursor <= end:
            keys.append(week_key(cursor))
            cursor += timedelta(days=7)
        return keys

    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        keys.append(f"{year:04d}-{month:02d}")
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return keys


def days_elapsed(period: str, period_key: str, as_of: date) -> int:
    """Whole days of the bucket that have passed as of `as_of` (0-based)."""
    start = period_start(period, period_key)
    return max(0, min(period_length(period, period_key), (as_of - start).days + 1))


def elapsed_fraction(period: str, period_key: str, as_of: date) -> float:
    length = period_length(period, period_key)
    return min(1.0, max(0.0, days_elapsed(period, period_key, as_of) / length))


def month_sin_cos(month: int) -> tuple[float, float]:
    import math

    angle = 2.0 * math.pi * (month - 1) / 12.0
    return math.sin(angle), math.cos(angle)
