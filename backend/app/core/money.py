"""Money helpers.

Money is persisted as BSON double (float64) for painless interchange with the
pandas/scikit-learn pipeline, but every value crossing the API or service
boundary is quantised to 2 decimal places through `Decimal` so that repeated
aggregation cannot accumulate binary-float drift.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable

TWO_PLACES = Decimal("0.01")


def to_decimal(value: float | int | Decimal | str | None, default: str = "0") -> Decimal:
    if value is None:
        return Decimal(default)
    return Decimal(str(value))


def round_money(value: float | int | Decimal | str | None) -> float:
    """Quantise to 2 dp using banker-free half-up rounding, return as float."""
    return float(to_decimal(value).quantize(TWO_PLACES, rounding=ROUND_HALF_UP))


def add_money(*values: float | int | Decimal | str | None) -> float:
    total = Decimal("0")
    for value in values:
        total += to_decimal(value)
    return float(total.quantize(TWO_PLACES, rounding=ROUND_HALF_UP))


def sum_money(values: Iterable[float | int | Decimal | str | None]) -> float:
    return add_money(*values)


def is_positive(value: float | int | Decimal | str | None) -> bool:
    return to_decimal(value) > 0


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that yields `default` instead of raising/NaN on a zero divisor."""
    if not denominator:
        return default
    return numerator / denominator


def percent_of(part: float, whole: float) -> float:
    """`part` as a percentage of `whole`; 0 when `whole` is 0."""
    if not whole:
        return 0.0
    return (part / whole) * 100.0
