"""Feature engineering.

Builds the four training/inference frames. The design constraint from the brief
is that a **single pooled model** must serve every user, so every feature is
either

  * a per-user *relative* measure (share of the user's spend, amount vs. the
    user's own category mean), which normalises away the scale difference
    between a student and a CTO, or
  * a calendar feature, shared by everyone.

That is what makes per-user inputs meaningful to a global model.

`SpendMatrices` is the workhorse: it pivots transactions into dense
(user x category) x date frames with running totals, so "how much had this user
spent in this category by day 14 of March" is an O(1) lookup rather than a scan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from mercy_ml import periods as P
from mercy_ml.config import get_settings
from mercy_ml.preprocess import log1p_amount, safe_ratio
from mercy_ml.schema import (
    CATEGORY,
    CATEGORY_NAME,
    CLUSTER_FEATURE_COLUMNS,
    CLUSTER_NUMERIC_FEATURES,
    FORECAST_FEATURE_COLUMNS,
    OVERSPEND_FEATURE_COLUMNS,
    SAVINGS_FEATURE_COLUMNS,
    USER,
)

logger = logging.getLogger(__name__)

#: Days into a period at which we take a "snapshot" during training. Inference
#: happens on an arbitrary date, so the model must generalise across positions.
MONTHLY_SNAPSHOTS: tuple[int, ...] = (7, 14, 21, 28)
WEEKLY_SNAPSHOTS: tuple[int, ...] = (2, 4, 6)
YEARLY_SNAPSHOTS: tuple[int, ...] = (60, 150, 240, 330)


def snapshots_for(period: str) -> tuple[int, ...]:
    if period == "weekly":
        return WEEKLY_SNAPSHOTS
    if period == "yearly":
        return YEARLY_SNAPSHOTS
    return MONTHLY_SNAPSHOTS


# --------------------------------------------------------------------------- #
# Spend matrices
# --------------------------------------------------------------------------- #
@dataclass
class SpendMatrices:
    """Dense daily spend with running totals, indexed by date."""

    expense_daily: pd.DataFrame  # (user, category) x date
    income_daily: pd.DataFrame  # user x date
    expense_cum: pd.DataFrame
    income_cum: pd.DataFrame
    date_index: pd.DatetimeIndex

    # ------------------------------------------------------------- builders
    @classmethod
    def build(cls, transactions: pd.DataFrame) -> "SpendMatrices":
        expenses = transactions[transactions["type"] == "expense"]
        incomes = transactions[transactions["type"] == "income"]

        expense_daily = _pivot_amount(expenses)
        income_daily = _pivot_amount(incomes, per_user_only=True)

        # Union of dates so lookups never fall off the end of the axis.
        all_dates = pd.DatetimeIndex(sorted(set(expense_daily.columns) | set(income_daily.columns)))
        expense_daily = expense_daily.reindex(columns=all_dates, fill_value=0.0)
        income_daily = income_daily.reindex(columns=all_dates, fill_value=0.0)
        expense_daily = expense_daily.astype("float64")
        income_daily = income_daily.astype("float64")

        return cls(
            expense_daily=expense_daily,
            income_daily=income_daily,
            expense_cum=expense_daily.cumsum(axis=1),
            income_cum=income_daily.cumsum(axis=1),
            date_index=all_dates,
        )

    # -------------------------------------------------------------- lookups
    def _slice(self, cum: pd.DataFrame, key: Any, start: date, end: date) -> float:
        """Total in `[start, end]` from a running-total frame."""
        if cum.empty:
            return 0.0
        if key not in cum.index:
            return 0.0
        row = cum.loc[key]
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        before = float(row.loc[row.index < start_ts].iloc[-1]) if (row.index < start_ts).any() else 0.0
        upto = float(row.loc[row.index <= end_ts].iloc[-1]) if (row.index <= end_ts).any() else 0.0
        return max(0.0, upto - before)

    def expense_between(
        self, user_id: Any, category_id: Any, start: date, end: date, *, sum_categories: bool = False
    ) -> float:
        if sum_categories or category_id is None:
            rows = self.expense_cum.loc[self.expense_cum.index.get_level_values(USER) == user_id]
            if rows.empty:
                return 0.0
            combined = rows.sum(axis=0)
            start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
            before = float(combined[combined.index < start_ts].iloc[-1]) if (combined.index < start_ts).any() else 0.0
            upto = float(combined[combined.index <= end_ts].iloc[-1]) if (combined.index <= end_ts).any() else 0.0
            return max(0.0, upto - before)
        return self._slice(self.expense_cum, (user_id, category_id), start, end)

    def income_between(self, user_id: Any, start: date, end: date) -> float:
        return self._slice(self.income_cum, user_id, start, end)

    def user_expense_total(self, user_id: Any) -> float:
        rows = self.expense_daily.loc[self.expense_daily.index.get_level_values(USER) == user_id]
        return float(rows.to_numpy().sum()) if not rows.empty else 0.0

    def user_categories(self, user_id: Any) -> list[Any]:
        return list(self.expense_daily.loc[self.expense_daily.index.get_level_values(USER) == user_id].index.get_level_values(1))


def _pivot_amount(frame: pd.DataFrame, *, per_user_only: bool = False) -> pd.DataFrame:
    if frame.empty:
        index = [USER] if per_user_only else [USER, CATEGORY]
        return pd.DataFrame(0.0, index=pd.Index([], name="tuple"), columns=pd.DatetimeIndex([]))
    keys = [USER] if per_user_only else [USER, CATEGORY]
    pivot = frame.pivot_table(index=keys, columns="date", values="amount", aggfunc="sum", observed=True)
    return pivot.fillna(0.0)


# --------------------------------------------------------------------------- #
# Per-(user, period) history used by every model
# --------------------------------------------------------------------------- #
@dataclass
class PeriodHistory:
    """Per-(user, period) aggregates plus the look-back slices models need."""

    frame: pd.DataFrame  # indexed by (user_id, period_key)
    period_keys: dict[Any, list[str]]  # user -> sorted period keys with data
    income: dict[tuple[Any, str], float]
    expense: dict[tuple[Any, str], float]
    txn_count: dict[tuple[Any, str], int]
    category_period_expense: dict[tuple[Any, Any, str], float]
    category_period_count: dict[tuple[Any, Any, str], int]

    @classmethod
    def build(cls, transactions: pd.DataFrame) -> "PeriodHistory":
        if transactions.empty:
            return cls.empty()

        data = transactions.copy()
        data["signed"] = np.where(data["type"] == "income", data["amount"], 0.0)
        data["expense_only"] = np.where(data["type"] == "expense", data["amount"], 0.0)

        grouped = data.groupby([USER, "period_key"], observed=True)
        frame = grouped.agg(
            period_income=("signed", "sum"),
            period_expense=("expense_only", "sum"),
            txn_count=("amount", "size"),
            distinct_categories=(CATEGORY, "nunique"),
        )
        frame["period_net_savings"] = frame["period_income"] - frame["period_expense"]
        frame["recurring_share"] = grouped["is_recurring"].mean()
        frame = frame.reset_index()

        period_keys = (
            frame.groupby(USER, observed=True)["period_key"].apply(lambda s: sorted(s.astype(str))).to_dict()
        )

        cat_expense = (
            data[data["type"] == "expense"]
            .groupby([USER, CATEGORY, "period_key"], observed=True)
            .agg(total=("amount", "sum"), count=("amount", "size"))
        )
        category_period_expense = {key: float(row.total) for key, row in cat_expense.iterrows()}
        category_period_count = {key: int(row["count"]) for key, row in cat_expense.iterrows()}

        return cls(
            frame=frame,
            period_keys=period_keys,
            # `frame` is already reset, so itertuples fields are the real columns.
            income={(r.user_id, r.period_key): float(r.period_income) for r in frame.itertuples()},
            expense={(r.user_id, r.period_key): float(r.period_expense) for r in frame.itertuples()},
            txn_count={(r.user_id, r.period_key): int(r.txn_count) for r in frame.itertuples()},
            category_period_expense=category_period_expense,
            category_period_count=category_period_count,
        )

    @classmethod
    def empty(cls) -> "PeriodHistory":
        return cls(
            frame=pd.DataFrame(
                columns=[USER, "period_key", "period_income", "period_expense", "txn_count",
                         "distinct_categories", "period_net_savings", "recurring_share"]
            ),
            period_keys={},
            income={},
            expense={},
            txn_count={},
            category_period_expense={},
            category_period_count={},
        )

    # ------------------------------------------------------------- accessors
    def keys_before(self, user_id: Any, period_key: str) -> list[str]:
        """Sorted period keys for a user strictly before `period_key`."""
        return [key for key in self.period_keys.get(user_id, []) if key < period_key]

    def trailing(self, mapping: dict[tuple[Any, str], float], user_id: Any, period_key: str, n: int) -> list[float]:
        history = self.keys_before(user_id, period_key)[-n:]
        return [float(mapping.get((user_id, key), 0.0)) for key in history]

    def trailing_expense(self, user_id: Any, period_key: str, n: int) -> list[float]:
        return self.trailing(self.expense, user_id, period_key, n)

    def trailing_income(self, user_id: Any, period_key: str, n: int) -> list[float]:
        return self.trailing(self.income, user_id, period_key, n)

    def category_trailing(self, user_id: Any, category_id: Any, period_key: str, n: int) -> list[float]:
        history = self.keys_before(user_id, period_key)[-n:]
        return [float(self.category_period_expense.get((user_id, category_id, key), 0.0)) for key in history]

    def periods_observed(self, user_id: Any) -> int:
        return len(self.period_keys.get(user_id, []))

    def user_period_expense(self, user_id: Any) -> float:
        return sum(value for (uid, _), value in self.expense.items() if uid == user_id)


# --------------------------------------------------------------------------- #
# (c) Overspend classification
# --------------------------------------------------------------------------- #
def build_overspend_frame(
    transactions: pd.DataFrame,
    budgets: pd.DataFrame,
    categories: pd.DataFrame,
    *,
    period: str = "monthly",
    min_history: int | None = None,
) -> pd.DataFrame:
    """One row per (budget, period, snapshot day) predicting a budget overrun.

    Snapshots are taken part-way through the period so the training distribution
    matches live inference, which always sees a partially-elapsed budget.
    """
    settings = get_settings()
    min_history = settings.min_months_history if min_history is None else min_history
    if budgets.empty or transactions.empty:
        return _empty_feature_frame(OVERSPEND_FEATURE_COLUMNS)

    matrices = SpendMatrices.build(transactions)
    history = PeriodHistory.build(transactions)
    names = _category_names(categories)

    rows: list[dict[str, Any]] = []
    for budget in budgets.itertuples():
        user_id = budget.user_id
        category_id = getattr(budget, "category_id", None)
        limit = float(budget.limit_amount) if pd.notna(budget.limit_amount) else 0.0
        if limit <= 0:
            continue

        budget_period = str(getattr(budget, "period", period) or period)
        if budget_period != period:
            continue
        overall = category_id is None or (isinstance(category_id, float) and np.isnan(category_id))

        user_periods = history.period_keys.get(user_id, [])
        for period_key in user_periods:
            start = P.period_start(period, period_key)
            end = P.period_end(period, period_key)
            if start > matrices.date_index.max().date():
                continue

            full_spend = matrices.expense_between(
                user_id, None if overall else category_id, start, end, sum_categories=overall
            )
            prior_keys = history.keys_before(user_id, period_key)
            if len(prior_keys) < min_history:
                continue

            label = int(full_spend > limit)
            category_name = "Overall" if overall else names.get(category_id, "Unknown")
            if overall:
                # An all-categories budget only teaches the model something when
                # its limit is realistic; otherwise it is pure label noise.
                if full_spend > limit * 3:
                    continue

            for snapshot in snapshots_for(period):
                as_of = min(start + timedelta(days=snapshot - 1), end)
                elapsed_days = (as_of - start).days + 1
                length = P.period_length(period, period_key)
                elapsed_pct = elapsed_days / length
                spent_so_far = matrices.expense_between(
                    user_id, None if overall else category_id, start, as_of, sum_categories=overall
                )
                counts = 0
                if not overall:
                    counts = sum(
                        history.category_period_count.get((user_id, category_id, key), 0)
                        for key in prior_keys[-min_history:]
                    )

                scope_history = (
                    history.trailing_expense(user_id, period_key, 3) if overall
                    else history.category_trailing(user_id, category_id, period_key, 3)
                )
                rows.append(
                    _overspend_row(
                        user_id=user_id,
                        category_id=None if overall else category_id,
                        category_name=category_name,
                        period_key=period_key,
                        limit=limit,
                        spent_so_far=spent_so_far,
                        elapsed_pct=elapsed_pct,
                        days_remaining=max(0, length - elapsed_days),
                        period_length=length,
                        matrices=matrices,
                        history=history,
                        scope_history=scope_history,
                        prior_period_count=counts,
                        label=label,
                    )
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return _empty_feature_frame(OVERSPEND_FEATURE_COLUMNS)
    return _finalise(frame, OVERSPEND_FEATURE_COLUMNS, "label_overspend")


def _overspend_row(
    *,
    user_id: Any,
    category_id: Any,
    category_name: str,
    period_key: str,
    limit: float,
    spent_so_far: float,
    elapsed_pct: float,
    days_remaining: int,
    period_length: int,
    matrices: SpendMatrices,
    history: PeriodHistory,
    scope_history: list[float],
    prior_period_count: int,
    label: int,
) -> dict[str, Any]:
    window = 3
    user_income_hist = history.trailing_income(user_id, period_key, window)
    user_expense_hist = history.trailing_expense(user_id, period_key, window)

    avg_income = _mean(user_income_hist)
    avg_expense = _mean(user_expense_hist)
    scope_avg = _mean(scope_history)

    prior = history.keys_before(user_id, period_key)
    # A period counts as an overspend when its spend exceeded 1.15x the trailing
    # mean - a stable proxy for "did this user historically overrun" for scopes
    # where no explicit budget record exists for the earlier period.
    overspend_count = _count_above_mean_overspend(scope_history)
    overspend_rate = safe_ratio(overspend_count, len(scope_history), 0.0)

    month = int(period_key.split("-")[1]) if "-" in period_key else 1
    sin, cos = P.month_sin_cos(month)

    return {
        USER: str(user_id),
        "category_id": str(category_id) if category_id is not None else None,
        CATEGORY_NAME: category_name,
        "period_key": period_key,
        "limit_amount": limit,
        "spent_so_far": spent_so_far,
        "period_elapsed_pct": round(elapsed_pct * 100.0, 4),
        "days_remaining": days_remaining,
        "spend_velocity": round(safe_ratio(spent_so_far, max(1, period_length * elapsed_pct)), 4),
        "projected_total": round(safe_ratio(spent_so_far, elapsed_pct, spent_so_far), 4),
        "utilisation_pct": round(safe_ratio(spent_so_far, limit) * 100.0, 4),
        "txn_count_so_far": prior_period_count,
        "avg_txn_amount": round(safe_ratio(scope_avg, max(1, prior_period_count)), 4),
        "scope_avg_period_expense": round(scope_avg, 4),
        "scope_expense_trend_pct": round(_trend_pct(scope_history) * 100.0, 4),
        "user_avg_period_income": round(avg_income, 4),
        "user_avg_period_expense": round(avg_expense, 4),
        "user_expense_income_ratio": round(safe_ratio(avg_expense, avg_income), 4),
        "user_overspend_rate": round(float(overspend_rate), 4),
        "user_overspend_count": float(overspend_count),
        "user_periods_observed": float(len(prior)),
        "user_tenure_periods": float(history.periods_observed(user_id)),
        "month_sin": sin,
        "month_cos": cos,
        "is_month_start": int(period_key.endswith("-01")),
        "is_month_end": int(elapsed_pct > 0.92),
        "label_overspend": label,
    }


def _count_above_mean_overspend(values: Sequence[float]) -> int:
    """Count periods that spent more than 1.15x the trailing mean."""
    if not values:
        return 0
    baseline = _mean(values)
    if baseline <= 0:
        return 0
    return sum(1 for value in values if value > baseline * 1.15)


def build_overspend_inference_row(
    *,
    user_id: Any,
    category_id: Any,
    category_name: str,
    limit_amount: float,
    spent_so_far: float,
    as_of: date,
    period: str,
    period_key: str,
    transactions: pd.DataFrame,
    categories: pd.DataFrame,
) -> dict[str, Any]:
    """Feature vector for one live budget, matching the training distribution."""
    matrices = SpendMatrices.build(transactions)
    history = PeriodHistory.build(transactions)
    start = P.period_start(period, period_key)
    end = P.period_end(period, period_key)
    elapsed_days = max(1, (as_of - start).days + 1)
    length = P.period_length(period, period_key)
    elapsed_pct = min(1.0, elapsed_days / length)
    overall = category_id is None

    scope_history = (
        history.trailing_expense(user_id, period_key, 3) if overall
        else history.category_trailing(user_id, category_id, period_key, 3)
    )
    prior_keys = history.keys_before(user_id, period_key)
    txn_count = (
        sum(history.category_period_count.get((user_id, category_id, key), 0) for key in prior_keys[-3:])
        if not overall
        else sum(history.txn_count.get((user_id, key), 0) for key in prior_keys[-3:])
    )

    row = _overspend_row(
        user_id=user_id,
        category_id=category_id,
        category_name="Overall" if overall else category_name,
        period_key=period_key,
        limit=limit_amount,
        spent_so_far=spent_so_far,
        elapsed_pct=elapsed_pct,
        days_remaining=max(0, length - elapsed_days),
        period_length=length,
        matrices=matrices,
        history=history,
        scope_history=scope_history,
        prior_period_count=txn_count,
        label=0,  # unused at inference time
    )
    row["days_remaining"] = max(0, (end - as_of).days)
    return row


# --------------------------------------------------------------------------- #
# (b) Expense forecasting
# --------------------------------------------------------------------------- #
def build_forecast_frame(
    transactions: pd.DataFrame,
    categories: pd.DataFrame,
    *,
    period: str = "monthly",
    min_history: int | None = None,
) -> pd.DataFrame:
    """One row per (user, category, period) predicting the *next* period's spend."""
    settings = get_settings()
    min_history = settings.min_months_history if min_history is None else min_history
    if transactions.empty:
        return _empty_feature_frame(FORECAST_FEATURE_COLUMNS)

    expenses = transactions[transactions["type"] == "expense"]
    if expenses.empty:
        return _empty_feature_frame(FORECAST_FEATURE_COLUMNS)

    history = PeriodHistory.build(transactions)
    names = _category_names(categories)

    rows: list[dict[str, Any]] = []
    for (user_id, category_id, period_key), total in history.category_period_expense.items():
        prior = history.keys_before(user_id, period_key)
        if len(prior) < min_history:
            continue
        if history.category_period_count.get((user_id, category_id, period_key), 0) < settings.min_transactions_per_period:
            continue

        row = _forecast_row(
            user_id=user_id,
            category_id=category_id,
            category_name=names.get(category_id, "Unknown"),
            period_key=period_key,
            history=history,
            category_current=float(total),
            category_current_count=history.category_period_count.get((user_id, category_id, period_key), 0),
        )
        row["target_next_expense"] = history.category_period_expense.get(
            (user_id, category_id, P.next_period_key(period, period_key)), 0.0
        )
        if row["target_next_expense"] <= 0:
            # No spend next period is a real outcome, but training on "absent row"
            # as zero would teach the model that every category is monthly.
            continue
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return _empty_feature_frame(FORECAST_FEATURE_COLUMNS)
    return _finalise(frame, FORECAST_FEATURE_COLUMNS, "target_next_expense")


def _forecast_row(
    *,
    user_id: Any,
    category_id: Any,
    category_name: str,
    period_key: str,
    history: PeriodHistory,
    category_current: float,
    category_current_count: int,
) -> dict[str, Any]:
    values = history.category_trailing(user_id, category_id, period_key, 3)
    if not values:
        values = [category_current]
    values = values[-3:]

    lag_1 = values[-1]
    lag_2 = values[-2] if len(values) > 1 else lag_1
    lag_3 = values[-3] if len(values) > 2 else lag_2

    income_hist = history.trailing_income(user_id, period_key, 3)
    period_income = income_hist[-1] if income_hist else 0.0
    user_expense = history.expense.get((user_id, period_key), 0.0)

    month = int(period_key.split("-")[1]) if "-" in period_key else 1
    sin, cos = P.month_sin_cos(month)

    return {
        USER: str(user_id),
        "category_id": str(category_id),
        CATEGORY_NAME: category_name,
        "period_key": period_key,
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "rolling_mean_3": _mean(values),
        "rolling_std_3": _std(values),
        "rolling_max_3": max(values),
        "expense_trend_slope": _slope(values),
        "expense_trend_pct": _trend_pct(values) * 100.0,
        "rel_to_user_period_expense": safe_ratio(category_current, user_expense),
        "period_income": period_income,
        "period_income_trend_pct": _trend_pct(income_hist) * 100.0,
        "txn_count": category_current_count,
        "avg_txn_amount": safe_ratio(category_current, max(1, category_current_count)),
        "expense_share_of_period": safe_ratio(category_current, user_expense),
        "month_sin": sin,
        "month_cos": cos,
        "is_month_start": int(period_key.endswith("-01")),
        "is_month_end": int(period_key.endswith(("-03", "-06", "-09", "-12"))),
        "user_tenure_periods": float(history.periods_observed(user_id)),
        "user_periods_with_data": float(len(history.keys_before(user_id, period_key)) + 1),
    }


def build_forecast_inference_row(
    *,
    user_id: Any,
    category_id: Any,
    category_name: str,
    period_key: str,
    transactions: pd.DataFrame,
    categories: pd.DataFrame,
    period: str = "monthly",
) -> dict[str, Any]:
    """Feature vector for the upcoming period, from history up to `period_key`."""
    history = PeriodHistory.build(transactions)
    current = history.category_period_expense.get((user_id, category_id, period_key), 0.0)
    count = history.category_period_count.get((user_id, category_id, period_key), 0)
    return _forecast_row(
        user_id=user_id,
        category_id=category_id,
        category_name=category_name,
        period_key=period_key,
        history=history,
        category_current=current,
        category_current_count=count,
    )


# --------------------------------------------------------------------------- #
# (d) Savings recommendation
# --------------------------------------------------------------------------- #
def build_savings_frame(
    transactions: pd.DataFrame,
    *,
    period: str = "monthly",
    min_history: int | None = None,
) -> pd.DataFrame:
    """One row per (user, period) predicting that user's next-period net savings."""
    settings = get_settings()
    min_history = settings.min_months_history if min_history is None else min_history
    if transactions.empty:
        return _empty_feature_frame(SAVINGS_FEATURE_COLUMNS)

    history = PeriodHistory.build(transactions)
    rows: list[dict[str, Any]] = []
    for user_id, keys in history.period_keys.items():
        for period_key in keys:
            prior = history.keys_before(user_id, period_key)
            if len(prior) < min_history:
                continue
            income = history.income.get((user_id, period_key), 0.0)
            expense = history.expense.get((user_id, period_key), 0.0)
            target = history.income.get(
                (user_id, P.next_period_key(period, period_key)), 0.0
            ) - history.expense.get((user_id, P.next_period_key(period, period_key)), 0.0)
            if target <= 0:
                continue  # a user who spends more than they earn has no savings to model
            rows.append(
                _savings_row(
                    user_id=user_id,
                    period_key=period_key,
                    income=income,
                    expense=expense,
                    history=history,
                    target_next_net_savings=target,
                )
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return _empty_feature_frame(SAVINGS_FEATURE_COLUMNS)
    return _finalise(frame, SAVINGS_FEATURE_COLUMNS, "target_next_net_savings")


def _savings_row(
    *,
    user_id: Any,
    period_key: str,
    income: float,
    expense: float,
    history: PeriodHistory,
    target_next_net_savings: float | None = None,
) -> dict[str, Any]:
    income_hist = history.trailing_income(user_id, period_key, 3)
    expense_hist = history.trailing_expense(user_id, period_key, 3)
    month = int(period_key.split("-")[1]) if "-" in period_key else 1
    sin, cos = P.month_sin_cos(month)

    row = {
        USER: str(user_id),
        "period_key": period_key,
        "period_income": income,
        "period_expense": expense,
        "period_net_savings": income - expense,
        "avg_income_3": _mean(income_hist),
        "avg_expense_3": _mean(expense_hist),
        "income_trend_pct": _trend_pct(income_hist) * 100.0,
        "expense_trend_pct": _trend_pct(expense_hist) * 100.0,
        "income_volatility": _std(income_hist),
        "expense_volatility": _std(expense_hist),
        "savings_rate": safe_ratio(income - expense, income),
        "txn_count": history.txn_count.get((user_id, period_key), 0),
        "distinct_categories": _distinct_categories(history, user_id, period_key),
        "recurring_share": _recurring_share(history, user_id, period_key),
        "user_tenure_periods": float(history.periods_observed(user_id)),
        "month_sin": sin,
        "month_cos": cos,
    }
    if target_next_net_savings is not None:
        row["target_next_net_savings"] = target_next_net_savings
    return row


def build_savings_inference_row(
    *, user_id: Any, period_key: str, transactions: pd.DataFrame, period: str = "monthly"
) -> dict[str, Any]:
    history = PeriodHistory.build(transactions)
    return _savings_row(
        user_id=user_id,
        period_key=period_key,
        income=history.income.get((user_id, period_key), 0.0),
        expense=history.expense.get((user_id, period_key), 0.0),
        history=history,
    )


# --------------------------------------------------------------------------- #
# (a) Spending-pattern clustering
# --------------------------------------------------------------------------- #
def build_cluster_frame(transactions: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
    """One row per expense transaction, with user-relative spending features."""
    if transactions.empty:
        return _empty_feature_frame(CLUSTER_FEATURE_COLUMNS)

    expenses = transactions[transactions["type"] == "expense"].copy()
    if expenses.empty:
        return _empty_feature_frame(CLUSTER_FEATURE_COLUMNS)

    expenses["log_amount"] = log1p_amount(expenses["amount"])

    # User-relative measures: how big is this purchase against the user's own
    # behaviour in the same category? This is what lets one global clustering
    # describe a $3 coffee run and a $900 rent payment comparably.
    group = expenses.groupby([USER, CATEGORY], observed=True)["amount"]
    mean_amount = group.transform("mean")
    std_amount = group.transform("std").replace(0, np.nan)
    expenses["amount_zscore_in_category"] = ((expenses["amount"] - mean_amount) / std_amount).fillna(0.0).clip(-5, 5)
    expenses["amount_ratio_in_category"] = (expenses["amount"] / mean_amount.replace(0, np.nan)).fillna(1.0).clip(0, 10)

    period_expense = expenses.groupby([USER, "period_key"], observed=True)["amount"].sum()
    category_period = expenses.groupby([USER, CATEGORY, "period_key"], observed=True)["amount"].sum()

    expenses["user_period_expense"] = (
        expenses.set_index([USER, "period_key"]).index.map(period_expense).astype("float64")
    )
    expenses["category_period_expense"] = (
        expenses.set_index([USER, CATEGORY, "period_key"]).index.map(category_period).astype("float64")
    )
    expenses["category_share_of_period"] = (
        expenses["category_period_expense"] / expenses["user_period_expense"].replace(0, np.nan)
    ).fillna(0.0)
    expenses["day_of_week"] = expenses["day_of_week"].astype("int64")
    expenses["is_weekend"] = (expenses["day_of_week"] >= 5).astype("int64")
    expenses["is_month_start"] = expenses["day_of_month"].le(3).astype("int64")
    expenses["is_month_end"] = expenses["day_of_month"].ge(28).astype("int64")
    # No hour-of-day is stored, so bucket the creation timestamp where present.
    if "created_at" in expenses:
        expenses["txn_hour_bucket"] = (
            pd.to_datetime(expenses["created_at"], errors="coerce").dt.hour.fillna(12) // 6
        )
    else:
        expenses["txn_hour_bucket"] = 2
    expenses["txn_hour_bucket"] = expenses["txn_hour_bucket"].astype("int64").clip(0, 3)

    week = expenses["week_of_year"].astype("float64")
    expenses["week_of_year_sin"] = np.sin(2.0 * np.pi * week / 52.0)
    expenses["week_of_year_cos"] = np.cos(2.0 * np.pi * week / 52.0)

    month = expenses["month"].astype("float64")
    expenses["month_sin"] = np.sin(2.0 * np.pi * (month - 1) / 12.0)
    expenses["month_cos"] = np.cos(2.0 * np.pi * (month - 1) / 12.0)

    names = _category_names(categories)
    expenses[CATEGORY_NAME] = expenses[CATEGORY].map(lambda cid: names.get(cid, "Unknown"))
    expenses[USER] = expenses[USER].astype(str)
    expenses["category_id"] = expenses[CATEGORY].astype(str)

    frame = expenses.copy()
    frame = _finalise(frame, CLUSTER_FEATURE_COLUMNS, target=None, id_columns=[USER, CATEGORY, "date"])
    return frame


# --------------------------------------------------------------------------- #
# (d) Budget recommendation inputs
# --------------------------------------------------------------------------- #
def build_budget_recommendation_frame(
    transactions: pd.DataFrame, categories: pd.DataFrame, *, period: str = "monthly"
) -> pd.DataFrame:
    """Per (user, category) statistics used to suggest a limit.

    Not a supervised frame - `BudgetRecommender` is a statistical estimator over
    these aggregates, optionally blended with the pooled forecaster.
    """
    if transactions.empty:
        return pd.DataFrame(
            columns=[USER, CATEGORY, CATEGORY_NAME, "periods_observed", "avg_expense", "median_expense",
                     "std_expense", "trend_pct", "p75_expense", "p90_expense", "last_expense",
                     "last_period_key"]
        )
    history = PeriodHistory.build(transactions)
    names = _category_names(categories)

    rows: list[dict[str, Any]] = []
    for user_id, keys in history.period_keys.items():
        for category_id in {key[1] for key in history.category_period_expense if key[0] == user_id}:
            values = [history.category_period_expense[(user_id, category_id, key)] for key in keys
                      if (user_id, category_id, key) in history.category_period_expense]
            if not values:
                continue
            series = pd.Series(values, dtype="float64")
            rows.append(
                {
                    USER: str(user_id),
                    "category_id": str(category_id),
                    CATEGORY_NAME: names.get(category_id, "Unknown"),
                    "periods_observed": len(values),
                    "avg_expense": float(series.mean()),
                    "median_expense": float(series.median()),
                    "std_expense": float(series.std()) if len(values) > 1 else 0.0,
                    "trend_pct": _trend_pct(list(series)) * 100.0,
                    "p75_expense": float(series.quantile(0.75)),
                    "p90_expense": float(series.quantile(0.90)),
                    "last_expense": float(series.iloc[-1]),
                    "last_period_key": keys[-1],
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #
def _mean(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(clean)) if clean else 0.0


def _std(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0


def _slope(values: Sequence[float]) -> float:
    """Least-squares slope over the (ordered) series."""
    clean = [(i, float(v)) for i, v in enumerate(values) if v is not None and not pd.isna(v)]
    if len(clean) < 2:
        return 0.0
    xs = np.array([i for i, _ in clean], dtype="float64")
    ys = np.array([v for _, v in clean], dtype="float64")
    denominator = float(((xs - xs.mean()) ** 2).sum())
    return float(((xs - xs.mean()) * (ys - ys.mean())).sum() / denominator) if denominator else 0.0


def _trend_pct(values: Sequence[float]) -> float:
    """Average per-period growth rate as a fraction (0.12 == +12%)."""
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if len(clean) < 2:
        return 0.0
    first, last = _mean(clean[: max(1, len(clean) // 2)]), _mean(clean[max(1, len(clean) // 2):])
    if abs(first) < 1e-6:
        return 0.0
    return max(-1.0, min(3.0, (last - first) / abs(first)))


def _category_names(categories: pd.DataFrame) -> dict[Any, str]:
    if categories is None or categories.empty:
        return {}
    if CATEGORY_NAME not in categories:
        return {}
    return dict(zip(categories["category_id"], categories[CATEGORY_NAME].astype(str)))


def _distinct_categories(history: PeriodHistory, user_id: Any, period_key: str) -> float:
    return float(
        sum(1 for (uid, _cid, key) in history.category_period_expense if uid == user_id and key == period_key)
    )


def _recurring_share(history: PeriodHistory, user_id: Any, period_key: str) -> float:
    counts = sum(history.category_period_count.get((user_id, cid, period_key), 0)
                 for (uid, cid, key) in history.category_period_count if uid == user_id and key == period_key)
    total = history.txn_count.get((user_id, period_key), 0)
    return safe_ratio(counts, total, 0.0)


def _empty_feature_frame(columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({column: pd.Series(dtype="float64") for column in columns})
    return frame


def _finalise(
    frame: pd.DataFrame,
    columns: list[str],
    target: str | None,
    *,
    id_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Coerce every contract column to float and order the frame deterministically."""
    from mercy_ml.preprocess import coerce_frame_types

    numeric = [c for c in columns if c not in (CATEGORY_NAME,)]
    data = coerce_frame_types(frame, numeric)
    for column in (CATEGORY_NAME,):
        if column not in data:
            data[column] = "Unknown"
        data[column] = data[column].fillna("Unknown").astype(str)

    ordered = [c for c in columns if c in data.columns]
    if target and target in data.columns:
        ordered.append(target)
    for column in id_columns:
        if column in data.columns and column not in ordered:
            ordered.append(column)
    if USER in data.columns and USER not in ordered:
        ordered.insert(0, USER)
    return data[ordered].reset_index(drop=True)
