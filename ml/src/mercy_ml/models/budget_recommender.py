"""(d) Budget recommendation.

The input frame (`features.build_budget_recommendation_frame`) is deliberately
*not* supervised: there is no ground-truth "correct limit" for a budget, only a
user's own spending distribution. So rather than fitting a regressor to an
invented target, this estimator reasons directly over each user's observed
per-period spend in that category:

    base     = blend(p75, p90) of the user's per-period history
    base    *= (1 + trend)                     growth is real, not noise
    base    *= (1 + volatility headroom)       noisy history needs slack
    base    *= budget_buffer_pct               never propose a knife-edge limit
    result   = max(base, last_expense)         never below a spend already made
    result   = round up to the nearest step    a limit a person can act on

The pooled `ExpenseForecaster` can be blended in for users with enough history
(the caller passes `forecasted`), which is what lets a cold-history user still
get a sensible number.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from mercy_ml.config import get_settings
from mercy_ml.models.base import confidence_from_score, safe_float

logger = logging.getLogger(__name__)

#: Weighting between the p75 and p90 of history when forming the base ceiling.
P75_WEIGHT = 0.65
P90_WEIGHT = 0.35
#: A user needs at least this many observed periods before trend is trusted.
MIN_PERIODS_FOR_TREND = 3
#: Trend adjustments are capped so one unusual period cannot double a limit.
MAX_TREND_ADJUSTMENT = 0.35
#: Volatility contributes at most this much headroom.
MAX_VOLATILITY_HEADROOM = 0.30
#: Suggesting a limit below this share of typical spend is not worth showing.
MIN_RATIO_OF_AVG = 0.80


class BudgetRecommender:
    name = "budget_recommender"

    def __init__(
        self,
        *,
        buffer_pct: float | None = None,
        rounding: float | None = None,
        forecaster: Any = None,
        forecast_weight: float = 0.35,
    ) -> None:
        settings = get_settings()
        self.buffer_pct = settings.budget_buffer_pct if buffer_pct is None else buffer_pct
        self.rounding = settings.budget_rounding if rounding is None else rounding
        self.forecaster = forecaster
        self.forecast_weight = float(np.clip(forecast_weight, 0.0, 1.0))
        self.last_recommendation: dict[str, Any] | None = None

    # ------------------------------------------------------------- fitting
    def fit(self, frame: pd.DataFrame) -> "BudgetRecommender":
        """No-op kept for API symmetry: this estimator is closed-form.

        The only thing worth learning from the data is how much the recommended
        limit should sit above the user's average, and that is expressed as the
        explicit constants above so it stays reviewable and tunable.
        """
        self.n_reference_rows = int(len(frame)) if frame is not None else 0
        return self

    # --------------------------------------------------------- inference
    def recommend(
        self,
        stats: dict[str, Any] | pd.Series,
        *,
        current_limit: float | None = None,
        forecast: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Suggest a limit for one (user, category) scope."""
        get = (lambda key: stats.get(key, 0.0)) if isinstance(stats, dict) else (lambda key: stats.get(key, np.nan))

        avg = safe_float(get("avg_expense"), 0.0)
        p75 = safe_float(get("p75_expense"), 0.0)
        p90 = safe_float(get("p90_expense"), 0.0)
        std = safe_float(get("std_expense"), 0.0)
        last = safe_float(get("last_expense"), 0.0)
        periods = int(safe_float(get("periods_observed"), 0.0))
        trend = safe_float(get("trend_pct"), 0.0) / 100.0
        name = str(get("category_name") or "Overall")

        if avg <= 0 and p90 <= 0:
            return {
                "recommended_limit": 0.0,
                "current_limit": round(safe_float(current_limit, 0.0), 2),
                "delta": 0.0,
                "delta_pct": 0.0,
                "action": "insufficient_data",
                "rationale": "Not enough spending history in this category to suggest a limit.",
                "confidence_score": 0.05,
                "category_name": name,
            }

        # 1. Central estimate of a "typical high" month from the user's own history.
        base = P75_WEIGHT * p75 + P90_WEIGHT * p90 if p75 > 0 or p90 > 0 else avg

        # 2. Blend in the pooled forecaster when it has an opinion.
        blended_from_forecast = False
        predicted = safe_float((forecast or {}).get("predicted_amount"), 0.0)
        if self.forecaster is not None and forecast is None:
            try:
                forecast = self.forecaster.predict_with_interval(
                    _forecast_row_from_stats(stats)
                )
                predicted = safe_float(forecast.get("predicted_amount"), 0.0)
            except Exception as exc:  # pragma: no cover - scorer must never break the request
                logger.debug("Budget forecast blend skipped: %s", exc)
                forecast = None
        if predicted > 0 and self.forecast_weight > 0:
            upper = safe_float((forecast or {}).get("upper_bound"), predicted)
            base = (1.0 - self.forecast_weight) * base + self.forecast_weight * predicted
            base = max(base, upper * 0.75)
            blended_from_forecast = True

        # 3. Trend: respect genuine growth, ignore a single odd month.
        if periods >= MIN_PERIODS_FOR_TREND and trend > 0:
            base *= 1.0 + min(MAX_TREND_ADJUSTMENT, trend)

        # 4. Volatility headroom, scaled by how noisy the history actually is.
        if avg > 0:
            coefficient_of_variation = std / avg
            base *= 1.0 + min(MAX_VOLATILITY_HEADROOM, max(0.0, coefficient_of_variation) * 0.25)

        # 5. Final safety rails.
        recommended = base * (1.0 + self.buffer_pct)
        recommended = max(recommended, avg * MIN_RATIO_OF_AVG, last)
        recommended = _round_up(recommended, self.rounding)

        current = safe_float(current_limit, 0.0)
        if current > 0:
            delta = recommended - current
            delta_pct = delta / current * 100.0
        else:
            delta, delta_pct = recommended, 0.0

        action, rationale = _classify(current, recommended, avg, periods, blended_from_forecast)
        result = {
            "recommended_limit": round(recommended, 2),
            "current_limit": round(current, 2),
            "delta": round(delta, 2),
            "delta_pct": round(delta_pct, 2),
            "action": action,
            "rationale": rationale,
            "category_name": name,
            "basis": {
                "avg_expense": round(avg, 2),
                "p75_expense": round(p75, 2),
                "p90_expense": round(p90, 2),
                "std_expense": round(std, 2),
                "trend_pct": round(trend * 100.0, 2),
                "periods_observed": periods,
                "used_forecast": blended_from_forecast,
            },
            "confidence_score": confidence_from_score(_quality(avg, periods, std, blended_from_forecast), ceiling=0.88),
        }
        self.last_recommendation = result
        return result

    def recommend_all(
        self, frame: pd.DataFrame, *, current_limits: dict[Any, float] | None = None
    ) -> list[dict[str, Any]]:
        """Batch version: one recommendation per row of the statistics frame."""
        if frame is None or frame.empty:
            return []
        limits = current_limits or {}
        out: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            key = (str(row.get("user_id")), str(row.get("category_id")))
            out.append(
                self.recommend(
                    row,
                    current_limit=limits.get(key, limits.get(str(row.get("category_id")))),
                )
            )
        return out

    def is_trained(self) -> bool:  # pragma: no cover - parity with the sklearn models
        return True


def _classify(
    current: float, recommended: float, avg: float, periods: int, used_forecast: bool
) -> tuple[str, str]:
    if current <= 0:
        basis = "your recent history" if not used_forecast else "your recent history and a forecast"
        return "create", f"Suggesting {recommended:,.0f} per period based on {basis}."

    change = (recommended - current) / current if current else 0.0
    if abs(change) <= 0.05:
        return "keep", "The current limit is already within 5% of the suggested ceiling - leave it."
    if change < 0:
        return "decrease", (
            f"Recent activity supports tightening to about {recommended:,.0f} while still "
            f"covering a typical high month."
        )
    if periods < MIN_PERIODS_FOR_TREND:
        return "review", (
            f"Suggested ceiling is {recommended:,.0f} based on limited history "
            f"({periods} period(s)); raise it only if you expect higher costs."
        )
    return "increase", (
        f"Your spending pattern suggests a ceiling near {recommended:,.0f}; going lower has "
        f"historically led to overruns around {avg:,.0f}."
    )


def _quality(avg: float, periods: int, std: float, used_forecast: bool) -> float:
    depth = min(1.0, periods / 6.0)
    consistency = 1.0 / (1.0 + min(2.0, std / avg if avg > 0 else 2.0))
    signal = 1.0 if avg > 0 else 0.0
    extra = 0.08 if used_forecast else 0.0
    return float(np.clip(0.35 * depth + 0.35 * consistency + 0.30 * signal + extra, 0.0, 1.0))


def _round_up(value: float, increment: float) -> float:
    if value <= 0:
        return 0.0
    if increment <= 0:
        return round(value, 2)
    return math.ceil(value / increment) * increment


def _forecast_row_from_stats(stats: Any) -> pd.DataFrame:
    """Best-effort feature row so a forecaster can be blended in.

    The forecaster expects lag/rolling columns that the aggregate frame does not
    carry; a single-row frame with NaNs lets the pipeline's imputer supply
    neutral values rather than guessing history we do not have.
    """
    from mercy_ml.schema import FORECAST_FEATURE_COLUMNS

    get = (lambda key: stats.get(key, np.nan)) if isinstance(stats, dict) else (lambda key: stats.get(key, np.nan))
    row: dict[str, Any] = {column: np.nan for column in FORECAST_FEATURE_COLUMNS}
    row["category_name"] = str(get("category_name") or "Overall")
    row["user_tenure_periods"] = safe_float(get("periods_observed"), 0.0)
    row["user_periods_with_data"] = safe_float(get("periods_observed"), 0.0)
    row["avg_txn_amount"] = safe_float(get("avg_expense"), 0.0)
    return pd.DataFrame([row])


__all__ = ["BudgetRecommender"]
