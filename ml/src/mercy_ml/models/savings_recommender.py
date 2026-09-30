"""(e) Savings recommendation - next-period net savings, then an allocation.

The supervised part forecasts a user's **net savings for the next period** from
their own recent cash-flow dynamics (income/expense levels, trends, volatility,
savings rate, tenure, seasonality). Pooling across users is viable because every
input is either a level or a rate - the model never needs to know whether the
user earns 900 or 9,000 to reason about their behaviour.

The recommendation step is then business logic rather than more regression: the
forecast says what the user is likely to bank, and `clamp_contribution` decides
how much of that is *safe to commit* to a goal given existing goal load, income
instability, and the amount still outstanding. That separation matters - a model
should not be asked to encode "don't starve the user for their emergency fund".
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mercy_ml.config import get_settings
from mercy_ml.models.base import BaseMLModel, confidence_from_score, safe_float
from mercy_ml.schema import (
    SAVINGS_CATEGORICAL_FEATURES,
    SAVINGS_FEATURE_COLUMNS,
    SAVINGS_NUMERIC_FEATURES,
    SAVINGS_TARGET,
)

logger = logging.getLogger(__name__)

#: Share of forecast net savings a user is realistically willing to lock in.
DEFAULT_ALLOCATION_SHARE = 0.60
#: Cap on the share of forecast net savings, so a good month is not spent twice.
MAX_ALLOCATION_SHARE = 0.80
#: Extra goals in progress reduce the willingness to fund another one.
GOAL_LOAD_PENALTY = 0.12
MAX_GOAL_LOAD_PENALTY = 0.40
#: When income is unstable (cv above this), keep a wider safety margin.
UNSTABLE_INCOME_CV = 0.25
MAX_INSTABILITY_PENALTY = 0.30
#: Below this a suggested contribution is not worth presenting.
MIN_USEFUL_CONTRIBUTION = 5.0
#: Emergency-fund style goals are funded at a steadier pace than a holiday.
ESSENTIAL_GOAL_ALLOCATION_SHARE = 0.40


class SavingsRecommender(BaseMLModel):
    name = "savings_recommender"
    numeric_features = SAVINGS_NUMERIC_FEATURES
    categorical_features = SAVINGS_CATEGORICAL_FEATURES
    target = SAVINGS_TARGET
    time_ordered = True
    min_samples = 80

    notes = (
        "Target is next-period net savings (income - expense), not a wish amount.",
        "Chronological split; allocation to a goal is business logic, not learned.",
    )

    def make_estimator(self) -> Any:
        kind = str(self.params.get("estimator", "gradient_boosting"))
        if kind == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.params.get("n_estimators", 200),
                max_depth=self.params.get("max_depth", 8),
                min_samples_leaf=self.params.get("min_samples_leaf", 4),
                n_jobs=self.params.get("n_jobs", -1),
                random_state=get_settings().random_seed,
            )
        return GradientBoostingRegressor(
            n_estimators=self.params.get("n_estimators", 250),
            max_depth=self.params.get("max_depth", 3),
            learning_rate=self.params.get("learning_rate", 0.05),
            loss=self.params.get("loss", "huber"),
            random_state=get_settings().random_seed,
        )

    def evaluate(self, x_test, y_test, x_train, y_train) -> dict[str, Any]:
        y_pred = self.predict(x_test)
        y_true = np.asarray(y_test, dtype="float64")
        errors = y_pred - y_true

        metrics: dict[str, Any] = {
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "mse": round(float(mean_squared_error(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4) if len(y_true) > 1 else None,
            "median_abs_error": round(float(np.median(np.abs(errors))), 4),
            "mape": self._mape(y_true, y_pred),
            "mean_true_savings": round(float(y_true.mean()), 4),
            "mean_predicted_savings": round(float(y_pred.mean()), 4),
            "bias": round(float(errors.mean()), 4),
            "overestimate_rate": round(float((errors > 0).mean()), 4),
        }
        if "period_net_savings" in x_test:
            baseline = x_test["period_net_savings"].to_numpy(dtype="float64")
            metrics["naive_last_period_mae"] = round(float(mean_absolute_error(y_true, baseline)), 4)
            metrics["mae_skill_vs_naive"] = (
                round(1.0 - metrics["mae"] / metrics["naive_last_period_mae"], 4)
                if metrics["naive_last_period_mae"] > 0
                else None
            )
        importances = self.feature_importances()
        if importances:
            metrics["top_features"] = importances[:12]
        return metrics

    @staticmethod
    def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
        mask = y_true > 1e-6
        if not mask.any():
            return None
        return round(float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0), 2)

    def feature_importances(self) -> list[dict[str, Any]]:
        if not hasattr(self.pipeline, "feature_importances_"):
            return []
        names = list(getattr(self.preprocessor, "get_feature_names_out", lambda: [])())
        values = np.asarray(self.pipeline.feature_importances_, dtype="float64")
        if len(names) != len(values):
            return []
        return [
            {"feature": name, "importance": round(float(value), 5)}
            for value, name in sorted(zip(values, names), reverse=True)
        ]

    # --------------------------------------------------------- inference
    def forecast_next_savings(self, row: pd.DataFrame) -> float:
        """Predicted net savings for the user's next period. May be negative."""
        if not self.is_trained:
            raise RuntimeError("savings recommender is not trained")
        return safe_float(self.predict(row)[0])

    def recommend(
        self,
        row: pd.DataFrame,
        *,
        target_amount: float = 0.0,
        current_amount: float = 0.0,
        active_goals: int = 0,
        goal_is_essential: bool = False,
        months_remaining: float | None = None,
    ) -> dict[str, Any]:
        """Turn a forecast into a concrete per-period contribution.

        `row` is the user's current-period feature row from
        `features.build_savings_inference_row`; the goal fields are business
        context that is not part of the feature vector.
        """
        if not self.is_trained:
            raise RuntimeError("savings recommender is not trained")

        record = row.iloc[0]
        predicted = self.forecast_next_savings(row)
        income = safe_float(record.get("period_income"), 0.0)
        expense = safe_float(record.get("period_expense"), 0.0)
        savings_rate = safe_float(record.get("savings_rate"), 0.0)
        income_volatility = safe_float(record.get("income_volatility"), 0.0)
        tenure = safe_float(record.get("user_tenure_periods"), 0.0)

        income_cv = income_volatility / income if income > 0 else 1.0
        contribution = clamp_contribution(
            predicted,
            income=income,
            income_cv=income_cv,
            active_goals=active_goals,
            goal_is_essential=goal_is_essential,
        )

        gap = max(0.0, target_amount - current_amount)
        months_needed = (gap / contribution) if contribution >= MIN_USEFUL_CONTRIBUTION and gap > 0 else None
        if months_remaining is not None and months_remaining > 0 and months_needed is not None:
            verdict = "achievable" if months_needed <= months_remaining else "tight"
        elif contribution < MIN_USEFUL_CONTRIBUTION:
            verdict = "insufficient"
        else:
            verdict = "suggested"

        return {
            "predicted_next_net_savings": round(predicted, 2),
            "suggested_contribution": contribution,
            "verdict": verdict,
            "current_savings_rate_pct": round(savings_rate * 100.0, 2),
            "income_cv": round(income_cv, 4),
            "gap_to_target": round(gap, 2),
            "months_to_target": round(months_needed, 1) if months_needed else None,
            "months_remaining": months_remaining,
            "active_goals": active_goals,
            "message": _message(verdict, contribution, predicted, months_needed, months_remaining, gap),
            "confidence_score": confidence_from_score(
                _quality(tenure, income, income_cv, predicted), ceiling=0.85
            ),
        }


def clamp_contribution(
    predicted_net_savings: float,
    *,
    income: float,
    income_cv: float,
    active_goals: int = 0,
    goal_is_essential: bool = False,
) -> float:
    """How much of the forecast surplus is safe to commit to a goal.

    Never exceeds the forecast itself, shrinks for unstable income, and shrinks
    further as the user accumulates goals. An essential goal (emergency fund,
    debt payoff) is funded at a steadier, lower rate so it survives a bad month.
    """
    if predicted_net_savings <= 0:
        return 0.0

    share = ESSENTIAL_GOAL_ALLOCATION_SHARE if goal_is_essential else DEFAULT_ALLOCATION_SHARE
    value = predicted_net_savings * min(share, MAX_ALLOCATION_SHARE)

    if active_goals > 0:
        penalty = min(MAX_GOAL_LOAD_PENALTY, GOAL_LOAD_PENALTY * active_goals)
        value *= 1.0 - penalty

    if income_cv > UNSTABLE_INCOME_CV:
        penalty = min(
            MAX_INSTABILITY_PENALTY, (income_cv - UNSTABLE_INCOME_CV) * 0.5
        )
        value *= 1.0 - penalty

    # Never commit more than half of income in a single period.
    if income > 0:
        value = min(value, income * 0.50)

    return round(max(0.0, value), 2)


def _message(
    verdict: str,
    contribution: float,
    predicted: float,
    months_needed: float | None,
    months_remaining: float | None,
    gap: float,
) -> str:
    if verdict == "insufficient":
        if predicted <= 0:
            return (
                "Based on recent cash flow this period does not produce a surplus to direct "
                "into savings. Tightening discretionary budgets would create room."
            )
        return (
            f"Only {contribution:,.0f} per period is safely available right now - not enough to "
            "move a goal forward meaningfully."
        )
    if verdict == "achievable":
        return (
            f"Contributing {contribution:,.0f} per period reaches the target in about "
            f"{months_needed:.1f} months, inside the {months_remaining:.0f} month(s) remaining."
        )
    if verdict == "tight":
        return (
            f"Contributing {contribution:,.0f} per period reaches the target in about "
            f"{months_needed:.1f} months - slower than the {months_remaining:.0f} month(s) "
            "you had in mind. Either extend the deadline or reduce the target."
        )
    return f"Adding {contribution:,.0f} per period to this goal fits your recent cash flow."


def _quality(tenure: float, income: float, income_cv: float, predicted: float) -> float:
    depth = min(1.0, tenure / 6.0)
    stability = 1.0 / (1.0 + min(2.0, income_cv))
    signal = 1.0 if income > 0 else 0.0
    magnitude = 1.0 if predicted > 0 else 0.35
    return float(np.clip(0.30 * depth + 0.30 * stability + 0.20 * signal + 0.20 * magnitude, 0.0, 1.0))


__all__ = ["SavingsRecommender", "SAVINGS_FEATURE_COLUMNS", "clamp_contribution"]
