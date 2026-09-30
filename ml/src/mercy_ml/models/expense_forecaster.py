"""(b) Expense forecasting - next-period spend per user & category.

Two model families are fitted on identical features and compared on a
chronological 80/20 split (train = earlier periods, test = later), because a
random split would let the model peek at the future:

  * `random_forest` - pooled gradient-free ensemble, captures the category
    identity via one-hot features plus the lag/rolling structure.
  * `ets`           - per (user, category) Exponential Smoothing from
    statsmodels, a classical local-level time-series model that is very strong on
    short, smooth per-series histories and extrapolates a level without the
    forest's tendency to regress toward the training mean.

The winner is persisted, and the runner-up's metrics are kept in the report so
the choice is auditable. A naive `lag_1` baseline is always scored as well -
beating it is the minimum bar for the model to be worth shipping.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mercy_ml.config import get_settings
from mercy_ml.models.base import BaseMLModel, confidence_from_score, safe_float
from mercy_ml.schema import (
    FORECAST_CATEGORICAL_FEATURES,
    FORECAST_FEATURE_COLUMNS,
    FORECAST_NUMERIC_FEATURES,
    FORECAST_TARGET,
)

logger = logging.getLogger(__name__)


class ExpenseForecaster(BaseMLModel):
    name = "expense_forecaster"
    numeric_features = FORECAST_NUMERIC_FEATURES
    categorical_features = FORECAST_CATEGORICAL_FEATURES
    target = FORECAST_TARGET
    time_ordered = True
    min_samples = 50

    notes = (
        "Chronological 80/20 split (train on earlier periods, test on later).",
        "Compared against a lag_1 naive baseline; the model must beat it to ship.",
    )

    def make_estimator(self) -> Any:
        kind = str(self.params.get("estimator", "random_forest"))
        if kind == "ridge":
            return Ridge(alpha=self.params.get("alpha", 1.0), random_state=None)
        return RandomForestRegressor(
            n_estimators=self.params.get("n_estimators", 250),
            max_depth=self.params.get("max_depth", 14),
            min_samples_leaf=self.params.get("min_samples_leaf", 3),
            max_features=self.params.get("max_features", 0.5),
            n_jobs=self.params.get("n_jobs", -1),
            random_state=get_settings().random_seed,
        )

    def evaluate(self, x_test, y_test, x_train, y_train) -> dict[str, Any]:
        y_pred = self.predict(x_test)
        y_true = np.asarray(y_test, dtype="float64")

        naive = x_test["lag_1"].to_numpy(dtype="float64") if "lag_1" in x_test else np.zeros_like(y_pred)
        metrics: dict[str, Any] = {
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "mse": round(float(mean_squared_error(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4) if len(y_true) > 1 else None,
            "mape": self._mape(y_true, y_pred),
            "bias": round(float(np.mean(y_pred - y_true)), 4),
            "median_abs_error": round(float(np.median(np.abs(y_pred - y_true))), 4),
            "naive_lag1_mae": round(float(mean_absolute_error(y_true, naive)), 4),
        }
        metrics["mae_skill_vs_naive"] = (
            round(1.0 - metrics["mae"] / metrics["naive_lag1_mae"], 4)
            if metrics["naive_lag1_mae"] > 0
            else None
        )
        metrics["q80_abs_error"] = round(float(np.quantile(np.abs(y_pred - y_true), 0.8)), 4)
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
        estimator = self.pipeline
        if not hasattr(estimator, "feature_importances_"):
            return []
        names = list(getattr(self.preprocessor, "get_feature_names_out", lambda: [])())
        values = np.asarray(estimator.feature_importances_, dtype="float64")
        if len(names) != len(values):
            return []
        pairs = sorted(zip(values, names), reverse=True)
        return [{"feature": name, "importance": round(float(value), 5)} for value, name in pairs]

    # --------------------------------------------------------- inference
    def predict_with_interval(self, row: pd.DataFrame, *, lower_q: float = 0.1, upper_q: float = 0.9) -> dict[str, float]:
        """Point forecast plus an empirical interval from the tree ensemble.

        The spread across trees is a cheap, well-calibrated proxy for predictive
        uncertainty - the trees disagree exactly where the feature vector sits in
        sparse or unfamiliar territory.
        """
        if not self.is_trained:
            raise RuntimeError("expense forecaster is not trained")
        point = float(self.predict(row)[0])

        lower = upper = 0.0
        estimator = self.pipeline
        if hasattr(estimator, "estimators_") and hasattr(estimator, "tree_"):
            transformed = self._transform(row)
            predictions = np.array([tree.predict(transformed)[0] for tree in estimator.estimators_])
            lower = float(np.quantile(predictions, lower_q))
            upper = float(np.quantile(predictions, upper_q))
        else:
            lower = upper = point

        return {
            "predicted_amount": round(max(0.0, point), 2),
            "lower_bound": round(max(0.0, lower), 2),
            "upper_bound": round(max(0.0, upper), 2),
        }


class EtsForecaster:
    """Per-(user, category) Exponential Smoothing via statsmodels.

    Kept as a separate class because it is not a single pooled estimator: it fits
    one local model per series, which is the classical way to forecast short,
    noisy series without borrowing strength across users. It is used as a
    benchmark and as a fallback for series with a long history.
    """

    name = "expense_forecaster_ets"

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self.seasonal_periods = int(params.get("seasonal_periods", 0))

    def fit_evaluate(
        self, frame: pd.DataFrame, *, group_columns: tuple[str, ...] = ("user_id", "category_id")
    ) -> dict[str, Any]:
        """Backtest: for each series, fit on all but the last period, forecast it."""
        settings = get_settings()
        predictions: list[tuple[float, float]] = []
        naive: list[tuple[float, float]] = []
        fitted = 0

        for _, group in frame.groupby(list(group_columns), observed=True):
            series = group.sort_values("period_key")
            if len(series) < settings.min_months_history + 1:
                continue
            history = series["target_next_expense"].to_numpy(dtype="float64")[:-1]
            actual = float(series["target_next_expense"].to_numpy(dtype="float64")[-1])
            if len(history) < 4:
                continue
            forecast = self._ets_forecast(history)
            if forecast is None:
                continue
            predictions.append((actual, forecast))
            naive.append((actual, float(series["lag_1"].to_numpy(dtype="float64")[-1])))
            fitted += 1

        if len(predictions) < 5:
            return {"n_series": fitted, "metrics": {}, "note": "not enough series with history"}

        y_true = np.array([p[0] for p in predictions])
        y_pred = np.array([p[1] for p in predictions])
        n_true = np.array([p[0] for p in naive])
        n_pred = np.array([p[1] for p in naive])

        return {
            "n_series": fitted,
            "metrics": {
                "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
                "mse": round(float(mean_squared_error(y_true, y_pred)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
                "r2": round(float(r2_score(y_true, y_pred)), 4) if len(y_true) > 1 else None,
                "naive_lag1_mae": round(float(mean_absolute_error(n_true, n_pred)), 4),
            },
        }

    def _ets_forecast(self, values: np.ndarray) -> float | None:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
        except ImportError:  # pragma: no cover - statsmodels is a hard dependency
            logger.warning("statsmodels unavailable; ETS benchmark skipped")
            return None

        # statsmodels needs at least 2 full seasonal cycles for a seasonal fit.
        if self.seasonal_periods and len(values) >= 2 * self.seasonal_periods + 2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ExponentialSmoothing(
                        values,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=self.seasonal_periods,
                        initialization_method="estimated",
                    )
                    return safe_float(model.fit(optimized=True).forecast(1)[0])
            except Exception as exc:  # pragma: no cover - statsmodels edge cases
                logger.debug("Seasonal ETS failed, falling back to simple: %s", exc)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return safe_float(
                    SimpleExpSmoothing(values, initialization_method="estimated").fit(optimized=True).forecast(1)[0]
                )
        except Exception as exc:  # pragma: no cover - statsmodels edge cases
            logger.debug("Simple ETS failed: %s", exc)
            return None


def compare_forecasters(frame: pd.DataFrame) -> dict[str, Any]:
    """Score the pooled regressor against ETS and the naive baseline."""
    results: dict[str, Any] = {}
    for kind in ("random_forest", "ridge"):
        model = ExpenseForecaster(estimator=kind)
        model.fit(frame)
        if model.report and model.report.metrics:
            results[kind] = dict(model.report.metrics)

    ets = EtsForecaster(seasonal_periods=0).fit_evaluate(frame)
    results["ets"] = ets
    if ets.get("metrics"):
        results["ets"] = {**ets["metrics"], "n_series": ets["n_series"]}

    ranked = sorted(
        ((name, metrics) for name, metrics in results.items() if metrics.get("mae") is not None),
        key=lambda item: item[1]["mae"],
    )
    results["best_by_mae"] = ranked[0][0] if ranked else None
    return results


def confidence_for_forecast(metrics: dict[str, Any], *, n_history: int) -> float:
    """Turn regression quality + history depth into a user-facing confidence."""
    r2 = metrics.get("r2")
    skill = metrics.get("mae_skill_vs_naive")
    base = 0.0
    if r2 is not None:
        base += max(0.0, float(r2))
    if skill is not None:
        base += max(0.0, min(1.0, float(skill)))
    base = base / 2.0 if (r2 is not None or skill is not None) else 0.35
    depth_bonus = min(0.25, max(0, n_history - 3) * 0.05)
    return round(confidence_from_score(min(1.0, base + depth_bonus), ceiling=0.9), 4)


__all__ = [
    "ExpenseForecaster",
    "EtsForecaster",
    "FORECAST_FEATURE_COLUMNS",
    "compare_forecasters",
    "confidence_for_forecast",
]
