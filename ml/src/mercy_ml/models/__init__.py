"""Statistical and machine-learning models for the finance manager.

The five models map one-to-one onto the business questions the product asks:

    (a) SpendingClusterer   - what kind of spender is this person?
    (b) ExpenseForecaster    - how much will they spend next period?
    (c) OverspendClassifier - will this budget break?
    (d) BudgetRecommender   - what should this limit be?
    (e) SavingsRecommender  - how much can go into this goal per period?

Every model is *pooled* - trained once across all users - so that a new user
with a handful of transactions immediately inherits a prediction. The
user-relative features in `mercy_ml.features` are what make that possible.
"""

from __future__ import annotations

from mercy_ml.models.base import (
    BaseMLModel,
    TrainingReport,
    band_label,
    confidence_from_score,
    safe_float,
)
from mercy_ml.models.budget_recommender import BudgetRecommender
from mercy_ml.models.expense_forecaster import (
    EtsForecaster,
    ExpenseForecaster,
    compare_forecasters,
    confidence_for_forecast,
)
from mercy_ml.models.overspend_classifier import OverspendClassifier, compare_estimators
from mercy_ml.models.savings_recommender import SavingsRecommender
from mercy_ml.models.spending_clusterer import SpendingClusterer

# Register artifact loaders so `registry.load_model(name)` can round-trip.
# BudgetRecommender is closed-form and never persisted, so it is excluded.
from mercy_ml.registry import register as _register

for _model_type in (
    ExpenseForecaster,
    OverspendClassifier,
    SavingsRecommender,
    SpendingClusterer,
):
    _register(_model_type.name)(_model_type)

__all__ = [
    "BaseMLModel",
    "BudgetRecommender",
    "EtsForecaster",
    "ExpenseForecaster",
    "OverspendClassifier",
    "SavingsRecommender",
    "SpendingClusterer",
    "TrainingReport",
    "band_label",
    "compare_estimators",
    "compare_forecasters",
    "confidence_for_forecast",
    "confidence_from_score",
    "safe_float",
]
