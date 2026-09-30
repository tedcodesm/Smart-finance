"""(c) Overspend prediction - binary classifier.

Pooled across every user: one model learns "given how fast money is leaving this
budget, how much of it is left, and how this user has behaved historically, is
the budget going to break?" Per-user scale is removed by expressing features as
*rates and ratios* (utilisation %, spend velocity, overspend rate) rather than raw
currency amounts - `limit_amount` and `spent_so_far` are kept, but the tree
ensemble can split on their ratio rather than memorising an absolute threshold.

Random Forest is the default because it handles the mixed scale, correlated
features and missing values in the panel data well, and it exposes feature
importances that make the recommendation explainable. Logistic Regression is
available as a calibrated, lower-variance alternative and is scored on the same
split so the two can be compared directly.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mercy_ml.models.base import BaseMLModel, band_label, confidence_from_score
from mercy_ml.schema import (
    OVERSPEND_CATEGORICAL_FEATURES,
    OVERSPEND_FEATURE_COLUMNS,
    OVERSPEND_NUMERIC_FEATURES,
    OVERSPEND_TARGET,
    RISK_BANDS,
)

logger = logging.getLogger(__name__)


class OverspendClassifier(BaseMLModel):
    name = "overspend_classifier"
    numeric_features = OVERSPEND_NUMERIC_FEATURES
    categorical_features = OVERSPEND_CATEGORICAL_FEATURES
    target = OVERSPEND_TARGET
    stratify = True
    min_samples = 60

    notes = (
        "Pooled model: trained on all users at once, scored with per-user rate features.",
        "80/20 stratified hold-out; class_weight='balanced' to absorb class imbalance.",
    )

    def make_estimator(self) -> Any:
        kind = str(self.params.get("estimator", "random_forest"))
        if kind == "logistic":
            return LogisticRegression(
                max_iter=self.params.get("max_iter", 2000),
                class_weight="balanced",
                C=self.params.get("C", 1.0),
                random_state=get_settings_seed(),
            )
        return RandomForestClassifier(
            n_estimators=self.params.get("n_estimators", 300),
            max_depth=self.params.get("max_depth", 12),
            min_samples_leaf=self.params.get("min_samples_leaf", 5),
            max_features=self.params.get("max_features", "sqrt"),
            class_weight="balanced",
            n_jobs=self.params.get("n_jobs", -1),
            random_state=get_settings_seed(),
        )

    # ------------------------------------------------------------ metrics
    def evaluate(self, x_test, y_test, x_train, y_train) -> dict[str, Any]:
        y_pred = self.predict(x_test)
        y_proba = self.predict_proba(x_test)
        positive_index = 1 if self.classes_ is not None and 1 in list(self.classes_) else 0
        scores = y_proba[:, positive_index] if y_proba.ndim == 2 else y_proba.ravel()

        y_true = np.asarray(y_test)
        y_true_binary = (y_true == 1).astype(int)
        y_pred_binary = (np.asarray(y_pred) == 1).astype(int)

        train_scores = self.predict_proba(x_train)
        train_prob = train_scores[:, positive_index] if train_scores.ndim == 2 else train_scores.ravel()

        metrics: dict[str, Any] = {
            "accuracy": round(float(accuracy_score(y_true_binary, y_pred_binary)), 4),
            "precision": round(float(precision_score(y_true_binary, y_pred_binary, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true_binary, y_pred_binary, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true_binary, y_pred_binary, zero_division=0)), 4),
            "positive_rate_test": round(float(y_true_binary.mean()), 4),
            "positive_rate_train": round(float((np.asarray(y_train) == 1).mean()), 4),
            "predicted_positive_rate": round(float(y_pred_binary.mean()), 4),
            "confusion_matrix": confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).tolist(),
        }
        if len(np.unique(y_true_binary)) > 1:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true_binary, scores)), 4)
            metrics["average_precision"] = round(float(average_precision_score(y_true_binary, scores)), 4)
        else:
            # Degenerate test split: fall back to train-set separation so the
            # report is not silently empty.
            metrics["roc_auc"] = (
                round(float(roc_auc_score((np.asarray(y_train) == 1).astype(int), train_prob)), 4)
                if len(np.unique(np.asarray(y_train))) > 1
                else None
            )
        metrics["classification_report"] = classification_report(
            y_true_binary, y_pred_binary, zero_division=0, output_dict=True
        )
        importances = self.feature_importances()
        if importances:
            metrics["top_features"] = importances[:12]
        return metrics

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
    def predict_risk(self, row: pd.DataFrame) -> dict[str, Any]:
        """Probability that a single budget overruns, plus a banded label."""
        if not self.is_trained:
            raise RuntimeError("overspend classifier is not trained")
        proba = self.predict_proba(row)
        positive_index = 1 if self.classes_ is not None and 1 in list(self.classes_) else 0
        probability = float(proba[0, positive_index])
        return {
            "probability": round(probability, 4),
            "label": band_label(probability, RISK_BANDS),
            "confidence_score": confidence_from_score(
                1.0 - abs(probability - 0.5) * 2.0, ceiling=0.9
            ),
            "threshold": 0.5,
        }

    def explain(self, row: pd.DataFrame, *, top: int = 5) -> list[dict[str, Any]]:
        """Which inputs pushed this particular prediction up or down."""
        if not self.is_trained or not hasattr(self.pipeline, "feature_importances_"):
            return []
        names = list(getattr(self.preprocessor, "get_feature_names_out", lambda: [])())
        importances = np.asarray(self.pipeline.feature_importances_, dtype="float64")
        values = self._transform(row)[0]
        lookup = {name: float(values[i]) for i, name in enumerate(names)}

        contributions: list[dict[str, Any]] = []
        for importance, name in sorted(zip(importances, names), reverse=True)[:top]:
            contributions.append(
                {
                    "feature": name,
                    "value": round(lookup.get(name, 0.0), 4),
                    "importance": round(float(importance), 5),
                }
            )
        return contributions


def get_settings_seed() -> int:
    from mercy_ml.config import get_settings

    return get_settings().random_seed


def compare_estimators(
    frame: pd.DataFrame, *, kinds: tuple[str, ...] = ("random_forest", "logistic")
) -> dict[str, dict[str, Any]]:
    """Train both candidate families on the same split and return their metrics.

    Used by the training script so the choice between them is evidence-based
    rather than hard-coded.
    """
    results: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        model = OverspendClassifier(estimator=kind)
        model.fit(frame)
        if model.report and model.report.metrics:
            results[kind] = dict(model.report.metrics)
    return results
