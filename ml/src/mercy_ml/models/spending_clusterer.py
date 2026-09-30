"""(a) Spending pattern analysis - K-Means over transaction features.

One global clustering is fitted across all users' expenses. The user-relative
features built in `features.build_cluster_frame` (amount vs. the user's own
category mean, share of the user's period spend) are what allow a single set of
clusters to describe both a $4 coffee purchase and a $1,400 rent payment: the
model reasons about *behavioural shape*, not absolute currency.

K is selected by silhouette score over a sweep, and the number of clusters is
capped at 6 so every cluster stays large enough to be a meaningful "habit" rather
than a mathematical artefact.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from mercy_ml.config import get_settings
from mercy_ml.models.base import BaseMLModel, band_label
from mercy_ml.preprocess import OutlierHandler, build_preprocessor
from mercy_ml.schema import (
    CLUSTER_CATEGORICAL_FEATURES,
    CLUSTER_FEATURE_COLUMNS,
    CLUSTER_NUMERIC_FEATURES,
    CLUSTER_LABEL_KEYWORDS,
)

logger = logging.getLogger(__name__)


class SpendingClusterer(BaseMLModel):
    name = "spending_clusters"
    numeric_features = CLUSTER_NUMERIC_FEATURES
    categorical_features = CLUSTER_CATEGORICAL_FEATURES
    target = None
    min_samples = 200

    notes = (
        "Unsupervised: no train/test split - evaluated with silhouette + "
        "Calinski-Harabasz + Davies-Bouldin on the full feature matrix.",
        "K chosen by silhouette over 2..6; features standardised before fitting.",
    )

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.scaler = StandardScaler()
        self.centroid_frame: pd.DataFrame | None = None
        self.labels_: np.ndarray | None = None
        self.best_k: int | None = None
        self.silhouette: float | None = None
        self.search_report: dict[str, Any] = {}
        self.search_results: dict[int, float] = {}
        self._encoded_feature_names: list[str] = []

    # ------------------------------------------------------------------ fit
    def fit(self, frame: pd.DataFrame, *, user_column: str = "user_id") -> "SpendingClusterer":
        data = self._prepare(frame, fit=True)
        if len(data) < self.min_samples:
            self._record_skipped(data, reason=f"only {len(data)} sample(s), need {self.min_samples}")
            return self

        self.preprocessor = build_preprocessor(self.numeric_features, self.categorical_features)
        matrix = self.preprocessor.fit_transform(data[list(self.all_features)])
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        self._encoded_feature_names = list(getattr(self.preprocessor, "get_feature_names_out", lambda: [])())

        # Standardise the encoded matrix so no one-hot or wide-range feature
        # dominates the Euclidean distance K-Means minimises.
        self.scaler.fit(matrix)
        scaled = self.scaler.transform(matrix)

        best = self._select_k(scaled)
        if best is None:
            self._record_skipped(data, reason="no viable k produced a positive silhouette")
            return self

        self.best_k = best["k"]
        self.pipeline = best["model"]
        self.labels_ = best["model"].labels_
        self.silhouette = best["silhouette"]
        self.search_report = {name: round(float(sil), 4) for name, sil in self.search_results.items()}

        self.centroid_frame = self._describe_clusters(scaled)
        metrics = {
            "silhouette_score": round(float(self.silhouette), 4),
            "n_clusters": int(self.best_k),
            "inertia": round(float(best["model"].inertia_), 4),
            "calinski_harabasz": self._safe_metric(calinski_harabasz_score, scaled, self.labels_),
            "davies_bouldin": self._safe_metric(davies_bouldin_score, scaled, self.labels_),
            "silhouette_by_k": self.search_report,
            "cluster_sizes": self.cluster_sizes(),
        }
        self._record_unsupervised(data, metrics)
        return self

    def _select_k(self, matrix: np.ndarray) -> dict[str, Any] | None:
        settings = get_settings()
        upper = min(settings.kmeans_k_max, max(2, len(matrix) // 30))
        self.search_results = {}
        best: dict[str, Any] | None = None

        for k in range(settings.kmeans_k_min, upper + 1):
            model = KMeans(
                n_clusters=k,
                n_init=int(self.params.get("n_init", 10)),
                max_iter=int(self.params.get("max_iter", 300)),
                random_state=settings.random_seed,
            )
            labels = model.fit_predict(matrix)
            if len(np.unique(labels)) < 2:
                continue
            score = float(silhouette_score(matrix, labels, sample_size=min(4000, len(matrix)),
                                           random_state=settings.random_seed))
            self.search_results[k] = round(score, 4)
            logger.info("k=%d silhouette=%.4f", k, score)
            if best is None or score > best["silhouette"]:
                best = {"k": k, "model": model, "silhouette": score}

        if best is None or best["silhouette"] < settings.silhouette_floor:
            if best is not None:
                logger.warning(
                    "Best silhouette %.4f is below the %.2f floor - keeping it but flagging weak structure",
                    best["silhouette"],
                    settings.silhouette_floor,
                )
            return best
        return best

    def _safe_metric(self, fn, matrix: np.ndarray, labels: np.ndarray) -> float | None:
        try:
            return round(float(fn(matrix, labels)), 4)
        except Exception:  # pragma: no cover - metric guards
            return None

    def _record_unsupervised(self, data: pd.DataFrame, metrics: dict[str, Any]) -> None:
        from mercy_ml.models.base import TrainingReport
        from mercy_ml.config import model_version
        from datetime import datetime, timezone

        self.report = TrainingReport(
            model_name=self.name,
            model_version=model_version(self.name),
            trained_at=datetime.now(timezone.utc).isoformat(),
            n_samples=len(data),
            n_train=len(data),
            n_test=0,
            n_features=len(self.all_features),
            feature_names=list(self.all_features),
            metrics=metrics,
            params=dict(self.params),
            notes=list(self.notes),
            n_users=int(data["user_id"].nunique()) if "user_id" in data else 0,
        )

    # ------------------------------------------------------------- describe
    def _describe_clusters(self, scaled: np.ndarray) -> pd.DataFrame:
        """Human-readable profile of each centroid, used for the cluster label."""
        names = self._encoded_feature_names or list(self.all_features)
        records: list[dict[str, Any]] = []
        for cluster_id, centre in enumerate(self.pipeline.cluster_centers_):
            record = {"cluster_id": cluster_id}
            for position, name in enumerate(names[: len(centre)]):
                record[name] = float(centre[position])
            records.append(record)
        frame = pd.DataFrame(records).set_index("cluster_id")

        summary: list[dict[str, Any]] = []
        for cluster_id, row in frame.iterrows():
            amounts = row.get("log_amount", 0.0)
            weekend = row.get("is_weekend", 0.0)
            recurring = row.get("is_recurring", 0.0)
            label, description = _label_cluster(float(amounts), float(weekend), float(recurring))
            top_categories = (
                row.sort_values(ascending=False)
                .loc[lambda s: s.index.astype(str).str.startswith("category_name_")]
                .head(3)
                .index.astype(str)
                .str.replace("category_name_", "", regex=False)
                .tolist()
            )
            summary.append(
                {
                    "cluster_id": int(cluster_id),
                    "label": label,
                    "description": description,
                    "dominant_categories": top_categories,
                    "centroid": {k: round(float(v), 4) for k, v in row.items()},
                }
            )
        return pd.DataFrame(summary).set_index("cluster_id")

    def cluster_sizes(self) -> dict[str, int]:
        if self.labels_ is None:
            return {}
        values, counts = np.unique(self.labels_, return_counts=True)
        return {str(int(value)): int(count) for value, count in zip(values, counts)}

    # ----------------------------------------------------------- inference
    def assign(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("spending clusterer is not trained")
        matrix = self.preprocessor.transform(frame[list(self.all_features)])
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return self.pipeline.predict(self.scaler.transform(matrix))

    def describe(self, *, max_per_user: int = 20) -> list[dict[str, Any]]:
        """Cluster profiles with sizes, ready to be written to `predictions`."""
        if self.centroid_frame is None or self.pipeline is None:
            return []
        sizes = self.cluster_sizes()
        total = max(1, sum(sizes.values()))
        out: list[dict[str, Any]] = []
        for cluster_id, row in self.centroid_frame.iterrows():
            size = sizes.get(str(int(cluster_id)), 0)
            out.append(
                {
                    "cluster_id": int(cluster_id),
                    "label": row.get("label"),
                    "description": row.get("description"),
                    "dominant_categories": row.get("dominant_categories", []),
                    "size": size,
                    "share_pct": round(size / total * 100.0, 2),
                    "mean_log_amount": row.get("centroid", {}).get("log_amount"),
                }
            )
        return out

    def user_cluster_profile(self, frame: pd.DataFrame, user_id: Any, *, max_rows: int = 20) -> list[dict[str, Any]]:
        """Which clusters a given user's spending falls into."""
        subset = frame[frame["user_id"] == str(user_id)] if "user_id" in frame else frame
        if subset.empty:
            return []
        labels = self.assign(subset)
        profile: list[dict[str, Any]] = []
        for cluster_id in sorted(set(int(v) for v in labels)):
            mask = labels == cluster_id
            mean_amount: float | None
            if "amount" in subset:
                mean_amount = round(float(subset.loc[mask, "amount"].mean()), 2)
            elif "log_amount" in subset:
                mean_amount = round(float(np.expm1(subset.loc[mask, "log_amount"].mean())), 2)
            else:
                mean_amount = None
            profile.append(
                {
                    "cluster_id": cluster_id,
                    "count": int(mask.sum()),
                    "share_pct": round(float(mask.mean()) * 100.0, 2),
                    "mean_amount": mean_amount,
                    "label": self._label_for(cluster_id),
                }
            )
        profile.sort(key=lambda item: item["count"], reverse=True)
        for item in profile:
            if item["mean_amount"] is None:
                item["mean_amount"] = 0.0
        return profile[:max_rows]

    def _label_for(self, cluster_id: int) -> str:
        if self.centroid_frame is None or cluster_id not in self.centroid_frame.index:
            return "spending pattern"
        return str(self.centroid_frame.loc[cluster_id].get("label"))

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    # ----------------------------------------------------------- persistence
    # The base bundle covers estimator/preprocessor/outliers/report; clustering
    # adds the cluster-metadata that describe() and user_cluster_profile() need.
    def to_bundle(self) -> dict[str, Any]:
        bundle = super().to_bundle()
        bundle.update(
            {
                "scaler": self.scaler,
                "centroid_frame": (
                    self.centroid_frame.reset_index().to_dict("records")
                    if self.centroid_frame is not None else None
                ),
                "best_k": self.best_k,
                "silhouette": self.silhouette,
                "search_results": self.search_results,
                "encoded_feature_names": self._encoded_feature_names,
            }
        )
        return bundle

    @classmethod
    def load(cls, path: Any) -> "SpendingClusterer":
        from pathlib import Path

        import joblib as _joblib

        bundle = _joblib.load(Path(path))
        instance = cls(**bundle.get("params", {}))
        instance.pipeline = bundle["estimator"]
        instance.preprocessor = bundle["preprocessor"]
        instance.outliers = OutlierHandler.from_dict(bundle.get("outliers"))
        instance.numeric_features = bundle.get("numeric_features", instance.numeric_features)
        instance.categorical_features = bundle.get("categorical_features", instance.categorical_features)
        instance._encoded_feature_names = bundle.get("encoded_feature_names", [])
        instance.best_k = bundle.get("best_k")
        instance.silhouette = bundle.get("silhouette")
        instance.search_results = bundle.get("search_results", {})
        scaler = bundle.get("scaler")
        if scaler is not None:
            instance.scaler = scaler
        frame = bundle.get("centroid_frame")
        if frame:
            instance.centroid_frame = pd.DataFrame(frame).set_index("cluster_id")
        report = bundle.get("report")
        if report:
            from mercy_ml.models.base import TrainingReport

            instance.report = TrainingReport(
                **{k: v for k, v in report.items() if k in TrainingReport.__annotations__}
            )
        return instance


def _label_cluster(amount: float, weekend: float, recurring: float) -> tuple[str, str]:
    """Rule-based labelling from centroid shape - cheap and explainable."""
    if amount > 0.8 and weekend > 0.3:
        return "weekend / leisure", "Larger purchases clustered on Saturdays and Sundays."
    if amount > 0.8:
        return "large one-off purchase", "Big-ticket items, likely infrequent and irregular."
    if recurring > 0.6:
        return "monthly recurring bill", "Small, regular, highly predictable payments."
    if weekend > 0.3:
        return "weekday routine", "Modest purchases spread across the working week."
    return "everyday small spend", "Frequent low-value purchases - the core daily habit."


def silhouette_for(matrix: np.ndarray, labels: np.ndarray) -> float:
    try:
        return float(silhouette_score(matrix, labels))
    except Exception:  # pragma: no cover - degenerate label sets
        return float("nan")


__all__ = ["SpendingClusterer", "CLUSTER_FEATURE_COLUMNS", "silhouette_for", "band_label"]
