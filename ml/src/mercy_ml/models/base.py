"""Shared training/evaluation/persistence wrapper.

Every model follows the same contract:

    model.fit(frame)          -> trains, evaluates on a held-out split, stores metrics
    model.predict(frame)      -> raw model output
    model.save(path)          -> joblib bundle (estimator + preprocessor + metadata)
    Model.load(path)          -> ready-to-score instance

The persisted bundle always carries the fitted preprocessor and outlier bounds so
inference can never drift from training.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mercy_ml.config import as_jsonable, get_settings, model_version
from mercy_ml.preprocess import OutlierHandler, build_preprocessor

logger = logging.getLogger(__name__)


@dataclass
class TrainingReport:
    """Outcome of a single `fit` call, persisted alongside the artifact."""

    model_name: str
    model_version: str
    trained_at: str
    n_samples: int
    n_train: int
    n_test: int
    n_features: int
    feature_names: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    target_stats: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: Population used at inference time; the backend surfaces this so the UI can
    #: say "based on N users".
    n_users: int = 0

    def to_dict(self) -> dict[str, Any]:
        return as_jsonable(asdict(self))


class BaseMLModel:
    """Base class handling the train/evaluate/persist boilerplate."""

    name: str = "base"
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    target: str | None = None
    #: Minimum rows required before a model will report metrics.
    min_samples: int = 40

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        self.pipeline: Any = None
        self.preprocessor: Any = None
        self.outliers = OutlierHandler()
        self.report: TrainingReport | None = None
        self.feature_names: list[str] = []
        self.classes_: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, frame: pd.DataFrame, *, user_column: str = "user_id") -> "BaseMLModel":
        if self.target is None:
            raise ValueError(f"{self.name} does not define a target column")

        data = self._prepare(frame, fit=True)
        if len(data) < self.min_samples:
            self._record_skipped(data, reason=f"only {len(data)} sample(s), need {self.min_samples}")
            return self

        x_train, x_test, y_train, y_test = self.split(data)
        self._build_and_fit(x_train, y_train)
        metrics = self.evaluate(x_test, y_test, x_train, y_train)
        self._record(data, x_train, x_test, y_train, y_test, metrics)
        return self

    def _prepare(self, frame: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        columns = self.all_features
        wanted = columns if self.target is None else columns + [self.target]
        missing = [c for c in wanted if c not in frame.columns]
        if missing:
            raise KeyError(f"{self.name}: frame is missing column(s) {missing}")

        keep = list(wanted)
        # Preserve the row identity so reports can count unique users without
        # letting the column leak into training.
        if "user_id" in frame.columns and "user_id" not in keep:
            keep.append("user_id")
        data = frame[keep].copy()
        if self.target is not None:
            data[self.target] = pd.to_numeric(data[self.target], errors="coerce")
            data = data[data[self.target].notna()]
        data = data.dropna(subset=[c for c in self.numeric_features if c in data], how="all")

        if fit:
            self.outliers.fit(data, self.numeric_features)
        data = self.outliers.transform(data)

        data = data.replace([np.inf, -np.inf], np.nan)
        self.feature_names = list(columns)
        return data.reset_index(drop=True)

    @property
    def all_features(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]

    def split(self, data: pd.DataFrame):
        """80/20 hold-out. Classifiers stratify; regressors use a time-ordered
        split when the frame carries a period column, so we never train on the
        future to predict the past."""
        settings = get_settings()
        test_size = 1.0 - settings.train_fraction

        stratify = None
        if self.stratify and data[self.target].nunique() > 1:
            counts = data[self.target].value_counts()
            if counts.min() >= 2:
                stratify = data[self.target]

        if self.time_ordered and "period_key" in data.columns:
            ordered = data.sort_values("period_key")
            cut = max(1, int(len(ordered) * settings.train_fraction))
            return ordered.iloc[:cut], ordered.iloc[cut:], ordered[self.target].iloc[:cut], ordered[self.target].iloc[cut:]

        x_train, x_test, y_train, y_test = train_test_split(
            data[list(self.all_features)],
            data[self.target],
            test_size=test_size,
            random_state=settings.random_seed,
            stratify=stratify,
        )
        return x_train, x_test, y_train, y_test

    #: Classifiers set this to request stratified splitting.
    stratify: bool = False
    #: Regressors set this to request a chronological split.
    time_ordered: bool = False

    def _build_and_fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.preprocessor = build_preprocessor(self.numeric_features, self.categorical_features)
        estimator = self.make_estimator()
        transformed = self.preprocessor.fit_transform(x_train)
        estimator.fit(transformed, y_train)
        self.pipeline = estimator
        classes = getattr(estimator, "classes_", None)
        if classes is not None:
            self.classes_ = np.asarray(classes)

    def make_estimator(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    # -------------------------------------------------------------- predict
    def _transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name} has not been fitted")
        for column in self.all_features:
            if column not in frame:
                frame = frame.copy()
                frame[column] = np.nan
        data = self.outliers.transform(frame[list(self.all_features)])
        data = data.replace([np.inf, -np.inf], np.nan)
        return self.preprocessor.transform(data)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(self._transform(frame))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.pipeline, "predict_proba"):
            raise AttributeError(f"{self.name} does not expose predict_proba")
        return self.pipeline.predict_proba(self._transform(frame))

    # ----------------------------------------------------------- evaluation
    def evaluate(
        self, x_test: pd.DataFrame, y_test: pd.Series, x_train: pd.DataFrame, y_train: pd.Series
    ) -> dict[str, Any]:
        raise NotImplementedError

    def predict_one(self, row: pd.DataFrame) -> Any:  # pragma: no cover - overridden
        return self.predict(row)

    def _record(
        self,
        data: pd.DataFrame,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        metrics: dict[str, Any],
    ) -> None:
        target = data[self.target]
        self.report = TrainingReport(
            model_name=self.name,
            model_version=model_version(self.name),
            trained_at=datetime.now(timezone.utc).isoformat(),
            n_samples=len(data),
            n_train=len(x_train),
            n_test=len(x_test),
            n_features=len(self.all_features),
            feature_names=list(self.all_features),
            metrics=metrics,
            params=dict(self.params),
            target_stats={
                "mean": float(target.mean()),
                "std": float(target.std()) if len(target) > 1 else 0.0,
                "min": float(target.min()),
                "max": float(target.max()),
                "positive_rate": float((target > 0).mean()),
            },
            notes=list(self.notes),
            n_users=int(data["user_id"].nunique()) if "user_id" in data else 0,
        )

    def _record_skipped(self, data: pd.DataFrame, *, reason: str) -> None:
        logger.warning("%s: skipping training - %s", self.name, reason)
        self.report = TrainingReport(
            model_name=self.name,
            model_version=model_version(self.name),
            trained_at=datetime.now(timezone.utc).isoformat(),
            n_samples=len(data),
            n_train=0,
            n_test=0,
            n_features=len(self.all_features),
            feature_names=list(self.all_features),
            metrics={},
            params=dict(self.params),
            notes=[reason],
        )

    notes: Sequence[str] = ()

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    # ----------------------------------------------------------- persistence
    def to_bundle(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": model_version(self.name),
            "estimator": self.pipeline,
            "preprocessor": self.preprocessor,
            "outliers": self.outliers.to_dict() if self.outliers is not None else None,
            "report": self.report.to_dict() if self.report else None,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "target": self.target,
            "params": self.params,
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
            "notes": list(self.notes),
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.to_bundle(), path, compress=3)
        metrics_path = path.with_name(path.stem + "_metrics.json")
        metrics_path.write_text(
            json.dumps(self.report.to_dict() if self.report else {}, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Saved %s -> %s", self.name, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "BaseMLModel":
        bundle = joblib.load(path)
        instance = cls(**bundle.get("params", {}))
        instance.pipeline = bundle["estimator"]
        instance.preprocessor = bundle["preprocessor"]
        instance.outliers = OutlierHandler.from_dict(bundle.get("outliers"))
        instance.numeric_features = bundle.get("numeric_features", instance.numeric_features)
        instance.categorical_features = bundle.get("categorical_features", instance.categorical_features)
        instance.target = bundle.get("target", instance.target)
        instance.notes = tuple(bundle.get("notes", ()))
        classes = bundle.get("classes")
        instance.classes_ = np.asarray(classes) if classes is not None else None
        report = bundle.get("report")
        if report:
            instance.report = TrainingReport(**{k: v for k, v in report.items() if k in TrainingReport.__annotations__})
        return instance


def confidence_from_score(score: float, *, ceiling: float = 0.95, floor: float = 0.05) -> float:
    """Map a [0, 1] quality signal onto a [floor, ceiling] confidence band."""
    value = float(np.clip(score, 0.0, 1.0))
    return float(np.clip(floor + value * (ceiling - floor), floor, ceiling))


def band_label(probability: float, bands: Sequence[tuple[float, str]]) -> str:
    for threshold, label in bands:
        if probability >= threshold:
            return label
    return bands[-1][1] if bands else "low"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result
