"""Preprocessing: cleaning, outlier handling, imputation, encoding.

The pipeline is split into
  * `clean_transactions`   - row-level hygiene on the raw transaction frame
  * `OutlierHandler`       - IQR winsorisation fitted on train data only
  * `build_preprocessor`   - sklearn ColumnTransformer for numeric + categorical
and the fitted artefacts are persisted with each model so inference reuses the
exact same imputation statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from mercy_ml.config import get_settings

logger = logging.getLogger(__name__)

#: Values above this are almost certainly data-entry errors, not real spending.
HARD_AMOUNT_CEILING = 1_000_000.0


def clean_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Row-level hygiene applied before any feature is derived.

    - drops rows with no user, no positive amount, or an unparseable date
    - de-duplicates on (user, date, category, amount, type)
    - clips physically impossible amounts
    - recomputes the denormalised time buckets from `date` (source of truth)
    """
    if frame.empty:
        return frame

    data = frame.copy()

    if "date" in data:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data[data["date"].notna()]
    data = data[data["user_id"].notna()]
    if "type" in data:
        data = data[data["type"].isin(["income", "expense"])]
    if "amount" in data:
        data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
        data = data[data["amount"].notna() & (data["amount"] > 0)]
        data["amount"] = data["amount"].clip(upper=HARD_AMOUNT_CEILING)

    subset = [c for c in ("user_id", "date", "category_id", "amount", "type") if c in data]
    if subset:
        before = len(data)
        data = data.drop_duplicates(subset=subset, keep="first")
        if before != len(data):
            logger.info("Removed %d duplicate transaction(s)", before - len(data))

    data = derive_time_features(data)
    return data.reset_index(drop=True)


def derive_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """(Re)derive every denormalised calendar field from `date`."""
    if frame.empty or "date" not in frame:
        return frame

    data = frame.copy()
    date = data["date"]
    data["day_of_week"] = date.dt.weekday.astype("int64")
    data["day_of_month"] = date.dt.day.astype("int64")
    data["month"] = date.dt.month.astype("int64")
    data["year"] = date.dt.year.astype("int64")
    data["week_of_year"] = date.dt.isocalendar().week.astype("int64").fillna(1).astype("int64")
    data["period_key"] = date.dt.strftime("%Y-%m")
    iso = date.dt.isocalendar()
    data["week_key"] = (
        iso["year"].astype("Int64").astype(str) + "-W" + iso["week"].astype("Int64").astype(str).str.zfill(2)
    )
    if "is_recurring" not in data:
        data["is_recurring"] = 0.0
    data["is_recurring"] = pd.to_numeric(data["is_recurring"], errors="coerce").fillna(0.0).astype(float)
    return data


def add_cyclical(frame: pd.DataFrame, column: str, period: float, prefix: str | None = None) -> pd.DataFrame:
    """Encode a cyclical integer as sin/cos so the model sees continuity."""
    prefix = prefix or column
    data = frame.copy()
    angle = 2.0 * np.pi * data[column].fillna(0).astype(float) / period
    data[f"{prefix}_sin"] = np.sin(angle)
    data[f"{prefix}_cos"] = np.cos(angle)
    return data


@dataclass
class OutlierHandler:
    """IQR-based winsorisation, fitted on the training split only.

    Fitted bounds are stored so the same clipping is applied at inference time;
    re-deriving them from live data would make scores non-reproducible.
    """

    k: float = field(default_factory=lambda: get_settings().outlier_iqr_k)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    fitted: bool = False

    def fit(self, frame: pd.DataFrame, columns: list[str]) -> "OutlierHandler":
        self.bounds = {}
        for column in columns:
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                continue
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            iqr = q3 - q1
            if iqr <= 0:
                # Degenerate spread (e.g. a constant column): leave untouched.
                continue
            low = q1 - self.k * iqr
            high = q3 + self.k * iqr
            self.bounds[column] = (float(low), float(high))
        self.fitted = bool(self.bounds)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            return frame
        data = frame.copy()
        for column, (low, high) in self.bounds.items():
            if column in data:
                data[column] = pd.to_numeric(data[column], errors="coerce").clip(lower=low, upper=high)
        return data

    def fit_transform(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return self.fit(frame, columns).transform(frame)

    def to_dict(self) -> dict[str, Any]:
        return {"k": self.k, "bounds": {k: list(v) for k, v in self.bounds.items()}, "fitted": self.fitted}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "OutlierHandler":
        if not payload:
            return cls()
        return cls(
            k=payload.get("k", get_settings().outlier_iqr_k),
            bounds={k: (float(v[0]), float(v[1])) for k, v in payload.get("bounds", {}).items()},
            fitted=payload.get("fitted", False),
        )


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
    *,
    scale: bool = True,
) -> ColumnTransformer:
    """Median-impute + scale numerics; impute + one-hot categoricals.

    `handle_unknown="ignore"` matters here: a category created after training
    must not crash inference, it must simply contribute no signal.
    """
    categorical_features = categorical_features or []
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler() if scale else RobustScaler(quantile_range=(10, 90))),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, list(numeric_features)))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipeline, list(categorical_features)))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def safe_ratio(numerator: pd.Series | float, denominator: pd.Series | float, default: float = 0.0):
    """Element-wise division that yields `default` instead of inf/NaN."""
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        num = numerator if isinstance(numerator, pd.Series) else pd.Series(numerator)
        den = denominator if isinstance(denominator, pd.Series) else pd.Series(denominator, index=num.index)
        return num / den.replace(0, np.nan).fillna(default)
    return default if not denominator else numerator / denominator


def log1p_amount(amount: pd.Series) -> pd.Series:
    """Signed-safe log transform for a positive amount column."""
    return np.log1p(pd.to_numeric(amount, errors="coerce").clip(lower=0))


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std()
    if not std or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (values - values.mean()) / std


def coerce_frame_types(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ensure the listed columns exist (as NaN) and are numeric."""
    data = frame.copy()
    for column in columns:
        if column not in data:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float64")
    return data
