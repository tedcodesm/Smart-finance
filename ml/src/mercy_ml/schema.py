"""Feature contract shared by training and inference.

Keeping the column names in one module guarantees the exact same transformation
runs when a model is fitted offline and when it is scored for a live user - a
mismatch here is the classic source of silent inference bugs.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Panel identifiers
# --------------------------------------------------------------------------- #
USER = "user_id"
PERIOD = "period_key"
CATEGORY = "category_id"
CATEGORY_NAME = "category_name"
PERIOD_INDEX = "period_index"

# --------------------------------------------------------------------------- #
# (c) Overspend classification - one row per (user, budget scope, period)
# --------------------------------------------------------------------------- #
OVERSPEND_NUMERIC_FEATURES: list[str] = [
    "limit_amount",
    "spent_so_far",
    "period_elapsed_pct",
    "days_remaining",
    "spend_velocity",
    "projected_total",
    "utilisation_pct",
    "txn_count_so_far",
    "avg_txn_amount",
    "scope_avg_period_expense",
    "scope_expense_trend_pct",
    "user_avg_period_income",
    "user_avg_period_expense",
    "user_expense_income_ratio",
    "user_overspend_rate",
    "user_overspend_count",
    "user_periods_observed",
    "user_tenure_periods",
    "month_sin",
    "month_cos",
    "is_month_start",
    "is_month_end",
]
OVERSPEND_CATEGORICAL_FEATURES: list[str] = [CATEGORY_NAME]
OVERSPEND_TARGET = "label_overspend"
OVERSPEND_FEATURE_COLUMNS = OVERSPEND_NUMERIC_FEATURES + OVERSPEND_CATEGORICAL_FEATURES

# --------------------------------------------------------------------------- #
# (b) Expense forecasting - one row per (user, category, period) -> next period
# --------------------------------------------------------------------------- #
FORECAST_NUMERIC_FEATURES: list[str] = [
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_std_3",
    "rolling_max_3",
    "expense_trend_slope",
    "expense_trend_pct",
    "rel_to_user_period_expense",
    "period_income",
    "period_income_trend_pct",
    "txn_count",
    "avg_txn_amount",
    "expense_share_of_period",
    "month_sin",
    "month_cos",
    "is_month_start",
    "is_month_end",
    "user_tenure_periods",
    "user_periods_with_data",
]
FORECAST_CATEGORICAL_FEATURES: list[str] = [CATEGORY_NAME]
FORECAST_TARGET = "target_next_expense"
FORECAST_FEATURE_COLUMNS = FORECAST_NUMERIC_FEATURES + FORECAST_CATEGORICAL_FEATURES

# --------------------------------------------------------------------------- #
# (d) Savings recommendation - one row per (user, period)
# --------------------------------------------------------------------------- #
SAVINGS_NUMERIC_FEATURES: list[str] = [
    "period_income",
    "period_expense",
    "period_net_savings",
    "avg_income_3",
    "avg_expense_3",
    "income_trend_pct",
    "expense_trend_pct",
    "income_volatility",
    "expense_volatility",
    "savings_rate",
    "txn_count",
    "distinct_categories",
    "recurring_share",
    "user_tenure_periods",
    "month_sin",
    "month_cos",
]
SAVINGS_CATEGORICAL_FEATURES: list[str] = []
SAVINGS_TARGET = "target_next_net_savings"
SAVINGS_FEATURE_COLUMNS = SAVINGS_NUMERIC_FEATURES + SAVINGS_CATEGORICAL_FEATURES

# --------------------------------------------------------------------------- #
# (a) Spending-pattern clustering - one row per transaction
# --------------------------------------------------------------------------- #
CLUSTER_NUMERIC_FEATURES: list[str] = [
    "log_amount",
    "amount_zscore_in_category",
    "amount_ratio_in_category",
    "day_of_week",
    "is_weekend",
    "day_of_month",
    "is_month_start",
    "is_month_end",
    "week_of_year_sin",
    "week_of_year_cos",
    "month_sin",
    "month_cos",
    "txn_hour_bucket",
    "user_period_expense",
    "category_period_expense",
    "category_share_of_period",
    "is_recurring",
]
CLUSTER_CATEGORICAL_FEATURES: list[str] = [CATEGORY_NAME]
CLUSTER_FEATURE_COLUMNS = CLUSTER_NUMERIC_FEATURES + CLUSTER_CATEGORICAL_FEATURES

#: Human-readable cluster labels, chosen from the dominant features of each
#: centroid. Index positions are post-KMeans labelling order.
CLUSTER_LABEL_KEYWORDS: tuple[str, ...] = (
    "everyday small spend",
    "weekend / leisure",
    "large one-off purchase",
    "monthly recurring bill",
    "payday top-up",
    "impulse / discretionary",
)

#: Probability bands used to translate classifier output into UI labels.
RISK_BANDS: tuple[tuple[float, str], ...] = ((0.66, "high"), (0.33, "medium"), (0.0, "low"))


# --------------------------------------------------------------------------- #
# Contract registry
# --------------------------------------------------------------------------- #
class FeatureContract:
    """Declarative description of one model's input frame.

    Training and inference both validate through this object, so a column that
    is added, renamed, or dropped in `features.py` fails loudly in one place
    instead of silently producing wrong predictions.
    """

    __slots__ = ("name", "numeric", "categorical", "target", "unit")

    def __init__(
        self,
        name: str,
        numeric: list[str],
        categorical: list[str],
        target: str | None,
        unit: str = "money",
    ) -> None:
        self.name = name
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.target = target
        self.unit = unit

    @property
    def columns(self) -> list[str]:
        return [*self.numeric, *self.categorical]

    @property
    def all_columns(self) -> list[str]:
        return self.columns if self.target is None else [*self.columns, self.target]

    def missing(self, frame) -> list[str]:
        """Columns the contract needs that a frame does not provide."""
        available = set(getattr(frame, "columns", []))
        return [column for column in self.all_columns if column not in available]

    def validate(self, frame, *, allow_missing: bool = False) -> None:
        missing = self.missing(frame)
        if missing and not allow_missing:
            raise KeyError(f"{self.name} frame is missing contract column(s): {missing}")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "numeric_features": self.numeric,
            "categorical_features": self.categorical,
            "target": self.target,
            "unit": self.unit,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FeatureContract({self.name!r}, features={len(self.columns)}, target={self.target!r})"


FEATURE_SCHEMA: dict[str, FeatureContract] = {
    contract.name: contract
    for contract in (
        FeatureContract("overspend_classifier", OVERSPEND_NUMERIC_FEATURES, OVERSPEND_CATEGORICAL_FEATURES, OVERSPEND_TARGET, unit="label"),
        FeatureContract("expense_forecaster", FORECAST_NUMERIC_FEATURES, FORECAST_CATEGORICAL_FEATURES, FORECAST_TARGET),
        FeatureContract("savings_recommender", SAVINGS_NUMERIC_FEATURES, SAVINGS_CATEGORICAL_FEATURES, SAVINGS_TARGET),
        FeatureContract("spending_clusters", CLUSTER_NUMERIC_FEATURES, CLUSTER_CATEGORICAL_FEATURES, None, unit="transaction"),
    )
}


def feature_frame(model_name: str) -> list[str]:
    """Contract column order for a model, target included."""
    try:
        return FEATURE_SCHEMA[model_name].all_columns
    except KeyError as exc:  # pragma: no cover - programming error
        raise KeyError(
            f"unknown model {model_name!r}; expected one of {sorted(FEATURE_SCHEMA)}"
        ) from exc


def contract_digest() -> list[dict[str, object]]:
    """Serializable view of every contract, for the training manifest."""
    return [contract.as_dict() for contract in FEATURE_SCHEMA.values()]
