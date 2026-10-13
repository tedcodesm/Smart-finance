"""Bridge from the FastAPI app to the `mercy_ml` package.

Three responsibilities:

1. Point `mercy_ml`'s read path at the *same* MongoDB handle the backend uses
   (so mongomock overrides in tests flow through unchanged).
2. Drive on-demand inference and batch refreshes, which the scheduler reuses.
3. Expose model-registry status and translate stored `predictions` documents
   into the API response models defined in `app.models.prediction`.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Iterable

from app.core.constants import PredictionType
from app.core.errors import NotFoundError, ServiceUnavailableError
from app.core.logging import get_logger
from app.core.time_utils import month_key, strip_tz, utc_now
from app.models.prediction import (
    BudgetRecommendationItem,
    BudgetRecommendationResponse,
    ExpenseForecastItem,
    ExpenseForecastResponse,
    ModelInfo,
    ModelRegistryResponse,
    ModelRunInfo,
    OverspendRiskItem,
    OverspendRiskResponse,
    SavingsRecommendationResponse,
    SpendingCluster,
    SpendingClusterResponse,
)

logger = get_logger(__name__)

#: model *source* name (artifact filename stem / loader key) -> PredictionType.
_MODEL_TYPE_BY_SOURCE = {
    "expense_forecaster": PredictionType.EXPENSE_FORECAST,
    "expense_forecast": PredictionType.EXPENSE_FORECAST,
    "overspend_classifier": PredictionType.OVERSPEND_RISK,
    "overspend_risk": PredictionType.OVERSPEND_RISK,
    "spending_clusters": PredictionType.SPENDING_CLUSTER,
    "spending_cluster": PredictionType.SPENDING_CLUSTER,
    "budget_recommender": PredictionType.BUDGET_RECOMMENDATION,
    "budget_recommendation": PredictionType.BUDGET_RECOMMENDATION,
    "savings_recommender": PredictionType.SAVINGS_RECOMMENDATION,
    "savings_recommendation": PredictionType.SAVINGS_RECOMMENDATION,
}

_PREDICTION_COLLECTION = "predictions"
_MODEL_RUNS_COLLECTION = "model_runs"


class MLService:
    def __init__(self, db: Any) -> None:
        self.db = db
        self._runtime_lock = threading.Lock()
        self._runtime_ready = False

    # ------------------------------------------------------------------ runtime
    def _ensure_runtime(self) -> None:
        if self._runtime_ready:
            return
        with self._runtime_lock:
            if self._runtime_ready:
                return
            from mercy_ml.data import get_client, set_client

            if get_client() is not self.db.client:
                set_client(self.db.client)
            self._runtime_ready = True

    # ------------------------------------------------------------------ batch
    def run_inference(self, user_ids: Iterable[Any] | None = None, as_of: Any = None) -> dict[str, Any]:
        """Score users and refresh `predictions`. Returns the run summary."""
        try:
            self._ensure_runtime()
            from mercy_ml.infer import run_inference

            return run_inference(user_ids=list(user_ids) if user_ids else None, as_of=as_of)
        except RuntimeError as exc:
            if "No artifacts found" in str(exc) or "mercy-ml-train" in str(exc):
                raise ServiceUnavailableError("No trained ML model artifacts found. Please run training first.") from exc
            raise
        except ImportError as exc:
            raise ServiceUnavailableError("ML runtime is unavailable.") from exc

    def run_batch_inference(self) -> dict[str, Any]:
        """Nightly job: score every active user."""
        user_ids = [doc["_id"] for doc in self.db.users.find({"is_active": True}, {"_id": 1})]
        logger.info("Running nightly ML inference for %d active user(s)", len(user_ids))
        return self.run_inference(user_ids=user_ids)

    # ------------------------------------------------------------ model status
    def models_status(self) -> ModelRegistryResponse:
        try:
            self._ensure_runtime()
            from mercy_ml.config import get_settings
            from mercy_ml.registry import available_models

            available = available_models()
            artifact_dir = str(get_settings().artifact_path)
        except Exception as exc:
            logger.warning("Failed to inspect ML models: %s", exc)
            return ModelRegistryResponse(
                available=False,
                artifact_dir="unavailable",
                models=[],
                latest_run=None,
                message="Models unavailable — run ml-train",
            )

        latest_run = (
            self.db[_MODEL_RUNS_COLLECTION].find_one({"run_type": "inference"}, sort=[("created_at", -1)])
            or {}
        )
        run_id = str(latest_run.get("run_id") or "") if latest_run else ""

        models: list[ModelInfo] = []
        for entry in available:
            source_name = entry.get("name") or ""
            model_type = _MODEL_TYPE_BY_SOURCE.get(
                source_name,
                next(
                    (t for key, t in _MODEL_TYPE_BY_SOURCE.items() if key in source_name),
                    PredictionType.EXPENSE_FORECAST,
                ),
            )
            trained_at = _parse_datetime(entry.get("trained_at"))
            models.append(
                ModelInfo(
                    model_type=model_type,
                    model_version=str(entry.get("version") or f"{source_name}_v1"),
                    run_id=run_id,
                    trained_at=trained_at,
                    is_active=True,
                    metrics=entry.get("metrics") or {},
                    n_train=None if entry.get("n_samples") is None else int(entry["n_samples"]),
                    n_test=entry.get("n_users"),
                )
            )

        message = None if models else "No trained artifacts found - run `mercy-ml-train` first."
        return ModelRegistryResponse(
            available=bool(models),
            artifact_dir=artifact_dir,
            models=models,
            latest_run=_model_run_info(latest_run) if latest_run else None,
            message=message,
        )

    # -------------------------------------------------------- on-demand reads
    @staticmethod
    def _predictions_for(db: Any, user_id: Any, prediction_type: str) -> list[dict[str, Any]]:
        return [
            doc for doc in db[_PREDICTION_COLLECTION].find({"user_id": user_id, "type": prediction_type})
        ]

    def _require_predictions(self, user_id: Any, prediction_type: str) -> list[dict[str, Any]]:
        docs = self._predictions_for(self.db, user_id, prediction_type.value)
        if not docs:
            raise NotFoundError(
                f"No {prediction_type.value} prediction available - run inference first "
                "(POST /api/v1/predictions/inference/run)."
            )
        return docs

    # -------------------------------------------------------------- endpoints
    def expense_forecast(self, user_id: Any) -> ExpenseForecastResponse:
        docs = self._require_predictions(user_id, PredictionType.EXPENSE_FORECAST)
        docs.sort(key=lambda d: str(d.get("scope", {}).get("period_key") or ""), reverse=True)
        model_version = docs[0].get("model_version") or "unknown"
        run_id = str(docs[0].get("model_run_id") or "")

        current_spend = _current_category_spend(self.db, user_id)
        items: list[ExpenseForecastItem] = []
        for doc in docs:
            scope = doc.get("scope") or {}
            extra = doc.get("extra") or {}
            category_id = scope.get("category_id")
            predicted = float(doc.get("value") or 0.0)
            lower = _coerce_float(extra.get("lower_bound"))
            upper = _coerce_float(extra.get("upper_bound"))
            items.append(
                ExpenseForecastItem(
                    category_id=category_id,
                    category_name=scope.get("category_name"),
                    period_key=scope.get("period_key") or month_key(strip_tz(utc_now())),
                    predicted_amount=predicted,
                    lower_bound=lower,
                    upper_bound=upper,
                    confidence_score=float(doc.get("confidence_score") or 0.0),
                    trend=_trend(predicted, current_spend.get(category_id)),
                )
            )

        totals = [i.predicted_amount for i in items]
        return ExpenseForecastResponse(
            run_id=run_id,
            model_version=model_version,
            target_period_key=items[0].period_key if items else month_key(strip_tz(utc_now())),
            period="monthly",
            total_predicted=round(sum(totals), 2),
            total_lower_bound=_sum_or_none([i.lower_bound for i in items]),
            total_upper_bound=_sum_or_none([i.upper_bound for i in items]),
            items=items,
        )

    def overspend_risk(self, user_id: Any) -> OverspendRiskResponse:
        docs = self._require_predictions(user_id, PredictionType.OVERSPEND_RISK)
        model_version = docs[0].get("model_version") or "unknown"
        run_id = str(docs[0].get("model_run_id") or "")

        items: list[OverspendRiskItem] = []
        for doc in docs:
            scope = doc.get("scope") or {}
            probability = float(doc.get("value") or 0.0)
            items.append(
                OverspendRiskItem(
                    category_id=scope.get("category_id"),
                    category_name=scope.get("category_name"),
                    budget_id=scope.get("budget_id"),
                    probability=probability,
                    label=doc.get("label") or _risk_label(probability),
                    confidence_score=float(doc.get("confidence_score") or 0.0),
                    model_version=model_version,
                    features=doc.get("features") or {},
                    computed_at=doc.get("generated_at") or datetime.now(timezone.utc),
                )
            )
        items.sort(key=lambda item: item.probability, reverse=True)

        global_p = round(sum(i.probability for i in items) / len(items), 4) if items else None
        return OverspendRiskResponse(
            run_id=run_id,
            as_of=datetime.now(timezone.utc),
            model_version=model_version,
            global_probability=global_p,
            items=items,
        )

    def budget_recommendations(self, user_id: Any) -> BudgetRecommendationResponse:
        docs = self._require_predictions(user_id, PredictionType.BUDGET_RECOMMENDATION)
        docs.sort(key=lambda d: -float(d.get("value") or 0.0))
        model_version = docs[0].get("model_version") or "unknown"
        run_id = str(docs[0].get("model_run_id") or "")

        items: list[BudgetRecommendationItem] = []
        for doc in docs:
            scope = doc.get("scope") or {}
            extra = doc.get("extra") or {}
            items.append(
                BudgetRecommendationItem(
                    category_id=scope.get("category_id"),
                    category_name=scope.get("category_name"),
                    current_limit=_coerce_float(extra.get("current_limit")),
                    recommended_limit=float(doc.get("value") or 0.0),
                    basis=doc.get("features") or {},
                    confidence_score=float(doc.get("confidence_score") or 0.0),
                )
            )
        return BudgetRecommendationResponse(
            run_id=run_id, model_version=model_version, period="monthly", items=items
        )

    def savings_recommendation(self, user_id: Any) -> SavingsRecommendationResponse:
        docs = self._require_predictions(user_id, PredictionType.SAVINGS_RECOMMENDATION)
        doc = docs[0]
        features = doc.get("features") or {}
        extra = doc.get("extra") or {}

        income = _coerce_float(features.get("period_income"))
        predicted_expenses = _coerce_float(features.get("period_expense"))
        predicted_net = float(doc.get("value") or 0.0)
        suggested = _coerce_float(extra.get("suggested_contribution_total"))
        return SavingsRecommendationResponse(
            run_id=str(doc.get("model_run_id") or ""),
            model_version=doc.get("model_version") or "unknown",
            period="monthly",
            average_income=income or 0.0,
            predicted_expenses=predicted_expenses if predicted_expenses is not None else max(0.0, income - predicted_net),
            discretionary_spend=predicted_net,
            recommended_savings=suggested or 0.0,
            savings_rate_pct=_coerce_float(extra.get("savings_rate_pct")) or 0.0,
            confidence_score=float(doc.get("confidence_score") or 0.0),
            basis={"applied_goals": extra.get("goals") or [], "horizon": "next month"},
        )

    def spending_clusters(self, user_id: Any | None = None) -> SpendingClusterResponse:
        self._ensure_runtime()
        from mercy_ml.registry import load_model

        clusterer = load_model("spending_clusters")
        if clusterer is None or getattr(clusterer, "centroid_frame", None) is None:
            raise ServiceUnavailableError("spending_clusters artifact is not available")

        run_id = ""
        confidence = 0.0
        if user_id is not None:
            docs = self._predictions_for(self.db, user_id, PredictionType.SPENDING_CLUSTER.value)
            if docs:
                run_id = str(docs[0].get("model_run_id") or "")
                confidence = float(docs[0].get("confidence_score") or 0.0)

        global_descriptions = clusterer.describe()
        profile: dict[str, Any] = {}
        per_user: dict[str, list[dict[str, Any]]] | None = None
        if user_id is not None:
            docs = self._predictions_for(self.db, user_id, PredictionType.SPENDING_CLUSTER.value)
            if docs:
                profile = (docs[0].get("extra") or {}).get("profile") or []
                per_user = {str(user_id): list(profile)}

        clusters: list[SpendingCluster] = []
        cluster_by_id = {int(item["cluster_id"]): item for item in global_descriptions}
        profile_by_id = {int(item.get("cluster_id") or 0): item for item in profile}
        for cluster_id in sorted(cluster_by_id):
            item = cluster_by_id[cluster_id]
            user_item = profile_by_id.get(cluster_id, {})
            mean = _coerce_float(user_item.get("mean_amount"))
            clusters.append(
                SpendingCluster(
                    cluster_id=cluster_id,
                    label=item.get("label") or f"Cluster {cluster_id}",
                    size=int(item.get("size") or 0),
                    share_pct=float(item.get("share_pct") or 0.0),
                    mean_amount=round(mean, 2) if mean is not None else 0.0,
                    dominant_category=_first_category(item.get("dominant_categories")),
                    description=item.get("description"),
                )
            )

        return SpendingClusterResponse(
            run_id=run_id,
            model_version=clusterer.report.model_version if clusterer.report else getattr(clusterer, "name", "spending_clusters"),
            silhouette_score=clusterer.report.metrics.get("silhouette_score") if clusterer.report else None,
            n_clusters=len(clusters),
            clusters=clusters,
            per_user=per_user,
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:  # pragma: no cover - defensive
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    return number if number == number else None  # NaN -> None


def _sum_or_none(values: list[float | None]) -> float | None:
    cleaned = [v for v in values if v is not None]
    return round(sum(cleaned), 2) if cleaned else None


def _model_run_info(doc: dict[str, Any]) -> ModelRunInfo:
    """Project a `model_runs` document written by `mercy_ml.infer` onto the API model."""
    started = _parse_datetime(doc.get("started_at"))
    finished = _parse_datetime(doc.get("finished_at"))
    duration = None
    if started and finished:
        duration = round((finished - started).total_seconds(), 3)
    errors = doc.get("errors") or []
    return ModelRunInfo(
        run_id=str(doc.get("run_id") or ""),
        run_type=doc.get("run_type"),
        model_type=(doc.get("models") or [None])[0] if doc.get("models") else None,
        status=doc.get("status"),
        is_active=bool(doc.get("is_active", True)),
        n_users=doc.get("users_scored"),
        n_predictions=doc.get("predictions_written"),
        started_at=started,
        finished_at=finished,
        duration_seconds=duration,
        message="; ".join(str(err) for err in errors) if errors else None,
    )


def _risk_label(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"


def _trend(predicted: float, actual: float | None) -> str | None:
    if actual is None or actual <= 0:
        return None
    if predicted > actual * 1.05:
        return "up"
    if predicted < actual * 0.95:
        return "down"
    return "flat"


def _current_category_spend(db: Any, user_id: Any) -> dict[Any, float]:
    from app.core.time_utils import period_window, strip_tz, utc_now

    from app.core.constants import BudgetPeriod, TransactionType

    now = strip_tz(utc_now())
    start, end, _ = period_window(BudgetPeriod.MONTHLY, now)
    spend: dict[Any, float] = {}
    pipeline = [
        {"$match": {"user_id": user_id, "type": TransactionType.EXPENSE.value, "date": {"$gte": start, "$lt": end}}},
{"$group": {"_id": "$category_id", "total": {"$sum": "$amount"}}},
    ]
    for row in db.transactions.aggregate(pipeline):
        spend[row["_id"]] = float(row["total"])
    return spend


def _first_category(names: Any) -> str | None:
    if not names:
        return None
    if isinstance(names, str):
        return names
    try:
        return str(names[0])
    except (TypeError, IndexError):  # pragma: no cover - defensive
        return None