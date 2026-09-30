"""Predictions, model registry status, and on-demand inference.

Two usage patterns:

- Read model: `GET /predictions` and `GET /predictions/latest` return whatever is
  already stored (refreshed nightly and on writes).
- On-demand: the `POST /predictions/*` endpoints run inference for the caller
  immediately and return the freshly stored values.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query, status as http_status

from app.api.deps import Services, get_current_user, get_services
from app.core.constants import PredictionType
from app.core.errors import ValidationError
from app.core.pagination import Page, PageParams
from app.core.time_utils import strip_tz, utc_now
from app.models.base import serialize_doc
from app.models.prediction import (
    BudgetRecommendationResponse,
    ExpenseForecastResponse,
    ModelRegistryResponse,
    OverspendRiskResponse,
    PredictionOut,
    PredictionSummary,
    SavingsRecommendationResponse,
    SpendingClusterResponse,
)

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("", response_model=Page[PredictionOut])
def list_predictions(
    params: PageParams = Depends(),
    types: str | None = Query(default=None, description="Comma-separated PredictionType values"),
    single_type: str | None = Query(default=None, alias="type", description="Single PredictionType value"),
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> Page[PredictionOut]:
    raw = ",".join(part for part in (types, single_type) if part)
    type_values = _parse_types(raw)
    rows = services.predictions().list_for_user(
        user["_id"], types=type_values, skip=params.skip, limit=params.limit
    )
    total = services.predictions().count_for_user(user["_id"], types=type_values)
    items = [r for r in (serialize_doc(doc) for doc in rows) if r]
    return Page.build(items=items, total=total, params=params)


@router.get("/latest", response_model=list[PredictionSummary])
def latest_by_type(
    types: str | None = Query(default=None, description="Comma-separated PredictionType values"),
    single_type: str | None = Query(default=None, alias="type", description="Single PredictionType value"),
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> list[PredictionSummary]:
    return _summaries(
        services.predictions().latest_by_type(user["_id"], types=_parse_types(types, single_type))
    )


def _ensure_fresh(services: Services, user_id: Any, *, max_age_hours: int = 12) -> None:
    """Run inference only when the caller's stored predictions are missing or stale.

    The on-demand endpoints are cheap to call from the UI, so a fresh set of
    predictions (written by the nightly job, an explicit `/inference/run`, or an
    earlier on-demand call) is reused instead of re-scoring the whole user.
    """
    latest = services.predictions().latest_by_type(user_id, limit_per_type=1)
    if latest:
        newest = max(
            (doc["generated_at"] for doc in latest if doc.get("generated_at")),
            default=None,
        )
        if newest is not None and strip_tz(utc_now()) - strip_tz(newest) < timedelta(hours=max_age_hours):
            return
    services.ml_service().run_inference(user_ids=[user_id])


def _parse_types(types: str | None, single_type: str | None = None) -> list[str] | None:
    """Validate a comma-separated `types` (or single `type`) query value."""
    raw = ",".join(part for part in (types, single_type) if part)
    if not raw:
        return None
    values: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            PredictionType(token)
        except ValueError as exc:
            raise ValidationError(f"Invalid PredictionType: {token!r}") from exc
        if token not in values:
            values.append(token)
    return values or None


@router.get("/models/status", response_model=ModelRegistryResponse)
def model_status(services: Services = Depends(get_services)) -> ModelRegistryResponse:
    return services.ml_service().models_status()


@router.post("/inference/run", status_code=http_status.HTTP_200_OK)
def run_inference_for_user(
    force: bool = Query(default=False, description="Re-score even when today's predictions are fresh"),
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict[str, Any]:
    summary = services.ml_service().run_inference(user_ids=[user["_id"]])
    return {"run_id": summary["run_id"], "users_scored": summary["users_scored"],
            "predictions_written": summary["predictions_written"], "errors": summary["errors"]}


@router.post("/expense-forecast", response_model=ExpenseForecastResponse)
def expense_forecast(
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> ExpenseForecastResponse:
    _ensure_fresh(services, user["_id"])
    return services.ml_service().expense_forecast(user["_id"])


@router.post("/overspend-risk", response_model=OverspendRiskResponse)
def overspend_risk(
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> OverspendRiskResponse:
    _ensure_fresh(services, user["_id"])
    return services.ml_service().overspend_risk(user["_id"])


@router.post("/budget-recommendations", response_model=BudgetRecommendationResponse)
def budget_recommendations(
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> BudgetRecommendationResponse:
    _ensure_fresh(services, user["_id"])
    return services.ml_service().budget_recommendations(user["_id"])


@router.post("/savings-recommendation", response_model=SavingsRecommendationResponse)
def savings_recommendation(
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> SavingsRecommendationResponse:
    _ensure_fresh(services, user["_id"])
    return services.ml_service().savings_recommendation(user["_id"])


@router.get("/clusters", response_model=SpendingClusterResponse)
def spending_clusters(
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> SpendingClusterResponse:
    return services.ml_service().spending_clusters(user["_id"])


def _summaries(rows: list[dict[str, Any]]) -> list[PredictionSummary]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(str(row.get("type")), []).append(row)

    out: list[PredictionSummary] = []
    for prediction_type, docs in by_type.items():
        values = [float(doc.get("value") or 0.0) for doc in docs]
        confidences = [float(doc.get("confidence_score") or 0.0) for doc in docs]
        generated = [doc.get("generated_at") for doc in docs if doc.get("generated_at")]
        out.append(
            PredictionSummary(
                type=PredictionType(prediction_type),
                count=len(docs),
                latest_value=values[-1] if values else None,
                average_confidence=round(sum(confidences) / len(confidences), 4) if confidences else None,
                latest_generated_at=generated[-1] if generated else None,
            )
        )
    return out