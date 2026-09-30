"""Budget CRUD and period projections."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status as http_status

from app.api.deps import Services, get_current_user, get_services, require_doc
from app.core.constants import BudgetPeriod
from app.core.pagination import Page, PageParams
from app.core.time_utils import strip_tz, utc_now
from app.models.base import serialize_doc, to_object_id
from app.models.budget import BudgetCreate, BudgetOut, BudgetProgress, BudgetUpdate

router = APIRouter(prefix="/budgets", tags=["budgets"])


@router.get("", response_model=Page[BudgetOut])
def list_budgets(
    params: PageParams = Depends(),
    is_active: bool | None = Query(default=True),
    period: BudgetPeriod | None = None,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> Page[BudgetOut]:
    rows = services.budget_service().list_budgets(
        user["_id"], is_active=is_active, period=period
    )
    page_rows = [serialize_doc(row) for row in rows[params.skip : params.skip + params.limit]]
    return Page.build(items=[r for r in page_rows if r is not None], total=len(rows), params=params)


@router.post("", response_model=BudgetOut, status_code=http_status.HTTP_201_CREATED)
def create_budget(
    payload: BudgetCreate,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    doc = services.budget_service().create_budget(user["_id"], payload.model_dump())
    return services.budget_service().serialize(require_doc(doc, "Budget could not be created"))


@router.get("/{budget_id}", response_model=BudgetOut)
def get_budget(
    budget_id: str,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    doc = services.budget_service().get_budget(user["_id"], to_object_id(budget_id, field="budget_id"))
    if doc is None:
        from app.core.errors import NotFoundError

        raise NotFoundError("Budget not found")
    return services.budget_service().serialize(doc)


@router.get("/{budget_id}/progress", response_model=BudgetProgress)
@router.get("/{budget_id}/projection", response_model=BudgetProgress, include_in_schema=False)
def budget_progress(
    budget_id: str,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    budget = services.budget_service().get_budget(user["_id"], to_object_id(budget_id, field="budget_id"))
    if budget is None:
        from app.core.errors import NotFoundError

        raise NotFoundError("Budget not found")

    spent = float(budget.get("spent_amount") or 0.0)
    limit = float(budget.get("limit_amount") or 0.0)
    elapsed = float(budget.get("period_elapsed_pct") or 0.0)
    return BudgetProgress(
        budget_id=budget_id,
        limit_amount=limit,
        spent_amount=spent,
        remaining_amount=max(0.0, limit - spent),
        usage_pct=float(budget.get("usage_pct") or 0.0),
        period=BudgetPeriod(budget.get("period") or BudgetPeriod.MONTHLY),
        period_key=budget.get("period_key"),
        days_remaining=int(budget.get("days_remaining") or 0),
        period_elapsed_pct=elapsed,
        pace_status=budget.get("pace_status") or "on_track",
        is_exceeded=bool(budget.get("is_exceeded")),
        is_threshold_crossed=bool(budget.get("is_threshold_crossed")),
        projected_total=budget.get("projected_total"),
    ).model_dump()


@router.patch("/{budget_id}", response_model=BudgetOut)
def update_budget(
    budget_id: str,
    changes: BudgetUpdate,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    doc = services.budget_service().update_budget(
        user["_id"], to_object_id(budget_id, field="budget_id"), changes.model_dump(exclude_none=True)
    )
    return services.budget_service().serialize(require_doc(doc, "Budget not found"))


@router.delete("/{budget_id}", status_code=http_status.HTTP_204_NO_CONTENT)
def delete_budget(
    budget_id: str,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> None:
    services.budget_service().delete_budget(user["_id"], to_object_id(budget_id, field="budget_id"))
    return None
