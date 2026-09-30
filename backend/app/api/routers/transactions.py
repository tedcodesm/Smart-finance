"""Transaction CRUD, summaries and export.

CSV/PDF exports stream bytes with a Content-Disposition header so the SPA can
download them directly from the API.
"""

from __future__ import annotations

import io  # noqa: F401  (kept for CSV buffer parity with the export service)

from fastapi import APIRouter, Depends, Query, Response, status as http_status
from fastapi.responses import StreamingResponse

from app.api.deps import Services, get_current_user, get_services, require_doc
from app.core.pagination import Page, PageParams, build_sort
from app.core.time_utils import resolve_range, strip_tz, utc_now
from app.models.base import serialize_doc, to_object_id, to_object_id_or_none
from app.models.transaction import (
    TransactionBulkCreate,
    TransactionCreate,
    TransactionOut,
    TransactionSummary,
    TransactionUpdate,
)
from app.services.export_service import CSV_MEDIA_TYPE, PDF_MEDIA_TYPE

router = APIRouter(prefix="/transactions", tags=["transactions"])

_SORT_FIELDS = {
    "date": "date",
    "amount": "amount",
    "type": "type",
    "created_at": "created_at",
    "period": "period_key",
    "category": "category_id",
}

_EXPORT_COLUMNS = ["date", "type", "category", "description", "amount", "signed_amount", "is_recurring"]


@router.get("", response_model=Page[TransactionOut])
def list_transactions(
    params: PageParams = Depends(),
    sort: str = Query(default="-date", description="Comma-separated, '-' for descending"),
    start: str | None = None,
    end: str | None = None,
    type: str | None = None,
    category_ids: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    search: str | None = None,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> Page[TransactionOut]:
    start_dt, end_dt = _parse_window(start, end)
    category_list = [to_object_id_or_none(c, field="category_id") for c in (category_ids or "").split(",") if c.strip()]
    category_list = category_list or None
    rows, total = services.transaction_service().list_transactions(
        user["_id"],
        start=start_dt,
        end=end_dt,
        type=type,
        category_ids=category_list,
        min_amount=min_amount,
        max_amount=max_amount,
        search=search,
        sort=build_sort(sort, _SORT_FIELDS, default="-date"),
        skip=params.skip,
        limit=params.limit,
    )
    return Page.build(items=_serialize_all(rows), total=total, params=params)


@router.post("", response_model=TransactionOut, status_code=http_status.HTTP_201_CREATED)
def create_transaction(
    payload: TransactionCreate,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    doc = services.transaction_service().create_transaction(
        user["_id"], payload.model_dump()
    )
    return serialize_doc(require_doc(doc, "Transaction not found")) or {}


@router.post("/bulk", response_model=Page[TransactionOut], status_code=http_status.HTTP_201_CREATED)
def bulk_create(
    payload: TransactionBulkCreate,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> Page[TransactionOut]:
    docs = services.transaction_service().bulk_create(
        user["_id"], [t.model_dump() for t in payload.transactions]
    )
    rows = _serialize_all(services.transaction_service().enrich_many(docs))
    return Page.build(items=rows, total=len(rows), params=PageParams())



@router.get("/summary", response_model=TransactionSummary)
def summary(
    start: str | None = None,
    end: str | None = None,
    period: str = Query(default=None, description="monthly | yearly"),
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    if period and not (start or end):
        from app.core.constants import BudgetPeriod
        from app.core.time_utils import period_window

        start_dt, end_dt, _ = period_window(BudgetPeriod(period), strip_tz(utc_now()))
        end_dt = end_dt - _ONE_MICROSECOND
    else:
        start_dt, end_dt = _parse_window(start, end)
    return services.transaction_service().summary(user["_id"], start=start_dt, end=end_dt)


@router.get("/export")
def export_transactions(
    format: str = Query(default="csv", pattern="^(csv|pdf)$"),
    start: str | None = None,
    end: str | None = None,
    type: str | None = None,
    category_ids: str | None = None,
    max_rows: int = Query(default=5000, ge=1, le=100_000),
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> Response:
    start_dt, end_dt = _parse_window(start, end)
    category_list = [to_object_id_or_none(c, field="category_id") for c in (category_ids or "").split(",") if c.strip()]
    service = services.transaction_service()
    rows = service.export_rows(user["_id"], start=start_dt, end=end_dt, type=type, category_ids=category_list or None, limit=max_rows)
    summary = service.summary(user["_id"], start=start_dt, end=end_dt)

    stamp = strip_tz(utc_now()).strftime("%Y-%m-%d")
    filename = f"mercy-transactions-{stamp}.{format}"
    if format == "csv":
        from app.services.export_service import rows_to_csv

        def iterator():
            yield "\ufeff"  # BOM so Excel opens UTF-8 CSVs correctly
            yield from rows_to_csv(rows, columns=_EXPORT_COLUMNS)

        return StreamingResponse(
            iterator(),
            media_type=CSV_MEDIA_TYPE,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    from app.services.export_service import transactions_to_pdf

    pdf = transactions_to_pdf(rows=rows, summary=dict(summary), subtitle=f"{start} to {end or strip_tz(utc_now()).date()}")
    return Response(content=pdf, media_type=PDF_MEDIA_TYPE, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/{transaction_id}", response_model=TransactionOut)
def get_transaction(
    transaction_id: str,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    doc = services.transaction_service().get_transaction(
        user["_id"], to_object_id(transaction_id, field="transaction_id")
    )
    return serialize_doc(require_doc(doc, "Transaction not found")) or {}


@router.patch("/{transaction_id}", response_model=TransactionOut)
def update_transaction(
    transaction_id: str,
    changes: TransactionUpdate,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    doc = services.transaction_service().update_transaction(
        user["_id"], to_object_id(transaction_id, field="transaction_id"), changes.model_dump(exclude_none=True)
    )
    return serialize_doc(require_doc(doc, "Transaction not found")) or {}


@router.delete("/{transaction_id}", status_code=http_status.HTTP_204_NO_CONTENT)
def delete_transaction(
    transaction_id: str,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> None:
    services.transaction_service().delete_transaction(
        user["_id"], to_object_id(transaction_id, field="transaction_id")
    )
    return None


# --------------------------------------------------------------------------- #
from datetime import timedelta as _timedelta

_ONE_MICROSECOND = _timedelta(microseconds=1)


def _parse_window(start: str | None, end: str | None) -> tuple[object, object]:
    from datetime import datetime

    start_dt: object = None
    end_dt: object = None
    if start:
        start_dt = datetime.fromisoformat(start)
    if end:
        end_dt = datetime.fromisoformat(end) + _ONE_MICROSECOND
    return start_dt, end_dt


def _serialize_all(rows: list[dict]) -> list[dict]:
    return [out for out in (serialize_doc(row) for row in rows) if out]


