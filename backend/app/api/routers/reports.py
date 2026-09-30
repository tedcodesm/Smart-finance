"""Reports: generate on demand, list persisted ones, export to CSV/PDF/JSON."""

from __future__ import annotations

import io
import json
from datetime import datetime

from fastapi import APIRouter, Depends, Query, Response

from app.api.deps import Services, get_current_user, get_services, require_doc
from app.core.constants import ReportType
from app.core.errors import NotFoundError, ValidationError
from app.core.pagination import Page, PageParams
from app.core.time_utils import strip_tz, utc_now
from app.models.base import serialize_doc, to_object_id
from app.models.report import ReportData, ReportOut, ReportRequest
from app.services.export_service import (
    CSV_MEDIA_TYPE,
    JSON_MEDIA_TYPE,
    PDF_MEDIA_TYPE,
    report_to_pdf,
    rows_to_csv,
    summary_csv_block,
)

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("", response_model=ReportOut, status_code=201)
def create_report(
    request: ReportRequest,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    """Generate a report and persist it so it can be listed, exported and deleted."""
    data = services.report_service().generate(user["_id"], request)
    report_id = services.report_service().store(user["_id"], data, request.model_dump(exclude={"persist"}))
    doc = services.report_service().get_report(user["_id"], to_object_id(report_id, field="report_id"))
    return require_doc(services.report_service().serialize(doc), "Report not found after storing")


@router.post("/generate", response_model=ReportData)
def generate_report(
    request: ReportRequest,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> ReportData:
    """Generate a report on demand without writing to the `reports` collection."""
    return services.report_service().generate(user["_id"], request)


@router.get("", response_model=Page[ReportOut])
def list_reports(
    params: PageParams = Depends(),
    report_type: ReportType | None = None,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> Page[ReportOut]:
    rows, total = services.report_service().list_reports(
        user["_id"], report_type=report_type, skip=params.skip, limit=params.limit
    )
    items = [r for r in (services.report_service().serialize(doc) for doc in rows) if r]
    return Page.build(items=items, total=total, params=params)


@router.get("/{report_id}", response_model=ReportOut)
def get_report(
    report_id: str,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    doc = services.report_service().get_report(user["_id"], to_object_id(report_id, field="report_id"))
    if doc is None:
        raise NotFoundError("Report not found")
    return services.report_service().serialize(doc) or {}


@router.delete("/{report_id}", status_code=204)
def delete_report(
    report_id: str,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> None:
    services.report_service().delete_report(user["_id"], to_object_id(report_id, field="report_id"))
    return None


@router.get("/{report_id}/export")
def export_report(
    report_id: str,
    format: str = Query(default="csv", pattern="^(csv|pdf|json)$"),
    include_summary: bool = True,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> Response:
    doc = services.report_service().get_report(user["_id"], to_object_id(report_id, field="report_id"))
    if doc is None:
        raise NotFoundError("Report not found")

    data: dict = doc.get("data") or {}
    summary: dict = doc.get("summary") or {}
    pretty = _report_title(doc)
    stamp = strip_tz(doc.get("generated_at") or utc_now()).strftime("%Y-%m-%d")
    filename = f"mercy-report-{doc.get('period_key') or stamp}-{format}"

    if format == "json":
        payload = {
            "report_type": doc.get("report_type"),
            "period": doc.get("period"),
            "period_key": doc.get("period_key"),
            "generated_at": doc.get("generated_at"),
            "summary": summary,
            "data": data,
        }
        return Response(
            content=json.dumps(payload, ensure_ascii=False, default=_json_default),
            media_type=JSON_MEDIA_TYPE,
            headers={"Content-Disposition": f'attachment; filename="{filename}.json"'},
        )

    rows, columns = _rows_for(data)

    if format == "csv":
        from app.services.export_service import rows_to_csv

        blocks: list[str] = []
        if include_summary:
            blocks.extend(summary_csv_block(pretty, summary))
        blocks.extend(rows_to_csv(rows, columns=columns))
        content = "".join(blocks)
        return Response(
            content="\ufeff" + content,
            media_type=CSV_MEDIA_TYPE,
            headers={"Content-Disposition": f'attachment; filename="{filename}.csv"'},
        )

    pdf = report_to_pdf(
        title=pretty,
        subtitle=f"{doc.get('period')} report for {doc.get('period_key')}",
        totals=summary if include_summary else {},
        sections=[(_section_name(doc), rows, columns)],
        insights=data.get("insights") or [],
    )
    return Response(content=pdf, media_type=PDF_MEDIA_TYPE, headers={"Content-Disposition": f'attachment; filename="{filename}.pdf"'})


# --------------------------------------------------------------------------- #
def _report_title(doc: dict) -> str:
    return f"{_section_name(doc)} report"


def _section_name(doc: dict) -> str:
    return str(doc.get("report_type") or "report").replace("_", " ").title()


def _rows_for(data: dict) -> tuple[list[dict], list[tuple[str, str]]]:
    report_type = str(data.get("report_type") or "")
    if data.get("by_category"):
        rows = []
        for item in data["by_category"]:
            rows.append(
                {
                    "category_name": item.get("category_name") or "Uncategorised",
                    "total": item.get("total", 0.0),
                    "share_pct": item.get("share_pct", 0.0),
                    "average": item.get("average", 0.0),
                    "previous_total": item.get("previous_total"),
                    "change_pct": item.get("change_pct"),
                    "count": item.get("count", 0),
                }
            )
        return rows, [
            ("category_name", "Category"),
            ("total", "Total"),
            ("share_pct", "Share %"),
            ("average", "Average"),
            ("previous_total", "Previous"),
            ("change_pct", "Change %"),
            ("count", "Count"),
        ]
    if data.get("budgets"):
        rows = [
            {
                "category_name": item.get("category_name"),
                "limit_amount": item.get("limit_amount", 0.0),
                "spent_amount": item.get("spent_amount", 0.0),
                "usage_pct": item.get("usage_pct", 0.0),
                "remaining_amount": item.get("remaining_amount", 0.0),
                "pace_status": item.get("pace_status"),
            }
            for item in data["budgets"]
        ]
        return rows, [
            ("category_name", "Budget"),
            ("limit_amount", "Limit"),
            ("spent_amount", "Spent"),
            ("usage_pct", "Used %"),
            ("remaining_amount", "Remaining"),
            ("pace_status", "Pace"),
        ]
    if data.get("goals"):
        rows = [
            {
                "name": item.get("name"),
                "target_amount": item.get("target_amount", 0.0),
                "current_amount": item.get("current_amount", 0.0),
                "progress_pct": item.get("progress_pct", 0.0),
            }
            for item in data["goals"]
        ]
        return rows, [
            ("name", "Goal"),
            ("target_amount", "Target"),
            ("current_amount", "Saved"),
            ("progress_pct", "Progress %"),
        ]
    return [], []


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)