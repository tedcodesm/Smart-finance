"""CSV and PDF export.

CSV is streamed from a generator so a large transaction range never has to be
materialised as one string; the PDF is rendered with ReportLab's platypus flow
ables so a long category table paginates automatically.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Any, Iterator, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.core.logging import get_logger

logger = get_logger(__name__)

CSV_MEDIA_TYPE = "text/csv"
PDF_MEDIA_TYPE = "application/pdf"
JSON_MEDIA_TYPE = "application/json"

_MONEY_COLUMNS = {"amount", "signed_amount", "income", "expense", "net", "total", "spent_amount"}


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def rows_to_csv(rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> Iterator[str]:
    """Yield CSV text, one row at a time."""
    if not rows:
        return
    buffer = io.StringIO()
    fieldnames = list(columns) if columns else list(rows[0].keys())
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    yield buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)

    for row in rows:
        writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:.2f}" if isinstance(value, float) else str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


def summary_csv_block(title: str, summary: dict[str, Any]) -> Iterator[str]:
    """A small `key,value` block prepended to a report CSV."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([f"# {title}"])
    for key, value in summary.items():
        writer.writerow([key, _csv_value(value)])
    writer.writerow([])
    yield buffer.getvalue()


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("MTitle", parent=base["Title"], fontSize=18, spaceAfter=4),
        "subtitle": ParagraphStyle("MSub", parent=base["Normal"], fontSize=9, textColor=colors.HexColor("#64748b")),
        "h2": ParagraphStyle("MH2", parent=base["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=6),
        "cell": ParagraphStyle("MCell", parent=base["Normal"], fontSize=8),
        "right": ParagraphStyle("MRight", parent=base["Normal"], fontSize=8, alignment=TA_RIGHT),
        "insight": ParagraphStyle(
            "MInsight", parent=base["Normal"], fontSize=9, leftIndent=6, textColor=colors.HexColor("#334155")
        ),
    }


def _table(rows: Sequence[dict[str, Any]], columns: Sequence[tuple[str, str]]) -> Table:
    """Build a styled table; `columns` is a list of `(key, heading)` pairs."""
    header = [heading for _, heading in columns]
    body = [header]
    numeric = {key for key, _ in columns}
    for row in rows:
        body.append([_cell(row.get(key), key in numeric) for key, _ in columns])
    table = Table(body, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for index, (key, _) in enumerate(columns):
        if key in _MONEY_COLUMNS or key in numeric:
            style.append(("ALIGN", (index, 1), (index, -1), "RIGHT"))
    table.setStyle(TableStyle(style))
    return table


def _cell(value: Any, numeric: bool) -> Paragraph:
    styles = _styles()
    text = "" if value is None else f"{value:,.2f}" if (numeric and isinstance(value, (int, float))) else str(value)
    return Paragraph(text, styles["right"] if numeric else styles["cell"])


def report_to_pdf(
    *,
    title: str,
    subtitle: str,
    totals: dict[str, Any],
    sections: Sequence[tuple[str, Sequence[dict[str, Any]], Sequence[tuple[str, str]]]],
    insights: Sequence[str] = (),
) -> bytes:
    """Render a multi-section report to PDF bytes."""
    styles = _styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title=title,
        author="Mercy",
    )

    story: list[Any] = [
        Paragraph(title, styles["title"]),
        Paragraph(subtitle, styles["subtitle"]),
        Spacer(1, 8 * mm),
    ]

    if totals:
        story.append(Paragraph("Summary", styles["h2"]))
        money_rows = [
            {"label": _prettify(key), "value": value} for key, value in totals.items() if isinstance(value, (int, float))
        ]
        story.append(_table(money_rows, [("label", "Metric"), ("value", "Value")]))
        story.append(Spacer(1, 6 * mm))

    for heading, rows, columns in sections:
        story.append(Paragraph(heading, styles["h2"]))
        if rows:
            story.append(_table(rows, columns))
        else:
            story.append(Paragraph("No data for this section.", styles["insight"]))
        story.append(Spacer(1, 6 * mm))
        if len(story) > 24:  # crude page-break guard between large sections
            story.append(PageBreak())

    if insights:
        story.append(Paragraph("Insights", styles["h2"]))
        for insight in insights:
            story.append(Paragraph(f"&bull;&nbsp; {insight}", styles["insight"]))
        story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("Generated by Mercy - personal finance management", styles["subtitle"]))
    doc.build(story)
    return buffer.getvalue()


def transactions_to_pdf(
    *,
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    subtitle: str,
) -> bytes:
    columns = [
        ("date", "Date"),
        ("type", "Type"),
        ("category", "Category"),
        ("description", "Description"),
        ("signed_amount", "Amount"),
    ]
    return report_to_pdf(
        title="Transaction statement",
        subtitle=subtitle,
        totals=summary,
        sections=[("Transactions", rows, columns)],
    )


def _prettify(key: str) -> str:
    return key.replace("_", " ").replace("pct", "%").title()
