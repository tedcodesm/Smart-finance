"""Report generation.

All analytics are computed server-side from the `transactions` / `budgets` /
`savings_goals` collections using the denormalised time-bucket fields, so a
report is a handful of index-backed aggregations rather than a full scan.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable

from app.core.constants import BudgetPeriod, ReportType, TransactionType
from app.core.logging import get_logger
from app.core.money import add_money, percent_of, round_money, safe_div
from app.core.time_utils import (
    month_key,
    period_window,
    resolve_range,
    utc_now,
)
from app.models.base import serialize_doc, to_object_id_or_none
from app.models.report import (
    BudgetSlice,
    CategorySlice,
    ReportData,
    ReportTotals,
    TimeBucket,
    compute_savings_rate,
    trend_label,
)
from app.repositories.budget_repo import BudgetRepository, effective_window
from app.repositories.category_repo import CategoryRepository
from app.repositories.goal_repo import GoalRepository
from app.repositories.report_repo import ReportRepository
from app.repositories.transaction_repo import TransactionRepository
from app.services.budget_service import BudgetService
from app.services.goal_service import GoalService

logger = get_logger(__name__)

#: Bucket granularity used for the `timeline` series, keyed by report period.
TIMELINE_BUCKET = {"daily": "day", "weekly": "week_key", "monthly": "period_key", "yearly": "year"}


class ReportService:
    def __init__(
        self,
        transaction_repo: TransactionRepository,
        category_repo: CategoryRepository,
        budget_repo: BudgetRepository,
        goal_repo: GoalRepository,
        report_repo: ReportRepository,
        budget_service: BudgetService,
        goal_service: GoalService,
    ) -> None:
        self.transactions = transaction_repo
        self.categories = category_repo
        self.budgets = budget_repo
        self.goals = goal_repo
        self.reports = report_repo
        self.budget_service = budget_service
        self.goal_service = goal_service

    # ------------------------------------------------------------- dispatch
    def generate(self, user_id: Any, request: Any) -> ReportData:
        report_type = ReportType(request.report_type)
        period = request.period
        start, end, period_key = request.resolved_range()
        category_ids = (
            [to_object_id_or_none(cid, field="category_id") for cid in request.category_ids]
            if request.category_ids
            else None
        )

        builders = {
            ReportType.SPENDING_BY_CATEGORY: self._spending_by_category,
            ReportType.INCOME_VS_EXPENSE: self._income_vs_expense,
            ReportType.NET_CASH_FLOW: self._income_vs_expense,
            ReportType.BUDGET_PERFORMANCE: self._budget_performance,
            ReportType.GOAL_PROGRESS: self._goal_progress,
        }
        builder = builders[report_type]

        if report_type in (ReportType.BUDGET_PERFORMANCE, ReportType.GOAL_PROGRESS):
            data = builder(user_id, start=start, end=end, period_key=period_key, period=period, limit_top=request.limit_top)
        else:
            data = builder(
                user_id,
                start=start,
                end=end,
                period_key=period_key,
                period=period,
                category_ids=category_ids,
                limit_top=request.limit_top,
            )

        data.insights = self._insights(user_id, data)
        return data

    # ------------------------------------------------- spending by category
    def _spending_by_category(
        self,
        user_id: Any,
        *,
        start: datetime,
        end: datetime,
        period_key: str,
        period: str,
        category_ids: list[Any] | None,
        limit_top: int,
    ) -> ReportData:
        rows = self.transactions.by_category(user_id, start=start, end=end, type=TransactionType.EXPENSE)
        names = self.categories.name_map(user_id)
        grand_total = sum(float(row["total"]) for row in rows) or 0.0

        # Compare against the immediately preceding window of equal length.
        span = end - start
        previous_rows = self.transactions.by_category(
            user_id, start=start - span - timedelta(microseconds=1), end=start - timedelta(microseconds=1),
            type=TransactionType.EXPENSE,
        )
        previous: dict[Any, float] = {row["_id"]: float(row["total"]) for row in previous_rows}

        slices: list[CategorySlice] = []
        for row in rows:
            category = names.get(row["_id"], {})
            total = round_money(row["total"])
            prev = round_money(previous.get(row["_id"], 0.0))
            change = None
            if prev > 0:
                change = round(percent_of(total - prev, prev), 2)
            slices.append(
                CategorySlice(
                    category_id=str(row["_id"]) if row["_id"] else None,
                    category_name=category.get("name", "Uncategorised"),
                    category_color=category.get("color"),
                    total=total,
                    count=row["count"],
                    share_pct=round(percent_of(total, grand_total), 2),
                    average=round_money(safe_div(total, row["count"])),
                    previous_total=prev,
                    change_pct=change,
                    trend=trend_label(change),
                )
            )
        slices.sort(key=lambda s: s.total, reverse=True)

        totals = self._totals(user_id, start, end)
        return ReportData(
            report_type=ReportType.SPENDING_BY_CATEGORY,
            period=period,
            period_key=period_key,
            start_date=start,
            end_date=end,
            totals=totals,
            by_category=slices[:limit_top],
            timeline=self._timeline(user_id, start, end, period),
        )

    # ----------------------------------------------------- income vs expense
    def _income_vs_expense(
        self,
        user_id: Any,
        *,
        start: datetime,
        end: datetime,
        period_key: str,
        period: str,
        category_ids: list[Any] | None,
        limit_top: int,
    ) -> ReportData:
        totals = self._totals(user_id, start, end)
        timeline = self._timeline(user_id, start, end, period)
        data = ReportData(
            report_type=ReportType.INCOME_VS_EXPENSE,
            period=period,
            period_key=period_key,
            start_date=start,
            end_date=end,
            totals=totals,
            timeline=timeline,
        )
        if category_ids:
            rows = self.transactions.by_category(user_id, start=start, end=end, type=TransactionType.EXPENSE)
            names = self.categories.name_map(user_id)
            wanted = set(category_ids)
            grand = sum(float(r["total"]) for r in rows if r["_id"] in wanted) or 1.0
            data.by_category = [
                CategorySlice(
                    category_id=str(row["_id"]),
                    category_name=names.get(row["_id"], {}).get("name", "Uncategorised"),
                    category_color=names.get(row["_id"], {}).get("color"),
                    total=round_money(row["total"]),
                    count=row["count"],
                    share_pct=round(percent_of(float(row["total"]), grand), 2),
                    average=round_money(safe_div(float(row["total"]), row["count"])),
                )
                for row in rows
                if row["_id"] in wanted
            ][:limit_top]
        return data

    # ---------------------------------------------------- budget performance
    def _budget_performance(
        self,
        user_id: Any,
        *,
        start: datetime,
        end: datetime,
        period_key: str,
        period: str,
        limit_top: int,
    ) -> ReportData:
        rows = self.budget_service.list_budgets(user_id, is_active=None, as_of=end)
        names = self.categories.name_map(user_id)
        slices: list[BudgetSlice] = []
        for row in rows:
            window_start, window_end = effective_window(row, as_of=end)
            if window_start >= end or window_end <= start:
                continue
            category = names.get(row.get("category_id"), {})
            limit_amount = round_money(row.get("limit_amount") or 0.0)
            spent = round_money(row.get("spent_amount") or 0.0)
            slices.append(
                BudgetSlice(
                    budget_id=str(row["_id"]),
                    category_name=category.get("name") or row.get("name") or "Overall budget",
                    limit_amount=limit_amount,
                    spent_amount=spent,
                    usage_pct=row.get("usage_pct", 0.0),
                    remaining_amount=row.get("remaining_amount", 0.0),
                    pace_status=row.get("pace_status", "on_track"),
                )
            )
        slices.sort(key=lambda s: s.usage_pct, reverse=True)

        total_limit = add_money(*[s.limit_amount for s in slices])
        total_spent = add_money(*[s.spent_amount for s in slices])
        return ReportData(
            report_type=ReportType.BUDGET_PERFORMANCE,
            period=period,
            period_key=period_key,
            start_date=start,
            end_date=end,
            totals=ReportTotals(
                expense=total_spent,
                net=round_money(-total_spent),
                transaction_count=len(slices),
                savings_rate_pct=0.0,
            ),
            budgets=slices[:limit_top],
        )

    # ---------------------------------------------------------- goal progress
    def _goal_progress(
        self,
        user_id: Any,
        *,
        start: datetime,
        end: datetime,
        period_key: str,
        period: str,
        limit_top: int,
    ) -> ReportData:
        goals = self.goal_service.list_goals(user_id)
        summary = self.goal_service.summary(user_id)
        return ReportData(
            report_type=ReportType.GOAL_PROGRESS,
            period=period,
            period_key=period_key,
            start_date=start,
            end_date=end,
            totals=ReportTotals(
                income=summary["total_saved"],
                net=round_money(summary["total_remaining"]),
                transaction_count=summary["total_goals"],
            ),
            goals=_goal_payload(goals, limit_top),
            insights=[
                f"{summary['active_goals']} active goal(s), {summary['achieved_goals']} achieved.",
                f"Overall progress {summary['overall_progress_pct']}% of {summary['total_target']:.2f}.",
            ],
        )

    # ------------------------------------------------------------- internals
    def _totals(self, user_id: Any, start: datetime, end: datetime) -> ReportTotals:
        raw = self.transactions.totals(user_id, start=start, end=end)
        income = round_money(raw["income"])
        expense = round_money(raw["expense"])
        count = raw["income_count"] + raw["expense_count"]

        largest_category: str | None = None
        if raw["largest_expense"]:
            biggest = self.transactions.by_category(user_id, start=start, end=end, type=TransactionType.EXPENSE)
            if biggest:
                names = self.categories.name_map(user_id)
                largest_category = names.get(biggest[0]["_id"], {}).get("name")

        return ReportTotals(
            income=income,
            expense=expense,
            net=round_money(income - expense),
            transaction_count=count,
            savings_rate_pct=compute_savings_rate(income, expense),
            average_transaction=round_money(safe_div(income + expense, count)),
            largest_expense=round_money(raw["largest_expense"]) or None,
            largest_expense_category=largest_category,
        )

    def _timeline(self, user_id: Any, start: datetime, end: datetime, period: str) -> list[TimeBucket]:
        bucket = TIMELINE_BUCKET.get(period, "period_key")
        rows = self.transactions.by_period(user_id, start=start, end=end, bucket=bucket)

        merged: dict[str, TimeBucket] = {}
        order: list[str] = []
        for row in rows:
            key = str(row["_id"]["bucket"])
            if key not in merged:
                merged[key] = TimeBucket(period_key=key)
                order.append(key)
            bucket_model = merged[key]
            bucket_model.transaction_count += row["count"]
            if row["_id"]["type"] == TransactionType.INCOME.value:
                bucket_model.income = round_money(bucket_model.income + float(row["total"]))
            else:
                bucket_model.expense = round_money(bucket_model.expense + float(row["total"]))

        running = 0.0
        out: list[TimeBucket] = []
        for key in sorted(merged):
            bucket_model = merged[key]
            bucket_model.net = round_money(bucket_model.income - bucket_model.expense)
            running = round_money(running + bucket_model.net)
            bucket_model.cumulative_net = running
            out.append(bucket_model)
        return self._fill_timeline_gaps(out, start, end, period)

    def _fill_timeline_gaps(self, timeline: list[TimeBucket], start: datetime, end: datetime, period: str) -> list[TimeBucket]:
        """Insert zero-value buckets so charts do not skip months without spend."""
        if period not in ("monthly", "weekly", "yearly"):
            return timeline
        granularity = {"monthly": BudgetPeriod.MONTHLY, "weekly": BudgetPeriod.WEEKLY, "yearly": BudgetPeriod.YEARLY}[period]
        existing = {bucket.period_key: bucket for bucket in timeline}

        keys: list[str] = []
        cursor = start
        guard = 0
        while cursor <= end and guard < 500:
            _, _, key = period_window(granularity, cursor)
            keys.append(key)
            cursor = _advance(granularity, cursor)
            guard += 1

        out: list[TimeBucket] = []
        for key in keys:
            out.append(existing.get(key) or TimeBucket(period_key=key))
        return out

    def _insights(self, user_id: Any, data: ReportData) -> list[str]:
        """Short, human-readable takeaways rendered on the reports page."""
        insights: list[str] = []
        totals = data.totals

        if totals.income > 0:
            insights.append(
                f"You saved {totals.savings_rate_pct}% of your income "
                f"({totals.net:.2f} net across {totals.transaction_count} transactions)."
            )
        elif totals.expense > 0:
            insights.append(f"No income recorded; you spent {totals.expense:.2f} in this period.")

        if data.by_category:
            top = data.by_category[0]
            insights.append(
                f"{top.category_name} was your largest expense at {top.total:.2f} "
                f"({top.share_pct}% of the total)."
            )
            risers = [c for c in data.by_category if c.trend == "up" and c.previous_total]
            if risers:
                biggest = max(risers, key=lambda c: c.change_pct or 0.0)
                insights.append(
                    f"{biggest.category_name} rose {biggest.change_pct}% versus the previous period."
                )

        over = [b for b in data.budgets if b.usage_pct >= 100.0]
        if over:
            insights.append(
                f"{len(over)} budget(s) exceeded: " + ", ".join(b.category_name or "Overall" for b in over[:3]) + "."
            )
        warning = [b for b in data.budgets if 80.0 <= b.usage_pct < 100.0]
        if warning:
            insights.append(f"{len(warning)} budget(s) are above 80% of their limit.")

        if not insights:
            insights.append("No transactions in this period.")
        return insights

    # ------------------------------------------------------------- lifecycle
    def store(
        self, user_id: Any, data: ReportData, params: dict[str, Any]
    ) -> str:
        from app.core.config import settings

        generated = utc_now()
        return self.reports.store(
            user_id=user_id,
            report_type=data.report_type,
            period=data.period,
            period_start=data.start_date,
            period_end=data.end_date,
            period_key=data.period_key,
            params=params,
            data=data.model_dump(mode="json"),
            summary=data.totals.model_dump(mode="json"),
            expires_at=generated + timedelta(days=settings.report_retention_days),
        )

    def list_reports(
        self, user_id: Any, *, report_type: ReportType | None = None, skip: int = 0, limit: int = 25
    ) -> tuple[list[dict[str, Any]], int]:
        total = self.reports.count_for_user(user_id, report_type=report_type)
        rows = self.reports.list_for_user(user_id, report_type=report_type, skip=skip, limit=limit)
        return rows, total

    def get_report(self, user_id: Any, report_id: Any) -> dict[str, Any] | None:
        return self.reports.get_owned(report_id, user_id)

    def delete_report(self, user_id: Any, report_id: Any) -> int:
        return self.reports.delete_owned(report_id, user_id)

    def purge_expired(self) -> int:
        return self.reports.purge_expired()

    def serialize(self, document: dict[str, Any] | None) -> dict[str, Any] | None:
        return serialize_doc(document)

    # --------------------------------------------------- raw series for export
    def transaction_rows(
        self,
        user_id: Any,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        type: TransactionType | None = None,
        category_ids: Iterable[Any] | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        """Flat rows for CSV export (joined with category names)."""
        names = self.categories.name_map(user_id)
        rows = self.transactions.list_for_user(
            user_id,
            start=start,
            end=end,
            type=type,
            category_ids=category_ids,
            sort=[("date", 1)],
            limit=limit,
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            category = names.get(row.get("category_id"), {})
            amount = float(row.get("amount") or 0.0)
            out.append(
                {
                    "date": row["date"].date().isoformat(),
                    "type": row.get("type"),
                    "category": category.get("name", "Uncategorised"),
                    "description": row.get("description") or "",
                    "amount": amount,
                    "signed_amount": -amount if row.get("type") == TransactionType.EXPENSE.value else amount,
                    "is_recurring": "yes" if row.get("is_recurring") else "no",
                }
            )
        return out

    def comparison_period(self, period: str, period_key: str) -> tuple[datetime, datetime, str]:
        """The window immediately preceding the requested one, for deltas."""
        anchor = _key_to_anchor(period, period_key)
        previous_key = _previous_key(period, period_key)
        p_start, p_end, _ = period_window(BudgetPeriod(period), anchor)
        p_start, p_end, _ = period_window(BudgetPeriod(period), _key_to_anchor(period, previous_key))
        return p_start, p_end - timedelta(microseconds=1), previous_key


def _advance(granularity: BudgetPeriod, cursor: datetime) -> datetime:
    if granularity is BudgetPeriod.WEEKLY:
        return cursor + timedelta(days=7)
    if granularity is BudgetPeriod.YEARLY:
        return cursor.replace(year=cursor.year + 1)
    return (cursor.replace(day=28) + timedelta(days=7)).replace(day=1)


def _goal_payload(goals: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [serialize_doc(goal) or {} for goal in goals[:limit]]


def _key_to_anchor(period: str, period_key: str) -> datetime:
    """Map a bucket key back to a representative date inside that bucket."""
    if period == BudgetPeriod.YEARLY.value:
        return datetime(int(period_key), 6, 1)
    if period == BudgetPeriod.WEEKLY.value:
        from datetime import date

        year_s, week_s = period_key.split("-W")
        return datetime.combine(date.fromisocalendar(int(year_s), int(week_s), 1), datetime.min.time())
    if period == "daily":
        return datetime.strptime(period_key, "%Y-%m-%d")
    return datetime.strptime(period_key, "%Y-%m")


def _previous_key(period: str, period_key: str) -> str:
    """Bucket key of the period before `period_key`."""
    if period == BudgetPeriod.YEARLY.value:
        return f"{int(period_key) - 1:04d}"
    if period == BudgetPeriod.WEEKLY.value:
        from datetime import date, timedelta

        year_s, week_s = period_key.split("-W")
        monday = date.fromisocalendar(int(year_s), int(week_s), 1) - timedelta(days=7)
        iso = monday.isocalendar()
        return f"{iso.year:04d}-W{iso.week:02d}"
    year_s, month_s = period_key.split("-")
    year, month = int(year_s), int(month_s)
    return month_key(datetime(year - 1, 12, 1) if month == 1 else datetime(year, month - 1, 1))


def resolve_report_range(period: str, period_key: str | None = None) -> tuple[datetime, datetime, str]:
    if period_key:
        start, end, key = period_window(BudgetPeriod(period), _key_to_anchor(period, period_key))
        return start, end - timedelta(microseconds=1), key
    return resolve_range(period, None)  # type: ignore[arg-type]
