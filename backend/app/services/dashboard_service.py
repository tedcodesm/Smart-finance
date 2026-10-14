"""Dashboard assembly.

A single round-trip read model: pulls from every existing collection/service and
returns one `DashboardResponse` so the React app can render a screen without
fanning out to a dozen endpoints. No cache - the payload is cheap for the
indexed reads it performs.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from app.core.constants import BudgetPeriod, PredictionType, TransactionType
from app.core.logging import get_logger
from app.core.money import round_money, safe_div
from app.core.time_utils import month_key, period_window, resolve_range, strip_tz, utc_now
from app.models.dashboard import (
    CategorySpend,
    DashboardResponse,
    MoneyCard,
    NetWorthPoint,
    PredictionWidget,
)
from app.models.report import TimeBucket
from app.models.base import serialize_doc
from app.repositories.alert_repo import AlertRepository
from app.repositories.budget_repo import BudgetRepository
from app.repositories.category_repo import CategoryRepository
from app.repositories.goal_repo import GoalRepository
from app.repositories.prediction_repo import PredictionRepository
from app.repositories.transaction_repo import TransactionRepository
from app.repositories.user_repo import UserRepository
from app.services.alert_service import AlertService
from app.services.budget_service import BudgetService
from app.services.goal_service import GoalService

logger = get_logger(__name__)

CASH_FLOW_MONTHS = 8
RECENT_TRANSACTION_LIMIT = 6
TOP_CATEGORY_LIMIT = 5


class DashboardService:
    def __init__(
        self,
        *,
        transactions: TransactionRepository,
        categories: CategoryRepository,
        budgets: BudgetRepository,
        goals: GoalRepository,
        predictions: PredictionRepository,
        alerts: AlertService,
        users: UserRepository,
        budget_service: BudgetService,
        goal_service: GoalService,
    ) -> None:
        self.transactions = transactions
        self.categories = categories
        self.budgets = budgets
        self.goals = goals
        self.predictions = predictions
        self.alert_service = alerts
        self.users = users
        self.budget_service = budget_service
        self.goal_service = goal_service

    def build(self, user_id: Any, *, period: str = "monthly", as_of: datetime | None = None) -> DashboardResponse:
        as_of = strip_tz(as_of or utc_now())
        start, end, period_key = resolve_range(period, as_of)
        user = self.users.find_by_id(user_id) or {}
        user_curr = user.get("base_currency")
        currency = settings.default_currency if not user_curr or user_curr.upper() == "USD" else user_curr

        totals = self.transactions.totals(user_id, start=start, end=end)
        income = round_money(totals["income"])
        expense = round_money(totals["expense"])
        net = round_money(income - expense)
        count = int(totals["income_count"] + totals["expense_count"])
        savings_rate = round(safe_div(income - expense, income) * 100.0, 2) if income else 0.0

        previous_start = start - (end - start) - timedelta(microseconds=1)
        previous_totals = self.transactions.totals(user_id, start=previous_start, end=start - timedelta(microseconds=1))
        previous_net = round_money(previous_totals["income"] - previous_totals["expense"])

        cards = {
            "income": MoneyCard(
                label="Income this month",
                value=income,
                previous_value=round_money(previous_totals["income"]),
                currency=currency,
                trend="up" if income >= round_money(previous_totals["income"]) else "down",
            ),
            "expense": MoneyCard(
                label="Spending this month",
                value=expense,
                previous_value=round_money(previous_totals["expense"]),
                currency=currency,
                trend="down" if expense <= round_money(previous_totals["expense"]) else "up",
            ),
            "net": MoneyCard(
                label="Net this month",
                value=net,
                previous_value=previous_net,
                currency=currency,
                change_pct=_delta_pct(net, previous_net),
                trend="up" if net >= previous_net else "down",
            ),
            "savings_rate": MoneyCard(
                label="Savings rate",
                value=round_money(savings_rate),
                previous_value=round_money(safe_div(previous_totals["income"] - previous_totals["expense"],
                                                    previous_totals["income"]) * 100.0) if previous_totals["income"] else None,
                currency=currency,
            ),
        }

        cash_flow = self._cash_flow(user_id)
        net_worth = self._net_worth(cash_flow)
        top_categories = self._top_categories(user_id, start, end)
        budgets = self.budget_service.list_budgets(user_id, is_active=True, period=BudgetPeriod.MONTHLY, as_of=as_of)
        recent = self.transactions.list_for_user(user_id, sort=[("date", -1)], skip=0, limit=RECENT_TRANSACTION_LIMIT)
        category_names = self.categories.name_map(user_id)
        for row in recent:
            category = category_names.get(row.get("category_id"), {})
            row["category_name"] = category.get("name")
            row["category_color"] = category.get("color")
            row["signed_amount"] = (
                -float(row["amount"]) if row.get("type") == TransactionType.EXPENSE.value else float(row["amount"])
            )

        prediction_rows = self.predictions.latest_by_type(user_id)
        prediction_widgets = [_to_widget(doc) for doc in prediction_rows if _to_widget(doc) is not None]

        alert_rows, _ = self.alert_service.list_alerts(user_id, is_read=False, limit=5)
        goal_summary = self.goal_service.summary(user_id)

        insights = _insights(net, expense, budget_rows=budgets, prediction_rows=prediction_rows)

        return DashboardResponse(
            generated_at=utc_now(),
            currency=currency,
            period="monthly",
            period_key=period_key,
            cards=cards,
            cash_flow=cash_flow,
            net_worth=net_worth,
            top_categories=top_categories,
            budgets=[b for b in (self.budget_service.serialize(row) for row in budgets) if b],
            recent_transactions=[t for t in (serialize_doc(row) for row in recent) if t],
            predictions=prediction_widgets,
            alerts=[a for a in (self.alert_service.serialize(doc) for doc in alert_rows) if a],
            goals_summary=goal_summary,
            insights=insights,
        )

    # ------------------------------------------------------------- sections
    def _cash_flow(self, user_id: Any) -> list[TimeBucket]:
        rows = self.transactions.monthly_rollup(user_id, months=CASH_FLOW_MONTHS)
        return [TimeBucket(**row) for row in rows]

    def _net_worth(self, cash_flow: list[TimeBucket]) -> list[NetWorthPoint]:
        cumulative_income = 0.0
        cumulative_expense = 0.0
        points: list[NetWorthPoint] = []
        for bucket in cash_flow:
            cumulative_income = round_money(cumulative_income + bucket.income)
            cumulative_expense = round_money(cumulative_expense + bucket.expense)
            points.append(
                NetWorthPoint(
                    period_key=bucket.period_key,
                    cumulative_income=cumulative_income,
                    cumulative_expense=cumulative_expense,
                    balance=round_money(cumulative_income - cumulative_expense),
                )
            )
        return points

    def _top_categories(self, user_id: Any, start: datetime, end: datetime) -> list[CategorySpend]:
        rows = self.transactions.by_category(user_id, start=start, end=end, type=TransactionType.EXPENSE)
        names = self.categories.name_map(user_id)
        grand = sum(float(row["total"]) for row in rows) or 0.0

        span = end - start
        previous_rows = self.transactions.by_category(
            user_id,
            start=start - span - timedelta(microseconds=1),
            end=start - timedelta(microseconds=1),
            type=TransactionType.EXPENSE,
        )
        previous = {row["_id"]: float(row["total"]) for row in previous_rows}

        out: list[CategorySpend] = []
        for row in rows[:TOP_CATEGORY_LIMIT]:
            category = names.get(row["_id"], {})
            total = float(row["total"])
            prev = previous.get(row["_id"], 0.0)
            change = round(safe_div(total - prev, prev) * 100.0, 2) if prev > 0 else None
            out.append(
                CategorySpend(
                    category_id=str(row["_id"]) if row["_id"] else None,
                    category_name=category.get("name", "Uncategorised"),
                    category_color=category.get("color"),
                    total=round_money(total),
                    share_pct=round(safe_div(total, grand) * 100.0, 2),
                    change_pct=change,
                )
            )
        return out


def _delta_pct(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return round((current - previous) / previous * 100.0, 2)


def _to_widget(doc: dict[str, Any]) -> PredictionWidget | None:
    from app.models.base import serialize_doc

    ptype = doc.get("type")
    scope = doc.get("scope") or {}
    category_name = scope.get("category_name")
    value = float(doc.get("value") or 0.0)
    confidence = float(doc.get("confidence_score") or 0.0)

    try:
        prediction_type = PredictionType(ptype)
    except ValueError:
        return None

    headline: str
    detail: str | None = None
    action: dict[str, Any] | None = None
    if prediction_type is PredictionType.EXPENSE_FORECAST:
        headline = f"Next {category_name or 'spending'} projected"
        period_key = scope.get("period_key")
        detail = f"Forecast for {period_key} at {confidence * 100:.0f}% confidence."
    elif prediction_type is PredictionType.OVERSPEND_RISK:
        headline = f"Overspend risk: {category_name or 'budget'}"
        detail = f"{value * 100:.1f}% probability ({doc.get('label') or 'unknown'} risk)."
    elif prediction_type is PredictionType.BUDGET_RECOMMENDATION:
        headline = f"Suggested budget: {category_name or 'Overall'}"
        delta = (doc.get("extra") or {}).get("delta")
        detail = f"{delta:+.2f} vs. current limit." if delta is not None else None
        action = {"kind": "create_budget", "category_id": scope.get("category_id")}
    elif prediction_type is PredictionType.SAVINGS_RECOMMENDATION:
        headline = "Projected net savings"
        contribution = (doc.get("extra") or {}).get("suggested_contribution_total")
        detail = f"Suggested contribution: {contribution:.2f}." if contribution is not None else None
        action = {"kind": "open_goals"}
    else:
        headline = "Top spending habit"
        detail = doc.get("label")
    return PredictionWidget(
        type=prediction_type,
        headline=headline,
        value=value,
        confidence_score=confidence,
        detail=detail,
        category_name=category_name,
        generated_at=doc.get("generated_at"),
        prediction_id=str(doc.get("_id")) if doc.get("_id") else None,
        action=action,
    )


def _insights(net: float, expense: float, *, budget_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> list[str]:
    insights: list[str] = []
    if net > 0:
        insights.append(f"You are net positive this month by {net:.2f}.")
    elif expense > 0:
        insights.append("Spending has outpaced income this month.")
    over = [b for b in budget_rows if b.get("is_exceeded")]
    if over:
        insights.append(f"{len(over)} budget(s) exceeded: " + ", ".join(
            str(b.get("name") or b.get("category_name") or "Overall") for b in over[:3]) + ".")
    risk = [p for p in prediction_rows if p.get("type") == PredictionType.OVERSPEND_RISK.value
            and float(p.get("value") or 0.0) >= 0.6]
    if risk:
        insights.append(f"{len(risk)} budget(s) flagged as likely to overspend this month.")
    if not insights:
        insights.append("Add transactions to see monthly insights.")
    return insights