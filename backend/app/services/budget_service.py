"""Budget lifecycle, live spend tracking, and threshold alert evaluation.

`evaluate_budgets` is the single entry point used by the transaction-write path
(requirement 5: *new transaction -> budget check -> Alerts*) and by the nightly
sweep. It is idempotent: repeated calls for the same period produce no new
alerts thanks to deterministic `dedupe_key`s.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.core.config import settings
from app.core.constants import AlertSeverity, AlertType, BudgetPeriod, TransactionType
from app.core.logging import get_logger
from app.core.money import percent_of, round_money, safe_div
from app.core.time_utils import (
    days_in_period,
    period_elapsed_fraction,
    period_window,
    strip_tz,
    utc_now,
)
from app.models.budget import compute_pace_status
from app.models.base import serialize_doc
from app.repositories.budget_repo import BudgetRepository, effective_window
from app.repositories.category_repo import CategoryRepository
from app.repositories.transaction_repo import TransactionRepository
from app.services.alert_service import (
    AlertService,
    budget_exceeded_dedupe_key,
    budget_threshold_dedupe_key,
)

logger = get_logger(__name__)

#: Fraction buckets used to build a stable threshold dedupe key. Without this,
#: every transaction past the threshold would spawn a new alert.
THRESHOLD_BUCKETS = (0.5, 0.75, 0.8, 0.9, 0.95, 1.0, 1.1, 1.25, 1.5)


class BudgetService:
    def __init__(
        self,
        budget_repo: BudgetRepository,
        transaction_repo: TransactionRepository,
        category_repo: CategoryRepository,
        alert_service: AlertService,
    ) -> None:
        self.budgets = budget_repo
        self.transactions = transaction_repo
        self.categories = category_repo
        self.alerts = alert_service

    # ------------------------------------------------------------------ CRUD
    def create_budget(self, user_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
        from app.models.base import to_object_id_or_none

        category_id = to_object_id_or_none(payload.get("category_id"), field="category_id")
        if category_id is not None:
            category = self.categories.get_owned(category_id, user_id)
            if category is None:
                from app.core.errors import ValidationError

                raise ValidationError("category_id does not reference a category you can use")
            if category.get("type") != TransactionType.EXPENSE.value:
                from app.core.errors import ValidationError

                raise ValidationError("Budgets can only be set on expense categories")
        elif payload.get("category_id"):
            from app.core.errors import ValidationError

            raise ValidationError("Invalid category_id")

        period = BudgetPeriod(payload.get("period", BudgetPeriod.MONTHLY.value))
        now = strip_tz(utc_now())
        start_date = payload.get("start_date") or now
        _, _, period_key = period_window(period, start_date)

        if category_id is not None and self.budgets.has_active_for(user_id, category_id, period):
            from app.core.errors import ConflictError

            raise ConflictError(
                f"An active {period.value} budget already exists for this category"
            )

        document = {
            "user_id": user_id,
            "category_id": category_id,
            "name": payload.get("name"),
            "limit_amount": round_money(payload["limit_amount"]),
            "period": period.value,
            "start_date": start_date,
            "end_date": payload.get("end_date"),
            "period_key": period_key,
            "alert_threshold_pct": float(
                payload.get("alert_threshold_pct") or settings.budget_alert_threshold_pct
            ),
            "rollover": bool(payload.get("rollover", False)),
            "is_active": bool(payload.get("is_active", True)),
            "spent_amount": 0.0,
            "last_alerted_at": None,
            "created_at": now,
            "updated_at": now,
        }
        self.budgets.insert_one(document)
        # Seed the cached spend from history so a newly created budget is not
        # reported as untouched when transactions already exist in its window.
        document["spent_amount"] = self._spend_for(user_id, document, as_of=now)
        return document

    def update_budget(self, user_id: Any, budget_id: Any, changes: dict[str, Any]) -> dict[str, Any] | None:
        from app.models.base import to_object_id_or_none

        existing = self.budgets.get_owned(budget_id, user_id)
        if existing is None:
            return None

        patch: dict[str, Any] = {}
        for field in ("limit_amount", "name", "rollover", "is_active", "start_date", "end_date"):
            if field in changes and changes[field] is not None:
                patch[field] = changes[field]
        if "alert_threshold_pct" in changes and changes["alert_threshold_pct"] is not None:
            patch["alert_threshold_pct"] = float(changes["alert_threshold_pct"])
        if "period" in changes and changes["period"] is not None:
            patch["period"] = BudgetPeriod(changes["period"]).value
        if "category_id" in changes:
            patch["category_id"] = to_object_id_or_none(changes["category_id"], field="category_id")

        if not patch:
            return existing

        anchor = patch.get("start_date") or existing.get("start_date")
        period = BudgetPeriod(patch.get("period") or existing.get("period"))
        _, _, period_key = period_window(period, anchor)
        patch["period_key"] = period_key
        patch["updated_at"] = strip_tz(utc_now())

        updated = self.budgets.find_one_and_update(
            {"_id": budget_id, "user_id": user_id}, {"$set": patch}, return_after=True
        )
        if updated is not None:
            updated["spent_amount"] = self._spend_for(user_id, updated)
            self.budgets.set_spent_amount(budget_id, updated["spent_amount"])
        return updated

    def delete_budget(self, user_id: Any, budget_id: Any) -> int:
        return self.budgets.delete_one({"_id": budget_id, "user_id": user_id})

    def get_budget(
        self, user_id: Any, budget_id: Any, *, as_of: datetime | None = None
    ) -> dict[str, Any] | None:
        """One budget with spend and every derived field recomputed for `as_of`.

        The stored document only caches `spent_amount`; the presentation fields
        (`usage_pct`, `is_exceeded`, `pace_status`, ...) are always re-derived so
        single-item reads agree with `list_budgets` and cannot go stale.
        """
        budget = self.budgets.get_owned(budget_id, user_id)
        if budget is None:
            return None
        as_of = as_of or strip_tz(utc_now())
        budget["spent_amount"] = self._spend_for(user_id, budget, as_of=as_of)
        self._decorate(
            user_id, budget, as_of=as_of, category_names=self.categories.name_map(user_id)
        )
        return budget

    def list_budgets(
        self,
        user_id: Any,
        *,
        is_active: bool | None = True,
        period: BudgetPeriod | str | None = None,
        as_of: datetime | None = None,
        with_spend: bool = True,
    ) -> list[dict[str, Any]]:
        as_of = as_of or strip_tz(utc_now())
        rows = self.budgets.list_for_user(user_id, is_active=is_active, period=period)
        category_names = self.categories.name_map(user_id)
        for row in rows:
            if with_spend:
                row["spent_amount"] = self._spend_for(user_id, row, as_of=as_of)
            self._decorate(user_id, row, as_of=as_of, category_names=category_names)
        return rows

    # --------------------------------------------------------------- progress
    def _spend_for(self, user_id: Any, budget: dict[str, Any], *, as_of: datetime | None = None) -> float:
        start, end = effective_window(budget, as_of=as_of or strip_tz(utc_now()))
        return round_money(
            self.transactions.spend_for_budget(
                user_id, start=start, end=end, category_id=budget.get("category_id")
            )
        )

    def _decorate(
        self,
        user_id: Any,
        budget: dict[str, Any],
        *,
        as_of: datetime,
        category_names: dict[Any, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Attach presentation fields (window, usage, pace) in place."""
        period = BudgetPeriod(budget.get("period", BudgetPeriod.MONTHLY.value))
        start, end = effective_window(budget, as_of=as_of)
        _, _, period_key = period_window(period, as_of)

        limit = float(budget.get("limit_amount") or 0.0)
        spent = float(budget.get("spent_amount") or 0.0)
        elapsed = period_elapsed_fraction(period, as_of)
        remaining_days = max(0, (end.date() - as_of.date()).days)

        budget["period_key"] = period_key
        budget["remaining_amount"] = round_money(max(0.0, limit - spent))
        budget["usage_pct"] = round(percent_of(spent, limit), 2)
        budget["days_remaining"] = remaining_days
        budget["period_elapsed_pct"] = round(elapsed * 100.0, 2)
        budget["is_exceeded"] = limit > 0 and spent >= limit
        budget["is_threshold_crossed"] = limit > 0 and spent >= limit * float(
            budget.get("alert_threshold_pct") or settings.budget_alert_threshold_pct
        )
        budget["projected_total"] = (
            round_money(safe_div(spent, elapsed, default=spent)) if elapsed > 0.05 and spent > 0 else None
        )
        budget["pace_status"] = compute_pace_status(budget["usage_pct"], elapsed * 100.0, limit, spent)

        category = (category_names or {}).get(budget.get("category_id")) or {}
        if category:
            budget["category_name"] = category.get("name")
            budget["category_color"] = category.get("color")
        if not budget.get("name"):
            budget["name"] = budget.get("category_name") or "Overall budget"
        _ = user_id
        return budget

    # ---------------------------------------------------------- alert engine
    def evaluate_budgets(
        self,
        user_id: Any,
        *,
        as_of: datetime | None = None,
        categories: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Recompute spend and raise threshold / exceeded alerts.

        Called on every expense write and by the nightly sweep. Returns the alerts
        that were newly created (duplicates are suppressed by the alert service).
        """
        as_of = as_of or strip_tz(utc_now())
        created: list[dict[str, Any]] = []
        budgets = self.budgets.live_at(user_id, as_of)
        if categories is not None:
            wanted = set(categories)
            budgets = [b for b in budgets if b.get("category_id") in wanted]
        if not budgets:
            return created

        category_names = self.categories.name_map(user_id)
        for budget in budgets:
            budget["spent_amount"] = self._spend_for(user_id, budget, as_of=as_of)
            self._decorate(user_id, budget, as_of=as_of, category_names=category_names)
            alert = self._alert_for(user_id, budget, as_of=as_of)
            if alert:
                created.append(alert)
        return created

    def _alert_for(self, user_id: Any, budget: dict[str, Any], *, as_of: datetime) -> dict[str, Any] | None:
        limit = float(budget.get("limit_amount") or 0.0)
        if limit <= 0:
            return None

        spent = float(budget.get("spent_amount") or 0.0)
        usage = safe_div(spent, limit)
        threshold = float(budget.get("alert_threshold_pct") or settings.budget_alert_threshold_pct)
        period_key = budget.get("period_key") or "unknown"
        category_name = budget.get("category_name") or "Overall budget"
        period_label = BudgetPeriod(budget.get("period", "monthly")).value

        if usage >= 1.0:
            over = round_money(spent - limit)
            alert = self.alerts.create(
                user_id,
                type=AlertType.BUDGET_EXCEEDED,
                severity=AlertSeverity.CRITICAL,
                title=f"{category_name} budget exceeded",
                message=(
                    f"You have spent {spent:.2f} against a {limit:.2f} {period_label} budget for "
                    f"{category_name} - {over:.2f} over the limit."
                ),
                context={
                    "category_id": str(budget["category_id"]) if budget.get("category_id") else None,
                    "category_name": category_name,
                    "budget_id": str(budget["_id"]),
                    "period_key": period_key,
                    "amount": spent,
                    "threshold_pct": threshold,
                },
                dedupe_key=budget_exceeded_dedupe_key(budget["_id"], period_key),
                triggered_at=as_of,
            )
            if alert:
                self.budgets.mark_alerted(budget["_id"], as_of)
            return alert

        if usage >= threshold:
            bucket = next((b for b in THRESHOLD_BUCKETS if usage >= b), threshold)
            remaining = round_money(limit - spent)
            days = budget.get("days_remaining", 0)
            pace = budget.get("pace_status", "on_track")
            alert = self.alerts.create(
                user_id,
                type=AlertType.BUDGET_THRESHOLD,
                severity=AlertSeverity.WARNING if pace != "on_track" else AlertSeverity.INFO,
                title=f"{category_name} budget at {usage * 100:.0f}%",
                message=(
                    f"You have used {usage * 100:.0f}% of your {limit:.2f} {period_label} budget for "
                    f"{category_name}. {remaining:.2f} left with {days} day(s) to go."
                ),
                context={
                    "category_id": str(budget["category_id"]) if budget.get("category_id") else None,
                    "category_name": category_name,
                    "budget_id": str(budget["_id"]),
                    "period_key": period_key,
                    "amount": spent,
                    "threshold_pct": threshold,
                },
                dedupe_key=budget_threshold_dedupe_key(budget["_id"], period_key, f"{bucket:.2f}"),
                triggered_at=as_of,
            )
            if alert:
                self.budgets.mark_alerted(budget["_id"], as_of)
            return alert

        return None

    # ------------------------------------------------------------ projections
    def project_period(
        self, user_id: Any, budget: dict[str, Any], *, as_of: datetime | None = None
    ) -> dict[str, Any]:
        """Break a single budget down to its period timeline (for the UI chart)."""
        as_of = as_of or strip_tz(utc_now())
        period = BudgetPeriod(budget.get("period", BudgetPeriod.MONTHLY.value))
        _, _, period_key = period_window(period, as_of)
        bucket = "week_key" if period is BudgetPeriod.WEEKLY else "period_key"
        start, _ = effective_window(budget, as_of=as_of)

        if budget.get("category_id") is not None:
            series = self._scoped_series(user_id, budget, as_of=as_of, bucket=bucket, start=start, end=as_of)
        else:
            series = {
                str(row["_id"]["bucket"]): round_money(row["total"])
                for row in self.transactions.by_period(
                    user_id,
                    start=start,
                    end=as_of,
                    bucket=bucket,
                    type=TransactionType.EXPENSE,
                )
            }

        return {
            "period": period.value,
            "period_key": period_key,
            "series": dict(sorted(series.items())),
            "limit_amount": round_money(budget.get("limit_amount") or 0.0),
            "total": round_money(sum(series.values())),
        }

    def _scoped_series(
        self,
        user_id: Any,
        budget: dict[str, Any],
        *,
        as_of: datetime,
        bucket: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, float]:
        pipeline = [
            {
                "$match": {
                    "user_id": user_id,
                    "type": TransactionType.EXPENSE.value,
                    "category_id": budget["category_id"],
                    "date": {"$gte": start, "$lte": end},
                }
            },
            {"$group": {"_id": f"${bucket}", "total": {"$sum": "$amount"}}},
        ]
        return {str(row["_id"]): round_money(row["total"]) for row in self.transactions.aggregate(pipeline)}

    # ---------------------------------------------------------------- sweeps
    def roll_periods(self, user_id: Any, *, as_of: datetime | None = None) -> int:
        """Reset cached `spent_amount` when a budget's period has rolled over."""
        as_of = as_of or strip_tz(utc_now())
        changed = 0
        for period in BudgetPeriod:
            changed += len(self.budgets.roll_period(user_id, period, as_of))
        return changed

    def period_length(self, period: BudgetPeriod | str, period_key: str) -> int:
        return days_in_period(period, period_key)

    def serialize(self, document: dict[str, Any]) -> dict[str, Any]:
        return serialize_doc(document) or {}
