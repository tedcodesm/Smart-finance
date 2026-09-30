"""Repository for `budgets`."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from app.core.constants import BudgetPeriod
from app.core.time_utils import period_window, strip_tz, utc_now
from app.repositories.base import BaseRepository


class BudgetRepository(BaseRepository):
    collection_name = "budgets"

    # ------------------------------------------------------------------ reads
    def list_for_user(
        self,
        user_id: Any,
        *,
        is_active: bool | None = True,
        period: BudgetPeriod | str | None = None,
        category_id: Any | None = None,
        category_ids: Iterable[Any] | None = None,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"user_id": user_id}
        if is_active is not None:
            query["is_active"] = is_active
        if period is not None:
            query["period"] = str(period)
        if category_id is not None:
            query["category_id"] = category_id
        if category_ids is not None:
            query["category_id"] = {"$in": list(category_ids)}
        return self.find(query, sort=[("period", 1), ("category_id", 1), ("start_date", -1)])

    def get_owned(self, budget_id: Any, user_id: Any) -> dict[str, Any] | None:
        return self.find_one({"_id": budget_id, "user_id": user_id})

    def live_at(self, user_id: Any, as_of: datetime) -> list[dict[str, Any]]:
        """Every active budget whose window contains `as_of`, across all periods."""
        return self.find(
            {
                "user_id": user_id,
                "is_active": True,
                "start_date": {"$lte": as_of},
                "$or": [{"end_date": None}, {"end_date": {"$gt": as_of}}],
            },
            sort=[("period", 1), ("category_id", 1)],
        )

    def overlapping(
        self,
        user_id: Any,
        *,
        category_id: Any | None,
        period: BudgetPeriod | str,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        """Active budgets whose window intersects `[start, end)` for one category.

        `category_id=None` matches the overall (all-category) budget only.
        """
        return self.find(
            {
                "user_id": user_id,
                "is_active": True,
                "period": str(period),
                "category_id": category_id,
                "start_date": {"$lt": end},
                "$or": [{"end_date": None}, {"end_date": {"$gt": start}}],
            }
        )

    def has_active_for(self, user_id: Any, category_id: Any, period: BudgetPeriod | str) -> bool:
        return self.exists(
            {"user_id": user_id, "category_id": category_id, "period": str(period), "is_active": True}
        )

    def list_for_category_period(
        self, user_id: Any, *, category_id: Any, period: BudgetPeriod | str
    ) -> list[dict[str, Any]]:
        return self.find({"user_id": user_id, "category_id": category_id, "period": str(period)})

    # ----------------------------------------------------------------- writes
    def set_spent_amount(self, budget_id: Any, spent_amount: float) -> None:
        self.update_one({"_id": budget_id}, {"$set": {"spent_amount": float(spent_amount)}})

    def mark_alerted(self, budget_id: Any, when: datetime) -> None:
        self.update_one({"_id": budget_id}, {"$set": {"last_alerted_at": when}})

    def roll_period(self, user_id: Any, period: BudgetPeriod | str, as_of: datetime) -> list[dict[str, Any]]:
        """Advance the cached `period_key`/`spent_amount` of live budgets.

        Called by the nightly sweep so a monthly budget resets on the 1st without
        requiring a write to the transactions collection.
        """
        _, _, period_key = period_window(period, as_of)
        updated: list[dict[str, Any]] = []
        for budget in self.live_at(user_id, as_of):
            if budget.get("period") != str(period) or budget.get("period_key") == period_key:
                continue
            result = self.find_one_and_update(
                {"_id": budget["_id"]},
                {"$set": {"period_key": period_key, "spent_amount": 0.0}},
                return_after=True,
            )
            if result:
                updated.append(result)
        return updated

    def reset_spent_for_category(self, user_id: Any, category_id: Any) -> int:
        return self.update_many(
            {"user_id": user_id, "category_id": category_id, "is_active": True},
            {"$set": {"spent_amount": 0.0}},
        )

    def delete_all_for_user(self, user_id: Any) -> int:
        return self.delete_many({"user_id": user_id})


def effective_window(budget: dict[str, Any], *, as_of: datetime | None = None) -> tuple[datetime, datetime]:
    """Resolve a budget's `[start, end)` window for the period containing `as_of`.

    The window is always the whole calendar period (the month for `monthly`, the
    ISO week for `weekly`, ...), matching `period_key` and the dashboard
    aggregation. `start_date` only decides *when the budget becomes active* (see
    `live_at`), so it deliberately does not clip the window: a budget created on
    the 26th must still account for the whole month it belongs to, otherwise
    `spent_amount`, `usage_pct` and the resulting alerts silently under-report.
    An explicit `end_date` is the only hard boundary, and it shortens the final
    period.
    """
    period = budget.get("period", BudgetPeriod.MONTHLY.value)
    anchor = as_of or budget.get("start_date") or strip_tz(utc_now())
    start, end, _ = period_window(period, anchor)

    explicit_end = budget.get("end_date")
    if explicit_end:
        end = min(end, strip_tz(explicit_end))
    return start, end
