"""Repository for `savings_goals`."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from app.core.constants import GoalStatus
from app.repositories.base import BaseRepository


class GoalRepository(BaseRepository):
    collection_name = "savings_goals"

    def list_for_user(
        self,
        user_id: Any,
        *,
        status: GoalStatus | str | None = None,
        include_achieved: bool = True,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"user_id": user_id}
        if status is not None:
            query["status"] = str(status)
        elif not include_achieved:
            query["status"] = {"$ne": GoalStatus.ACHIEVED.value}
        return self.find(query, sort=[("status", 1), ("target_date", 1), ("created_at", -1)])

    def get_owned(self, goal_id: Any, user_id: Any) -> dict[str, Any] | None:
        return self.find_one({"_id": goal_id, "user_id": user_id})

    def due_soon(
        self, user_id: Any, *, within_days: int, as_of: datetime
    ) -> list[dict[str, Any]]:
        """Active goals whose deadline lands inside the reminder horizon."""
        return self.find(
            {
                "user_id": user_id,
                "status": GoalStatus.ACTIVE.value,
                "target_date": {"$ne": None, "$lte": as_of + timedelta(days=within_days)},
            },
            sort=[("target_date", 1)],
        )

    def status_counts(self, user_id: Any) -> dict[str, int]:
        rows = self.aggregate([{"$match": {"user_id": user_id}}, {"$group": {"_id": "$status", "n": {"$sum": 1}}}])
        return {row["_id"]: row["n"] for row in rows}

    def totals(self, user_id: Any) -> dict[str, float]:
        rows = self.aggregate(
            [
                {"$match": {"user_id": user_id, "status": GoalStatus.ACTIVE.value}},
                {"$group": {"_id": None, "target": {"$sum": "$target_amount"}, "saved": {"$sum": "$current_amount"}}},
            ]
        )
        if not rows:
            return {"target": 0.0, "saved": 0.0, "remaining": 0.0}
        target = float(rows[0].get("target") or 0.0)
        saved = float(rows[0].get("saved") or 0.0)
        return {"target": target, "saved": saved, "remaining": max(0.0, round(target - saved, 2))}

    def apply_contribution(self, goal_id: Any, delta: float) -> dict[str, Any] | None:
        """Atomically adjust `current_amount` (never below zero)."""
        from app.core.money import round_money
        from app.core.time_utils import utc_now

        goal = self.get_any(goal_id)
        if not goal:
            return None
        new_balance = round_money(max(0.0, float(goal.get("current_amount") or 0.0) + float(delta)))
        patch: dict[str, Any] = {"current_amount": new_balance, "updated_at": utc_now()}
        if new_balance >= float(goal.get("target_amount") or 0.0):
            patch["status"] = GoalStatus.ACHIEVED.value
            patch["achieved_at"] = utc_now()
        elif goal.get("status") == GoalStatus.ACHIEVED.value:
            patch["status"] = GoalStatus.ACTIVE.value
            patch["achieved_at"] = None
        return self.find_one_and_update({"_id": goal_id}, {"$set": patch}, return_after=True)

    def get_any(self, goal_id: Any) -> dict[str, Any] | None:
        return self.find_one({"_id": goal_id})

    def delete_all_for_user(self, user_id: Any) -> int:
        return self.delete_many({"user_id": user_id})
