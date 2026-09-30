"""Savings goal lifecycle and deadline reminders."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.core.constants import AlertSeverity, AlertType, GoalStatus
from app.core.logging import get_logger
from app.core.money import round_money
from app.core.time_utils import strip_tz, utc_now
from app.models.base import serialize_doc, to_object_id_or_none
from app.models.goal import goal_metrics
from app.repositories.goal_repo import GoalRepository
from app.services.alert_service import AlertService, goal_achieved_dedupe_key, goal_deadline_dedupe_key

logger = get_logger(__name__)

#: Raise a reminder this many days before a goal's deadline.
DEADLINE_WARNING_DAYS = (30, 14, 7, 1)
#: Raise a goal reminder when it is already this many days past due.
OVERDUE_GRACE_DAYS = 7


class GoalService:
    def __init__(self, goal_repo: GoalRepository, alert_service: AlertService) -> None:
        self.goals = goal_repo
        self.alerts = alert_service

    # ------------------------------------------------------------------ CRUD
    def create_goal(self, user_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
        now = strip_tz(utc_now())
        target_amount = round_money(payload["target_amount"])
        current = round_money(payload.get("current_amount") or 0.0)
        document = {
            "user_id": user_id,
            "name": payload["name"],
            "description": payload.get("description"),
            "target_amount": target_amount,
            "current_amount": current,
            "target_date": strip_tz(payload["target_date"]) if payload.get("target_date") else None,
            "status": GoalStatus.ACHIEVED.value if current >= target_amount else GoalStatus.ACTIVE.value,
            "color": payload.get("color"),
            "linked_category_id": to_object_id_or_none(payload.get("linked_category_id"), field="linked_category_id"),
            "achieved_at": now if current >= target_amount else None,
            **goal_metrics({"target_amount": target_amount, "current_amount": current,
                            "target_date": strip_tz(payload["target_date"]) if payload.get("target_date") else None,
                            "created_at": now}),
            "created_at": now,
            "updated_at": now,
        }
        self.goals.insert_one(document)
        if document["status"] == GoalStatus.ACHIEVED.value:
            self._raise_achieved(user_id, document)
        return document

    def update_goal(self, user_id: Any, goal_id: Any, changes: dict[str, Any]) -> dict[str, Any] | None:
        existing = self.goals.get_owned(goal_id, user_id)
        if existing is None:
            return None

        patch: dict[str, Any] = {}
        for field in ("name", "description", "color", "status"):
            if changes.get(field) is not None:
                patch[field] = str(changes[field]) if field == "status" else changes[field]
        for field in ("target_amount", "current_amount"):
            if changes.get(field) is not None:
                patch[field] = round_money(changes[field])
        if "target_date" in changes:
            patch["target_date"] = strip_tz(changes["target_date"]) if changes["target_date"] else None
        if "linked_category_id" in changes:
            patch["linked_category_id"] = to_object_id_or_none(
                changes["linked_category_id"], field="linked_category_id"
            )
        if not patch:
            return existing

        patch["updated_at"] = strip_tz(utc_now())
        updated = self.goals.find_one_and_update(
            {"_id": goal_id, "user_id": user_id}, {"$set": patch}, return_after=True
        )
        assert updated is not None
        self._refresh_metrics(updated)

        # Crossing the target in either direction flips the status automatically.
        target = float(updated.get("target_amount") or 0.0)
        current = float(updated.get("current_amount") or 0.0)
        if current >= target and updated.get("status") == GoalStatus.ACTIVE.value:
            self.goals.update_one(
                {"_id": goal_id}, {"$set": {"status": GoalStatus.ACHIEVED.value, "achieved_at": strip_tz(utc_now())}}
            )
            self._raise_achieved(user_id, updated)
        elif current < target and updated.get("status") == GoalStatus.ACHIEVED.value:
            self.goals.update_one(
                {"_id": goal_id}, {"$set": {"status": GoalStatus.ACTIVE.value, "achieved_at": None}}
            )
        return updated

    def delete_goal(self, user_id: Any, goal_id: Any) -> int:
        return self.goals.delete_one({"_id": goal_id, "user_id": user_id})

    def get_goal(self, user_id: Any, goal_id: Any) -> dict[str, Any] | None:
        goal = self.goals.get_owned(goal_id, user_id)
        if goal is None:
            return None
        self._refresh_metrics(goal)
        return goal

    def list_goals(
        self, user_id: Any, *, status: GoalStatus | None = None, include_achieved: bool = True
    ) -> list[dict[str, Any]]:
        rows = self.goals.list_for_user(user_id, status=status, include_achieved=include_achieved)
        for row in rows:
            self._refresh_metrics(row)
        return rows

    # ---------------------------------------------------------- contributions
    def contribute(self, user_id: Any, goal_id: Any, amount: float, note: str | None = None) -> dict[str, Any]:
        """Add (or, with a negative amount, withdraw from) a goal."""
        existing = self.goals.get_owned(goal_id, user_id)
        if existing is None:
            from app.core.errors import NotFoundError

            raise NotFoundError("Savings goal not found")
        if existing.get("status") == GoalStatus.CANCELLED.value:
            from app.core.errors import ValidationError

            raise ValidationError("Cannot contribute to a cancelled goal")

        delta = round_money(amount)
        updated = self.goals.apply_contribution(goal_id, delta)
        assert updated is not None
        self._refresh_metrics(updated)

        just_achieved = (
            updated["status"] == GoalStatus.ACHIEVED.value
            and existing.get("status") != GoalStatus.ACHIEVED.value
        )
        if just_achieved:
            self._raise_achieved(user_id, updated)

        return {
            "goal_id": str(goal_id),
            "amount": delta,
            "new_balance": updated["current_amount"],
            "progress_pct": updated.get("progress_pct", 0.0),
            "goal_achieved": just_achieved,
            "note": note,
            "created_at": utc_now(),
        }

    # -------------------------------------------------------------- reminders
    def check_deadlines(self, user_id: Any, *, as_of: datetime | None = None) -> list[dict[str, Any]]:
        """Raise deadline / overdue alerts. Idempotent via dedupe keys."""
        as_of = as_of or strip_tz(utc_now())
        horizon = max(DEADLINE_WARNING_DAYS) + OVERDUE_GRACE_DAYS
        created: list[dict[str, Any]] = []
        for goal in self.goals.due_soon(user_id, within_days=horizon, as_of=as_of):
            target_date = goal.get("target_date")
            if target_date is None:
                continue
            days_left = (strip_tz(target_date).date() - as_of.date()).days
            remaining = max(0.0, round_money(float(goal["target_amount"]) - float(goal["current_amount"])))

            if days_left < -OVERDUE_GRACE_DAYS:
                continue

            if days_left < 0:
                bucket = "overdue"
                severity = AlertSeverity.CRITICAL
                message = (
                    f"Goal '{goal['name']}' was due {-days_left} day(s) ago and is still "
                    f"{remaining:.2f} short of its {float(goal['target_amount']):.2f} target."
                )
            elif days_left in DEADLINE_WARNING_DAYS:
                bucket = f"d{days_left}"
                severity = AlertSeverity.WARNING if days_left <= 7 else AlertSeverity.INFO
                per_day = round_money(remaining / days_left) if days_left else remaining
                message = (
                    f"Goal '{goal['name']}' is due in {days_left} day(s). You need "
                    f"{per_day:.2f} per day to reach {float(goal['target_amount']):.2f}."
                )
            else:
                continue

            alert = self.alerts.create(
                user_id,
                type=AlertType.SAVINGS_DEADLINE,
                severity=severity,
                title=f"{goal['name']} - {abs(days_left)} day(s) {'overdue' if days_left < 0 else 'left'}",
                message=message,
                context={
                    "goal_id": str(goal["_id"]),
                    "amount": remaining,
                    "period_key": strip_tz(target_date).date().isoformat(),
                },
                dedupe_key=goal_deadline_dedupe_key(goal["_id"], target_date, bucket),
                triggered_at=as_of,
            )
            if alert:
                created.append(alert)
        return created

    def _raise_achieved(self, user_id: Any, goal: dict[str, Any]) -> dict[str, Any] | None:
        return self.alerts.create(
            user_id,
            type=AlertType.GOAL_ACHIEVED,
            severity=AlertSeverity.INFO,
            title=f"Goal achieved: {goal['name']}",
            message=(
                f"Congratulations - '{goal['name']}' reached its "
                f"{float(goal['target_amount']):.2f} target."
            ),
            context={"goal_id": str(goal["_id"]), "amount": goal.get("current_amount")},
            dedupe_key=goal_achieved_dedupe_key(goal["_id"]),
        )

    def _refresh_metrics(self, goal: dict[str, Any]) -> dict[str, Any]:
        """Recompute + persist denormalised progress fields."""
        metrics = goal_metrics(goal)
        goal.update(metrics)
        self.goals.update_one({"_id": goal["_id"]}, {"$set": metrics})
        return goal

    def summary(self, user_id: Any) -> dict[str, Any]:
        totals = self.goals.totals(user_id)
        counts = self.goals.status_counts(user_id)
        target = totals["target"]
        return {
            "active_goals": counts.get(GoalStatus.ACTIVE.value, 0),
            "achieved_goals": counts.get(GoalStatus.ACHIEVED.value, 0),
            "total_goals": sum(counts.values()),
            "total_target": target,
            "total_saved": totals["saved"],
            "total_remaining": totals["remaining"],
            "overall_progress_pct": round((totals["saved"] / target) * 100.0, 2) if target else 0.0,
        }

    def serialize(self, document: dict[str, Any] | None) -> dict[str, Any] | None:
        return serialize_doc(document)
