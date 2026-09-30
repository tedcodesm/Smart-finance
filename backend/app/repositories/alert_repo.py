"""Repository for `alerts`.

`dedupe_key` is always populated by the alert service so that the sparse unique
index on `(user_id, dedupe_key)` provides idempotent alert creation - re-running
a budget check or the nightly sweep cannot duplicate a notification.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from app.core.constants import AlertSeverity, AlertType
from app.repositories.base import BaseRepository


class AlertRepository(BaseRepository):
    collection_name = "alerts"

    def list_for_user(
        self,
        user_id: Any,
        *,
        is_read: bool | None = None,
        types: Iterable[AlertType | str] | None = None,
        severities: Iterable[AlertSeverity | str] | None = None,
        since: datetime | None = None,
        skip: int = 0,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        return self.find(
            self._filter(user_id, is_read=is_read, types=types, severities=severities, since=since),
            sort=[("triggered_at", -1)],
            skip=skip,
            limit=limit,
        )

    def count_for_user(
        self,
        user_id: Any,
        *,
        is_read: bool | None = None,
        types: Iterable[AlertType | str] | None = None,
        severities: Iterable[AlertSeverity | str] | None = None,
        since: datetime | None = None,
    ) -> int:
        return self.count(self._filter(user_id, is_read=is_read, types=types, severities=severities, since=since))

    @staticmethod
    def _filter(
        user_id: Any,
        *,
        is_read: bool | None = None,
        types: Iterable[AlertType | str] | None = None,
        severities: Iterable[AlertSeverity | str] | None = None,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {"user_id": user_id}
        if is_read is not None:
            query["is_read"] = is_read
        if types:
            query["type"] = {"$in": [str(t) for t in types]}
        if severities:
            query["severity"] = {"$in": [str(s) for s in severities]}
        if since is not None:
            query["triggered_at"] = {"$gte": since}
        return query

    def counts(self, user_id: Any, *, since: datetime | None = None) -> dict[str, Any]:
        base = {"user_id": user_id}
        if since is not None:
            base["triggered_at"] = {"$gte": since}
        by_type = {
            row["_id"]: row["n"]
            for row in self.aggregate([{"$match": base}, {"$group": {"_id": "$type", "n": {"$sum": 1}}}])
        }
        by_sev = {
            row["_id"]: row["n"]
            for row in self.aggregate([{"$match": base}, {"$group": {"_id": "$severity", "n": {"$sum": 1}}}])
        }
        unread = self.count({**base, "is_read": False})
        total = self.count(base)
        return {
            "total": total,
            "unread": unread,
            "critical": by_sev.get(AlertSeverity.CRITICAL.value, 0),
            "warning": by_sev.get(AlertSeverity.WARNING.value, 0),
            "info": by_sev.get(AlertSeverity.INFO.value, 0),
            "by_type": by_type,
        }

    def mark_read(
        self,
        user_id: Any,
        *,
        alert_ids: Iterable[Any] | None = None,
        all_alerts: bool = False,
        types: Iterable[AlertType | str] | None = None,
        severities: Iterable[AlertSeverity | str] | None = None,
        read: bool = True,
        read_at: datetime | None = None,
    ) -> int:
        query: dict[str, Any] = {"user_id": user_id, "is_read": {"$ne": read}}
        if alert_ids:
            query["_id"] = {"$in": list(alert_ids)}
        if types:
            query["type"] = {"$in": [str(t) for t in types]}
        if severities:
            query["severity"] = {"$in": [str(s) for s in severities]}
        if not alert_ids and not all_alerts and not types and not severities:
            return 0
        patch: dict[str, Any] = {"is_read": read}
        patch["read_at"] = read_at if read else None
        return self.update_many(query, {"$set": patch})

    def delete_for_user(self, user_id: Any, alert_ids: Iterable[Any]) -> int:
        if not alert_ids:
            return 0
        return self.delete_many({"user_id": user_id, "_id": {"$in": list(alert_ids)}})

    def delete_all_for_user(self, user_id: Any) -> int:
        return self.delete_many({"user_id": user_id})

    def purge_expired(self, as_of: datetime | None = None) -> int:
        from app.core.time_utils import utc_now

        return self.delete_many({"expires_at": {"$ne": None, "$lte": as_of or utc_now()}})
