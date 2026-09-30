"""Alert creation and lifecycle.

Every alert carries a deterministic `dedupe_key`. Combined with the sparse unique
index on `(user_id, dedupe_key)` this makes alert creation idempotent: replaying
a budget check, re-running the nightly sweep, or retrying a failed request can
never produce a duplicate notification.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

from pymongo.errors import DuplicateKeyError

from app.core.constants import AlertSeverity, AlertType
from app.core.logging import get_logger
from app.core.time_utils import strip_tz, utc_now
from app.models.alert import AlertContext, AlertCreate, AlertCounts
from app.models.base import serialize_doc
from app.repositories.alert_repo import AlertRepository

logger = get_logger(__name__)

#: Alert payloads older than this are removed by the nightly sweep.
DEFAULT_TTL_DAYS = 90


class AlertService:
    def __init__(self, repo: AlertRepository) -> None:
        self.repo = repo

    # ------------------------------------------------------------- creation
    def create(
        self,
        user_id: Any,
        *,
        type: AlertType | str,
        message: str,
        title: str | None = None,
        severity: AlertSeverity | str = AlertSeverity.INFO,
        context: AlertContext | dict[str, Any] | None = None,
        dedupe_key: str | None = None,
        triggered_at: datetime | None = None,
        ttl_days: int | None = DEFAULT_TTL_DAYS,
        expires_at: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Insert an alert. Returns `None` if an identical alert already exists."""
        context_dict = context.model_dump(exclude_none=True) if isinstance(context, AlertContext) else dict(context or {})
        now = utc_now()
        document = {
            "user_id": user_id,
            "type": str(type),
            "title": title,
            "message": message,
            "severity": str(severity),
            "context": context_dict or None,
            "is_read": False,
            "read_at": None,
            "triggered_at": strip_tz(triggered_at or now),
            "expires_at": strip_tz(expires_at or (now + timedelta(days=ttl_days))) if ttl_days else expires_at,
            # Always present so the sparse unique index covers every document.
            "dedupe_key": dedupe_key or f"{type}:{uuid.uuid4().hex}",
        }
        try:
            self.repo.insert_one(document)
        except DuplicateKeyError:
            logger.debug("Duplicate alert suppressed for user=%s key=%s", user_id, document["dedupe_key"])
            return None
        return document

    def create_many(self, user_id: Any, alerts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        created = []
        for spec in alerts:
            document = self.create(user_id, **spec)
            if document:
                created.append(document)
        return created

    # --------------------------------------------------------------- reading
    def list_alerts(
        self,
        user_id: Any,
        *,
        is_read: bool | None = None,
        types: Iterable[AlertType | str] | None = None,
        severities: Iterable[AlertSeverity | str] | None = None,
        since: datetime | None = None,
        skip: int = 0,
        limit: int = 25,
    ) -> tuple[list[dict[str, Any]], int]:
        query_types = list(types) if types else None
        query_sevs = list(severities) if severities else None
        total = self.repo.count_for_user(
            user_id, is_read=is_read, types=query_types, severities=query_sevs, since=since
        )
        rows = self.repo.list_for_user(
            user_id, is_read=is_read, types=query_types, severities=query_sevs, since=since, skip=skip, limit=limit
        )
        return rows, total

    def counts(self, user_id: Any, *, since: datetime | None = None) -> AlertCounts:
        return AlertCounts(**self.repo.counts(user_id, since=since))

    def get(self, user_id: Any, alert_id: Any) -> dict[str, Any] | None:
        return self.repo.find_one({"_id": alert_id, "user_id": user_id})

    # --------------------------------------------------------------- mutating
    def mark_read(
        self,
        user_id: Any,
        *,
        alert_ids: Iterable[Any] | None = None,
        all_alerts: bool = False,
        types: Iterable[AlertType | str] | None = None,
        severities: Iterable[AlertSeverity | str] | None = None,
    ) -> int:
        if not (alert_ids or all_alerts or types or severities):
            return 0
        return self.repo.mark_read(
            user_id,
            alert_ids=alert_ids,
            all_alerts=all_alerts,
            types=types,
            severities=severities,
            read=True,
            read_at=strip_tz(utc_now()),
        )

    def mark_unread(self, user_id: Any, alert_ids: Iterable[Any]) -> int:
        return self.repo.mark_read(user_id, alert_ids=alert_ids, read=False)

    def update(self, user_id: Any, alert_id: Any, changes: dict[str, Any]) -> dict[str, Any] | None:
        patch = {k: v for k, v in changes.items() if v is not None}
        if "is_read" in patch:
            patch["read_at"] = strip_tz(utc_now()) if patch["is_read"] else None
        if not patch:
            return self.get(user_id, alert_id)
        return self.repo.find_one_and_update({"_id": alert_id, "user_id": user_id}, {"$set": patch})

    def delete(self, user_id: Any, alert_ids: Iterable[Any]) -> int:
        return self.repo.delete_for_user(user_id, alert_ids)

    def purge_expired(self) -> int:
        return self.repo.purge_expired()

    def serialize(self, document: dict[str, Any] | None) -> dict[str, Any] | None:
        return serialize_doc(document)

    def serialize_many(self, documents: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [serialize_doc(doc) for doc in documents if doc is not None]


def budget_threshold_dedupe_key(budget_id: Any, period_key: str, bucket: str) -> str:
    return f"{AlertType.BUDGET_THRESHOLD.value}:{budget_id}:{period_key}:{bucket}"


def budget_exceeded_dedupe_key(budget_id: Any, period_key: str) -> str:
    return f"{AlertType.BUDGET_EXCEEDED.value}:{budget_id}:{period_key}"


def goal_deadline_dedupe_key(goal_id: Any, target_date: Any, bucket: str) -> str:
    day = target_date.date().isoformat() if isinstance(target_date, datetime) else str(target_date)
    return f"{AlertType.SAVINGS_DEADLINE.value}:{goal_id}:{day}:{bucket}"


def goal_achieved_dedupe_key(goal_id: Any) -> str:
    return f"{AlertType.GOAL_ACHIEVED.value}:{goal_id}"


def ml_alert_dedupe_key(alert_type: str, user_id: Any, scope: str) -> str:
    return f"{alert_type}:{user_id}:{scope}"
