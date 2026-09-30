"""Read-only data access for the ML pipeline.

The ML module never writes to the user-facing collections. Its only write path is
the `predictions` collection (plus `model_runs`), and that is driven by the
backend's `ml_service` / batch job rather than by training code.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

import pandas as pd
from pymongo import MongoClient
from pymongo.database import Database

from mercy_ml.config import get_settings
from mercy_ml.schema import CATEGORY, PERIOD, USER

_client: MongoClient | None = None

TRANSACTION_PROJECTION: dict[str, int] = {
    "_id": 0,
    "user_id": 1,
    "category_id": 1,
    "type": 1,
    "amount": 1,
    "date": 1,
    "year": 1,
    "month": 1,
    "day": 1,
    "day_of_week": 1,
    "day_of_month": 1,
    "week_of_year": 1,
    "period_key": 1,
    "week_key": 1,
    "is_recurring": 1,
    "created_at": 1,
}


def get_client() -> MongoClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = MongoClient(
            settings.mongo_uri,
            serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
            tz_aware=False,
            appname="mercy-ml",
        )
    return _client


def set_client(client: MongoClient | None) -> MongoClient:
    """Override the client, typically with mongomock for tests and offline runs."""
    global _client
    if _client is not None and _client is not client:
        try:
            _client.close()
        except Exception:  # pragma: no cover - best-effort close
            pass
    _client = client
    return client


def get_db() -> Database:
    return get_client()[get_settings().mongo_db_name]


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


# --------------------------------------------------------------------------- #
# Loaders -> DataFrames
# --------------------------------------------------------------------------- #
def load_transactions(
    *,
    user_ids: Iterable[Any] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    types: list[str] | None = None,
    limit: int = 0,
) -> pd.DataFrame:
    """Transaction history as a DataFrame with typed columns."""
    query: dict[str, Any] = {}
    if user_ids:
        query["user_id"] = {"$in": list(user_ids)}
    if start is not None or end is not None:
        window: dict[str, Any] = {}
        if start is not None:
            window["$gte"] = start
        if end is not None:
            window["$lte"] = end
        query["date"] = window
    if types:
        query["type"] = {"$in": types}

    cursor = get_db()["transactions"].find(query, TRANSACTION_PROJECTION)
    if limit:
        cursor = cursor.limit(limit)
    records = list(cursor)
    if not records:
        return empty_transactions()

    frame = pd.DataFrame.from_records(records)
    return normalise_transactions(frame)


def empty_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": pd.Series(dtype="object"),
            "category_id": pd.Series(dtype="object"),
            "type": pd.Series(dtype="object"),
            "amount": pd.Series(dtype="float64"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "year": pd.Series(dtype="int64"),
            "month": pd.Series(dtype="int64"),
            "day": pd.Series(dtype="int64"),
            "day_of_week": pd.Series(dtype="int64"),
            "day_of_month": pd.Series(dtype="int64"),
            "week_of_year": pd.Series(dtype="int64"),
            "period_key": pd.Series(dtype="object"),
            "week_key": pd.Series(dtype="object"),
            "is_recurring": pd.Series(dtype="float64"),
            "created_at": pd.Series(dtype="datetime64[ns]"),
        }
    )


def normalise_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce BSON-derived columns into the dtypes the feature code assumes."""
    if frame.empty:
        return empty_transactions()

    frame = frame.copy()
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=False)
    if "created_at" in frame:
        frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce", utc=False)
    for column in ("amount",):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("year", "month", "day", "day_of_week", "day_of_month", "week_of_year"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    if "is_recurring" in frame:
        frame["is_recurring"] = frame["is_recurring"].fillna(0).astype("float64")

    # Derive any missing bucket fields from `date` rather than trusting the row.
    if "period_key" not in frame or frame["period_key"].isna().all():
        if "date" in frame:
            frame["period_key"] = frame["date"].dt.strftime("%Y-%m")
    if "day_of_week" in frame:
        # Normalise to ISO (0 = Monday) regardless of how it was stored.
        frame["day_of_week"] = frame["day_of_week"].astype("Int64")
    return frame.reset_index(drop=True)


def load_categories(*, user_ids: Iterable[Any] | None = None) -> pd.DataFrame:
    """Category dimension table (id -> name/type), including system categories."""
    query: dict[str, Any] = {}
    if user_ids:
        query["$or"] = [{"user_id": {"$in": list(user_ids)}}, {"user_id": None}]
    records = list(
        get_db()["categories"].find(query, {"_id": 1, "name": 1, "type": 1, "user_id": 1, "is_system": 1})
    )
    if not records:
        return pd.DataFrame(columns=["category_id", "category_name", "category_type", "is_system"])
    frame = pd.DataFrame.from_records(records).rename(columns={"_id": "category_id", "type": "category_type"})
    frame["category_name"] = frame["name"].fillna("Unknown").astype(str)
    return frame[["category_id", "category_name", "category_type", "is_system"]]


def load_budgets(*, user_ids: Iterable[Any] | None = None, include_inactive: bool = True) -> pd.DataFrame:
    """Budget definitions used to build overspend-training observations."""
    query: dict[str, Any] = {}
    if user_ids:
        query["user_id"] = {"$in": list(user_ids)}
    if not include_inactive:
        query["is_active"] = True

    records = list(
        get_db()["budgets"].find(
            query,
            {
                "_id": 1,
                "user_id": 1,
                "category_id": 1,
                "limit_amount": 1,
                "period": 1,
                "start_date": 1,
                "end_date": 1,
                "is_active": 1,
            },
        )
    )
    if not records:
        return pd.DataFrame(
            columns=["budget_id", "user_id", "category_id", "limit_amount", "period", "start_date", "is_active"]
        )
    frame = pd.DataFrame.from_records(records).rename(columns={"_id": "budget_id"})
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce")
    frame["limit_amount"] = pd.to_numeric(frame["limit_amount"], errors="coerce")
    return frame


def load_users(*, active_only: bool = True) -> pd.DataFrame:
    query: dict[str, Any] = {"is_active": True} if active_only else {}
    records = list(get_db()["users"].find(query, {"_id": 1, "created_at": 1, "monthly_income_estimate": 1}))
    if not records:
        return pd.DataFrame(columns=["user_id", "created_at", "monthly_income_estimate"])
    frame = pd.DataFrame.from_records(records).rename(columns={"_id": "user_id"})
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    return frame


def load_training_bundle(
    *,
    user_ids: Iterable[Any] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Everything training needs, in one call."""
    user_id_list = list(user_ids) if user_ids else None
    return {
        "transactions": load_transactions(user_ids=user_id_list, start=start, end=end),
        "categories": load_categories(user_ids=user_id_list),
        "budgets": load_budgets(user_ids=user_id_list),
        "users": load_users(),
    }


def panel_periods(transactions: pd.DataFrame) -> list[str]:
    """Sorted list of `YYYY-MM` buckets present in the data."""
    if transactions.empty or PERIOD not in transactions:
        return []
    return sorted(str(value) for value in transactions[PERIOD].dropna().unique())


def transactions_for_user(user_id: Any, *, months: int = 24) -> pd.DataFrame:
    """Per-user history used by on-demand inference."""
    frame = load_transactions(user_ids=[user_id])
    if frame.empty:
        return frame
    frame = frame[frame[USER] == user_id]
    return frame.tail(months * 200) if len(frame) > months * 200 else frame
