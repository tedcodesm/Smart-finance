"""Index definitions for every collection.

Kept in one module so that `migrations` can apply them, `scripts/init_db.py`
can (re)build them, and the schema doc stays in sync with the code. Indexes are
idempotent - MongoDB treats an identical `create_index` as a no-op.
"""

from __future__ import annotations

from typing import Any

from pymongo import ASCENDING, DESCENDING, TEXT

from app.core.constants import Collections

CollectionIndex = dict[str, Any]

# --------------------------------------------------------------------------- #
# users
# --------------------------------------------------------------------------- #
USER_INDEXES: list[CollectionIndex] = [
    {"name": "uq_users_email", "keys": [("email", ASCENDING)], "unique": True},
    {"name": "ix_users_created_at", "keys": [("created_at", DESCENDING)]},
    {"name": "ix_users_active", "keys": [("is_active", ASCENDING), ("created_at", DESCENDING)]},
]

# --------------------------------------------------------------------------- #
# categories
# --------------------------------------------------------------------------- #
CATEGORY_INDEXES: list[CollectionIndex] = [
    {"name": "ix_categories_user_type", "keys": [("user_id", ASCENDING), ("type", ASCENDING)]},
    {
        # One row per (user, name, type) for user-defined categories; the seed
        # system categories (user_id = null) are excluded via a partial filter.
        "name": "uq_categories_user_name_type",
        "keys": [("user_id", ASCENDING), ("name", ASCENDING), ("type", ASCENDING)],
        "unique": True,
        "partialFilterExpression": {"is_system": False},
    },
    {
        "name": "uq_categories_system_name_type",
        "keys": [("name", ASCENDING), ("type", ASCENDING)],
        "unique": True,
        "partialFilterExpression": {"is_system": True},
    },
    {"name": "ix_categories_archived", "keys": [("is_archived", ASCENDING), ("type", ASCENDING)]},
]

# --------------------------------------------------------------------------- #
# transactions  (the hot collection - every report scans by user + date)
# --------------------------------------------------------------------------- #
TRANSACTION_INDEXES: list[CollectionIndex] = [
    {"name": "ix_txn_user_date", "keys": [("user_id", ASCENDING), ("date", DESCENDING)]},
    {
        "name": "ix_txn_user_period_type_cat",
        "keys": [
            ("user_id", ASCENDING),
            ("period_key", ASCENDING),
            ("type", ASCENDING),
            ("category_id", ASCENDING),
        ],
    },
    {
        "name": "ix_txn_user_cat_date",
        "keys": [("user_id", ASCENDING), ("category_id", ASCENDING), ("date", DESCENDING)],
    },
    {
        "name": "ix_txn_user_type_date",
        "keys": [("user_id", ASCENDING), ("type", ASCENDING), ("date", DESCENDING)],
    },
    {"name": "ix_txn_user_created", "keys": [("user_id", ASCENDING), ("created_at", DESCENDING)]},
    {
        "name": "ix_txn_user_dow",
        "keys": [("user_id", ASCENDING), ("day_of_week", ASCENDING), ("type", ASCENDING)],
    },
    {"name": "ix_txn_description_text", "keys": [("description", TEXT)], "weights": {"description": 5}},
]

# --------------------------------------------------------------------------- #
# budgets
# --------------------------------------------------------------------------- #
BUDGET_INDEXES: list[CollectionIndex] = [
    {
        "name": "ix_budget_user_active_period",
        "keys": [("user_id", ASCENDING), ("is_active", ASCENDING), ("period", ASCENDING)],
    },
    {
        "name": "ix_budget_user_cat_period_start",
        "keys": [
            ("user_id", ASCENDING),
            ("category_id", ASCENDING),
            ("period", ASCENDING),
            ("start_date", DESCENDING),
        ],
    },
    {"name": "ix_budget_user_period_key", "keys": [("user_id", ASCENDING), ("period_key", ASCENDING)]},
    {
        "name": "ix_budget_user_start_date",
        "keys": [("user_id", ASCENDING), ("start_date", DESCENDING), ("end_date", DESCENDING)],
    },
]

# --------------------------------------------------------------------------- #
# savings_goals
# --------------------------------------------------------------------------- #
GOAL_INDEXES: list[CollectionIndex] = [
    {"name": "ix_goals_user_status", "keys": [("user_id", ASCENDING), ("status", ASCENDING)]},
    {
        "name": "ix_goals_user_target_date",
        "keys": [("user_id", ASCENDING), ("target_date", ASCENDING), ("status", ASCENDING)],
    },
    {"name": "ix_goals_user_name", "keys": [("user_id", ASCENDING), ("name", ASCENDING)]},
]

# --------------------------------------------------------------------------- #
# predictions
# --------------------------------------------------------------------------- #
PREDICTION_INDEXES: list[CollectionIndex] = [
    {
        "name": "ix_pred_user_type_recent",
        "keys": [("user_id", ASCENDING), ("type", ASCENDING), ("generated_at", DESCENDING)],
    },
    {"name": "ix_pred_user_recent", "keys": [("user_id", ASCENDING), ("generated_at", DESCENDING)]},
    {
        "name": "ix_pred_user_scope",
        "keys": [
            ("user_id", ASCENDING),
            ("type", ASCENDING),
            ("scope.period_key", ASCENDING),
            ("scope.category_id", ASCENDING),
        ],
    },
    {"name": "ix_pred_run", "keys": [("model_run_id", ASCENDING)]},
    {"name": "ix_pred_user_model_version", "keys": [("user_id", ASCENDING), ("model_version", ASCENDING)]},
    {
        "name": "ix_pred_expires",
        "keys": [("expires_at", ASCENDING)],
        "expireAfterSeconds": 0,
        "partialFilterExpression": {"expires_at": {"$type": "date"}},
        "note": "optional TTL; disabled by default - see MIGRATION_NOTES",
    },
]

# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #
REPORT_INDEXES: list[CollectionIndex] = [
    {
        "name": "ix_report_user_type_recent",
        "keys": [("user_id", ASCENDING), ("report_type", ASCENDING), ("generated_at", DESCENDING)],
    },
    {"name": "ix_report_user_range", "keys": [("user_id", ASCENDING), ("period_start", DESCENDING)]},
    {"name": "ix_report_user_recent", "keys": [("user_id", ASCENDING), ("generated_at", DESCENDING)]},
]

# --------------------------------------------------------------------------- #
# alerts
# --------------------------------------------------------------------------- #
ALERT_INDEXES: list[CollectionIndex] = [
    {
        "name": "ix_alert_user_unread_recent",
        "keys": [("user_id", ASCENDING), ("is_read", ASCENDING), ("triggered_at", DESCENDING)],
    },
    {"name": "ix_alert_user_type_recent", "keys": [("user_id", ASCENDING), ("type", ASCENDING), ("triggered_at", DESCENDING)]},
    {
        # Sparse unique index => automatic de-duplication: re-running a budget
        # check or the nightly sweep cannot create the same alert twice.
        "name": "uq_alert_user_dedupe",
        "keys": [("user_id", ASCENDING), ("dedupe_key", ASCENDING)],
        "unique": True,
        "sparse": True,
    },
    {"name": "ix_alert_severity", "keys": [("user_id", ASCENDING), ("severity", ASCENDING), ("triggered_at", DESCENDING)]},
]

# --------------------------------------------------------------------------- #
# model_runs  (ML training bookkeeping)
# --------------------------------------------------------------------------- #
MODEL_RUN_INDEXES: list[CollectionIndex] = [
    {"name": "uq_model_run_id", "keys": [("run_id", ASCENDING)], "unique": True},
    {"name": "ix_model_run_type_active", "keys": [("model_type", ASCENDING), ("is_active", DESCENDING)]},
    {"name": "ix_model_run_trained", "keys": [("trained_at", DESCENDING)]},
]

# --------------------------------------------------------------------------- #
# refresh_tokens  (server-side revocation list for JWT refresh tokens)
# --------------------------------------------------------------------------- #
REFRESH_TOKEN_INDEXES: list[CollectionIndex] = [
    {"name": "uq_refresh_jti", "keys": [("jti", ASCENDING)], "unique": True},
    {"name": "ix_refresh_user", "keys": [("user_id", ASCENDING), ("revoked", ASCENDING)]},
    {"name": "ix_refresh_expires", "keys": [("expires_at", ASCENDING)], "expireAfterSeconds": 0},
]

# --------------------------------------------------------------------------- #
# schema_migrations
# --------------------------------------------------------------------------- #
SCHEMA_MIGRATION_INDEXES: list[CollectionIndex] = [
    {"name": "uq_migration_version", "keys": [("version", ASCENDING)], "unique": True},
]


COLLECTION_INDEXES: dict[str, list[CollectionIndex]] = {
    Collections.USERS: USER_INDEXES,
    Collections.CATEGORIES: CATEGORY_INDEXES,
    Collections.TRANSACTIONS: TRANSACTION_INDEXES,
    Collections.BUDGETS: BUDGET_INDEXES,
    Collections.SAVINGS_GOALS: GOAL_INDEXES,
    Collections.PREDICTIONS: PREDICTION_INDEXES,
    Collections.REPORTS: REPORT_INDEXES,
    Collections.ALERTS: ALERT_INDEXES,
    Collections.MODEL_RUNS: MODEL_RUN_INDEXES,
    Collections.REFRESH_TOKENS: REFRESH_TOKEN_INDEXES,
    Collections.SCHEMA_MIGRATIONS: SCHEMA_MIGRATION_INDEXES,
}

ALL_COLLECTION_NAMES: tuple[str, ...] = tuple(COLLECTION_INDEXES)


def normalised(spec: CollectionIndex) -> dict[str, Any]:
    """Drop non-driver keys (e.g. `note`) before passing to `create_index`."""
    return {k: v for k, v in spec.items() if k != "note"}
