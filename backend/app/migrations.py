"""Index application + seed helpers.

`migrations.ensure_indexes_and_seeds()` is called on FastAPI startup and can be
run from the `scripts/init_db.py` CLI. It is safe to run repeatedly.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings
from app.core.constants import (
    Collections,
    SYSTEM_EXPENSE_CATEGORIES,
    SYSTEM_INCOME_CATEGORIES,
    TransactionType,
)
from app.db.client import get_db
from app.db.index_definitions import COLLECTION_INDEXES, normalised

logger = logging.getLogger("mercy.migrations")


def ensure_indexes(db: Any | None = None) -> dict[str, list[str]]:
    """Create all defined indexes. Returns a map of collection -> created index names."""
    db = get_db() if db is None else db
    created: dict[str, list[str]] = {}
    for collection_name, specs in COLLECTION_INDEXES.items():
        names: list[str] = []
        try:
            collection = db[collection_name]
            for spec in specs:
                options = {k: v for k, v in normalised(spec).items() if k != "keys"}
                idx = collection.create_index(spec["keys"], **options)
                if idx:
                    names.append(idx)
            if names:
                created[collection_name] = names
                logger.info("Created indexes for %s: %s", collection_name, names)
        except Exception:  # pragma: no cover - best-effort
            logger.exception("Failed to create indexes for %s", collection_name)
    return created


def seed_system_categories(db: Any | None = None) -> int:
    """Insert shared system categories (expense and income) if they are missing."""
    db = get_db() if db is None else db
    categories = db[Collections.CATEGORIES]
    count = 0
    seeds = (
        (SYSTEM_EXPENSE_CATEGORIES, TransactionType.EXPENSE.value),
        (SYSTEM_INCOME_CATEGORIES, TransactionType.INCOME.value),
    )
    for definitions, cat_type in seeds:
        for cat in definitions:
            query = {"name": cat["name"], "type": cat_type, "is_system": True}
            if not categories.find_one(query):
                categories.insert_one(
                    {**cat, "type": cat_type, "is_system": True, "user_id": None, "is_archived": False}
                )
                count += 1
    if count:
        logger.info("Seeded %d system category(ies)", count)
    return count


def ensure_indexes_and_seeds(db: Any | None = None) -> dict[str, Any]:
    """Idempotent startup bootstrap."""
    db = get_db() if db is None else db
    indexes = ensure_indexes(db)
    seeded = seed_system_categories(db)
    return {"indexes": indexes, "system_categories_seeded": seeded}