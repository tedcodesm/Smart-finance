"""MongoDB client / database access.

A single process-wide `MongoClient` is reused (PyMongo maintains its own
connection pool). Route handlers are declared as `def` (not `async def`) so
FastAPI executes the blocking driver calls in its worker threadpool.
"""

from __future__ import annotations

import threading
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_client: MongoClient | None = None
_override_db: Database | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = MongoClient(
                    settings.mongo_uri,
                    serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
                    tz_aware=False,
                    appname="mercy-api",
                )
                logger.info("MongoDB client created for %s", _mask_uri(settings.mongo_uri))
    return _client


def get_db() -> Database:
    """Database handle for the current request.

    Tests can call `override_database(...)` to inject an in-memory
    `mongomock` database; production always uses the real driver.
    """
    if _override_db is not None:
        return _override_db
    return get_client()[settings.mongo_db_name]


def override_database(db: Database) -> None:
    """Swap the database handle (tests / scripts). Pass `None` to restore."""
    global _override_db
    _override_db = db


_in_memory_client: Any = None


def use_in_memory_database(db_name: str | None = None) -> Database:
    """Point the whole process at an in-process `mongomock` database.

    Lets the API (and `mercy_ml`, which shares this client) run with no MongoDB
    server installed. Requires the optional `mongomock` package.
    """
    global _in_memory_client, _override_db
    from app.core.config import settings

    try:
        import mongomock
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "IN_MEMORY_DB=true requires the 'mongomock' package: pip install mongomock"
        ) from exc

    name = db_name or settings.mongo_db_name
    if _in_memory_client is None:
        _in_memory_client = mongomock.MongoClient()
    database = _in_memory_client[name]
    _override_db = database
    return database


def is_in_memory() -> bool:
    """True when the process is serving an in-process database."""
    return _in_memory_client is not None and _override_db is not None


def close_client() -> None:
    global _client
    with _lock:
        if _client is not None:
            _client.close()
            _client = None


def ping() -> bool:
    try:
        get_db().client.admin.command("ping")
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("MongoDB ping failed: %s", exc)
        return False


def _mask_uri(uri: str) -> str:
    if "@" not in uri:
        return uri
    scheme, _, rest = uri.partition("://")
    _, _, host = rest.partition("@")
    return f"{scheme}://***@{host}"


__all__ = [
    "ASCENDING",
    "DESCENDING",
    "Database",
    "close_client",
    "get_client",
    "get_db",
    "is_in_memory",
    "override_database",
    "ping",
    "use_in_memory_database",
]


def _unused() -> Any:  # pragma: no cover
    return None
