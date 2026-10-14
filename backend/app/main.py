"""FastAPI application entrypoint.

Run with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Ensure mercy_ml package in ml/src is importable regardless of deployment CWD
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ML_SRC = _REPO_ROOT / "ml" / "src"
if _ML_SRC.exists() and str(_ML_SRC) not in sys.path:
    sys.path.insert(0, str(_ML_SRC))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import ensure_ml_runtime
from app.api.errors import register_exception_handlers
from app.api.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.db.client import close_client, get_db, is_in_memory, use_in_memory_database
from app.demo_seed import seed_demo_data
from app.migrations import ensure_indexes_and_seeds
from app.scheduler import start_scheduler, stop_scheduler

logger = get_logger(__name__)

DESCRIPTION = """
Mercy is a personal finance manager. It tracks transactions, budgets, savings
goals and alerts, and surfaces ML predictions (spending forecasts, overspend
risk, budget suggestions, savings recommendations, spending clusters).

Authenticate with `POST /api/v1/auth/login` and send the returned access token
as `Authorization: Bearer <token>`.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Starting %s %s (%s)", settings.app_name, settings.version, settings.environment)
    if settings.in_memory_db:
        logger.warning(
            "IN_MEMORY_DB is enabled: using an in-process mongomock database "
            "(data is discarded when the process exits)"
        )
        use_in_memory_database()
    try:
        db = get_db()
        bootstrap = ensure_indexes_and_seeds(db)
        ensure_ml_runtime(db)
        logger.info("Database ready: %s", bootstrap)
    except Exception:  # pragma: no cover - fail soft so the app still boots
        logger.exception("Database bootstrap failed; continuing without indexes/seed")

    if settings.in_memory_db and settings.seed_demo_data:
        try:
            seed_demo_data(get_db())
        except Exception:  # pragma: no cover - never block startup on demo data
            logger.exception("Demo data seeding failed; continuing with an empty database")

    start_scheduler()
    yield
    stop_scheduler()
    close_client()


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"^https?://.*$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(api_router)


@app.get("/", include_in_schema=False)
def root() -> dict[str, Any]:
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "in_memory_db": is_in_memory(),
    }
