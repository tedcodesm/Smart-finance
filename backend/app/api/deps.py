"""Dependency injection: the request-scoped composition root.

Every router receives a `Services` bundle built once per request. Keeping the
graph construction here (rather than in each router) means tests can override a
single dependency and the wiring stays visible in one place.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any

from fastapi import Depends, Header
from pymongo.database import Database

from app.core.errors import AuthenticationError, NotFoundError
from app.core.logging import get_logger
from app.core.security import decode_token
from app.db.client import get_db as get_database
from app.models.base import to_object_id
from app.repositories.alert_repo import AlertRepository
from app.repositories.budget_repo import BudgetRepository
from app.repositories.category_repo import CategoryRepository
from app.repositories.goal_repo import GoalRepository
from app.repositories.prediction_repo import ModelRunRepository, PredictionRepository
from app.repositories.report_repo import ReportRepository
from app.repositories.transaction_repo import TransactionRepository
from app.repositories.user_repo import RefreshTokenRepository, UserRepository
from app.services.alert_service import AlertService
from app.services.auth_service import AuthService
from app.services.budget_service import BudgetService
from app.services.category_service import CategoryService
from app.services.dashboard_service import DashboardService
from app.services.goal_service import GoalService
from app.services.ml_service import MLService
from app.services.report_service import ReportService
from app.services.transaction_service import TransactionService

logger = get_logger(__name__)

MAX_TOKEN_LENGTH = 4096


def get_db() -> Database:
    """Database handle for the request (mongomock-overridable)."""
    return get_database()


def require_doc(doc: dict[str, Any] | None, message: str) -> dict[str, Any]:
    """Return `doc`, or raise 404 when a repository lookup found nothing.

    Service methods return `None` for missing rows so they stay storage-agnostic;
    routers funnel those results through here so a missing document surfaces as a
    404 instead of an empty body that fails response validation.
    """
    if not doc:
        raise NotFoundError(message)
    return doc


class Services:
    """Request-scoped service graph, built lazily and cached per request."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self._cache: dict[str, Any] = {}

    def _get(self, key: str, factory: Any) -> Any:
        service = self._cache.get(key)
        if service is None:
            service = self._cache[key] = factory()
        return service

    # ---------------------------------------------------------------- repos
    def alerts(self) -> AlertRepository:
        return self._get("alert_repo", lambda: AlertRepository(self.db))

    def budgets(self) -> BudgetRepository:
        return self._get("budget_repo", lambda: BudgetRepository(self.db))

    def categories(self) -> CategoryRepository:
        return self._get("category_repo", lambda: CategoryRepository(self.db))

    def goals(self) -> GoalRepository:
        return self._get("goal_repo", lambda: GoalRepository(self.db))

    def predictions(self) -> PredictionRepository:
        return self._get("prediction_repo", lambda: PredictionRepository(self.db))

    def model_runs(self) -> ModelRunRepository:
        return self._get("model_run_repo", lambda: ModelRunRepository(self.db))

    def reports(self) -> ReportRepository:
        return self._get("report_repo", lambda: ReportRepository(self.db))

    def transactions(self) -> TransactionRepository:
        return self._get("transaction_repo", lambda: TransactionRepository(self.db))

    def users(self) -> UserRepository:
        return self._get("user_repo", lambda: UserRepository(self.db))

    def refresh_tokens(self) -> RefreshTokenRepository:
        return self._get("refresh_repo", lambda: RefreshTokenRepository(self.db))

    # -------------------------------------------------------------- services
    def alert_service(self) -> AlertService:
        return self._get("alert_service", lambda: AlertService(self.alerts()))

    def auth_service(self) -> AuthService:
        return self._get("auth_service", lambda: AuthService(self.users(), self.refresh_tokens()))

    def budget_service(self) -> BudgetService:
        return self._get(
            "budget_service",
            lambda: BudgetService(self.budgets(), self.transactions(), self.categories(), self.alert_service()),
        )

    def category_service(self) -> CategoryService:
        return self._get(
            "category_service",
            lambda: CategoryService(self.categories(), self.transactions()),
        )

    def goal_service(self) -> GoalService:
        return self._get("goal_service", lambda: GoalService(self.goals(), self.alert_service()))

    def report_service(self) -> ReportService:
        return self._get(
            "report_service",
            lambda: ReportService(
                self.transactions(),
                self.categories(),
                self.budgets(),
                self.goals(),
                self.reports(),
                self.budget_service(),
                self.goal_service(),
            ),
        )

    def transaction_service(self) -> TransactionService:
        return self._get(
            "transaction_service",
            lambda: TransactionService(self.transactions(), self.categories(), self.budget_service()),
        )

    def dashboard_service(self) -> DashboardService:
        return self._get(
            "dashboard_service",
            lambda: DashboardService(
                transactions=self.transactions(),
                categories=self.categories(),
                budgets=self.budgets(),
                goals=self.goals(),
                predictions=self.predictions(),
                alerts=self.alert_service(),
                users=self.users(),
                budget_service=self.budget_service(),
                goal_service=self.goal_service(),
            ),
        )

    def ml_service(self) -> MLService:
        return self._get("ml_service", lambda: MLService(db=self.db))


def get_services(db: Database = Depends(get_db)) -> Services:
    return Services(db)


def get_current_user(
    authorization: str | None = Header(default=None),
    services: Services = Depends(get_services),
) -> dict[str, Any]:
    """Resolve the authenticated `users` document from the Bearer token."""
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AuthenticationError("Authentication required")
    if len(token) > MAX_TOKEN_LENGTH:
        raise AuthenticationError("Invalid authentication token")

    payload = decode_token(token, expected_type="access")
    subject = payload.get("sub")
    if not subject:
        raise AuthenticationError("Invalid authentication token")

    user_id = to_object_id(subject, field="user_id")
    user = services.users().find_by_id(user_id)
    if user is None or not user.get("is_active", True):
        raise AuthenticationError("Account is no longer active")
    return user


def get_active_user_ids(services: Services = Depends(get_services)) -> list[Any]:
    """All active user ids (used by admin/health endpoints and the nightly job)."""
    return services.users().list_active_user_ids()


#: Runtime entry point that points `mercy_ml` at the same MongoDB handle the
#: backend uses (so mongomock overrides in tests flow through unchanged).
_runtime_lock = threading.Lock()

_ml_import_lock = threading.Lock()
_ml_imports_warmed = False


def _warm_ml_imports() -> None:
    """Force the lazy `mercy_ml` imports (and their sklearn/pandas trees) to load once.

    `MLService` imports `mercy_ml.infer` / `mercy_ml.registry` inside the request
    handlers, and FastAPI runs `def` endpoints in anyio worker threads. Two
    concurrent *first* imports of the same module tree deadlock on CPython's
    per-module import locks (`_DeadlockError` on `sklearn.ensemble._forest`),
    which surfaced as a 500 when the Predictions page fired
    `GET /predictions/models/status` and `POST /predictions/expense-forecast`
    together on a database with no stored predictions yet.

    Resolving the modules here, single-threaded, during startup means every
    later import is a plain `sys.modules` lookup.
    """
    global _ml_imports_warmed
    if _ml_imports_warmed:
        return
    with _ml_import_lock:
        if _ml_imports_warmed:
            return
        importlib.import_module("mercy_ml.infer")
        importlib.import_module("mercy_ml.registry")
        _ml_imports_warmed = True


def ensure_ml_runtime(db: Database | None = None) -> None:
    from mercy_ml.data import set_client

    db = get_database() if db is None else db
    set_client(db.client)
    _warm_ml_imports()