"""Domain enumerations shared by models, services, ML features and the API."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """`str`-backed enum so values serialise transparently to JSON/BSON."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class TransactionType(StrEnum):
    INCOME = "income"
    EXPENSE = "expense"


class BudgetPeriod(StrEnum):
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class GoalStatus(StrEnum):
    ACTIVE = "active"
    ACHIEVED = "achieved"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class PredictionType(StrEnum):
    """`Predictions.type` values. Mirrors `mercy_ml` model registry."""

    EXPENSE_FORECAST = "expense_forecast"
    OVERSPEND_RISK = "overspend_risk"
    BUDGET_RECOMMENDATION = "budget_recommendation"
    SAVINGS_RECOMMENDATION = "savings_recommendation"
    SPENDING_CLUSTER = "spending_cluster"
    UNUSUAL_SPENDING = "unusual_spending"


class ReportType(StrEnum):
    SPENDING_BY_CATEGORY = "spending_by_category"
    INCOME_VS_EXPENSE = "income_vs_expense"
    BUDGET_PERFORMANCE = "budget_performance"
    NET_CASH_FLOW = "net_cash_flow"
    GOAL_PROGRESS = "goal_progress"


class AlertType(StrEnum):
    BUDGET_THRESHOLD = "budget_threshold"
    BUDGET_EXCEEDED = "budget_exceeded"
    SAVINGS_DEADLINE = "savings_deadline"
    GOAL_ACHIEVED = "goal_achieved"
    GOAL_MILESTONE = "goal_milestone"
    UNUSUAL_SPENDING = "unusual_spending"
    OVERSPEND_RISK = "overspend_risk"
    FORECAST_WARNING = "forecast_warning"


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


#: Collections - the single source of truth for names used by migrations,
#: repositories and the ML data loader.
class Collections(StrEnum):
    USERS = "users"
    CATEGORIES = "categories"
    TRANSACTIONS = "transactions"
    BUDGETS = "budgets"
    SAVINGS_GOALS = "savings_goals"
    PREDICTIONS = "predictions"
    REPORTS = "reports"
    ALERTS = "alerts"
    MODEL_RUNS = "model_runs"
    SCHEMA_MIGRATIONS = "schema_migrations"
    REFRESH_TOKENS = "refresh_tokens"


#: Ordering of week buckets. `day_of_week` is ISO-8601: 0 = Monday.
DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

#: Default system categories seeded on a fresh database.
SYSTEM_EXPENSE_CATEGORIES: tuple[dict, ...] = (
    {"name": "Housing", "icon": "home", "color": "#6366f1", "monthly_typical_amount": 1500.0},
    {"name": "Groceries", "icon": "cart", "color": "#22c55e", "monthly_typical_amount": 520.0},
    {"name": "Dining Out", "icon": "utensils", "color": "#f59e0b", "monthly_typical_amount": 260.0},
    {"name": "Transport", "icon": "car", "color": "#0ea5e9", "monthly_typical_amount": 180.0},
    {"name": "Utilities", "icon": "bolt", "color": "#eab308", "monthly_typical_amount": 165.0},
    {"name": "Healthcare", "icon": "heart", "color": "#ef4444", "monthly_typical_amount": 95.0},
    {"name": "Entertainment", "icon": "film", "color": "#a855f7", "monthly_typical_amount": 110.0},
    {"name": "Shopping", "icon": "bag", "color": "#ec4899", "monthly_typical_amount": 190.0},
    {"name": "Subscriptions", "icon": "repeat", "color": "#14b8a6", "monthly_typical_amount": 65.0},
    {"name": "Education", "icon": "book", "color": "#3b82f6", "monthly_typical_amount": 120.0},
    {"name": "Travel", "icon": "plane", "color": "#f97316", "monthly_typical_amount": 140.0},
    {"name": "Insurance", "icon": "shield", "color": "#64748b", "monthly_typical_amount": 210.0},
    {"name": "Personal Care", "icon": "sparkles", "color": "#f472b6", "monthly_typical_amount": 70.0},
    {"name": "Miscellaneous", "icon": "dots", "color": "#94a3b8", "monthly_typical_amount": 80.0},
)

SYSTEM_INCOME_CATEGORIES: tuple[dict, ...] = (
    {"name": "Salary", "icon": "briefcase", "color": "#22c55e", "monthly_typical_amount": 4200.0},
    {"name": "Freelance", "icon": "laptop", "color": "#0ea5e9", "monthly_typical_amount": 600.0},
    {"name": "Investments", "icon": "trending-up", "color": "#8b5cf6", "monthly_typical_amount": 180.0},
    {"name": "Other Income", "icon": "plus-circle", "color": "#94a3b8", "monthly_typical_amount": 100.0},
)
