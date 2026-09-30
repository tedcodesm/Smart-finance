"""Repositories - the only layer that speaks to the MongoDB driver."""

from app.repositories.alert_repo import AlertRepository
from app.repositories.base import BaseRepository
from app.repositories.budget_repo import BudgetRepository, effective_window
from app.repositories.category_repo import CategoryRepository
from app.repositories.goal_repo import GoalRepository
from app.repositories.prediction_repo import ModelRunRepository, PredictionRepository
from app.repositories.report_repo import ReportRepository
from app.repositories.transaction_repo import TransactionRepository
from app.repositories.user_repo import RefreshTokenRepository, UserRepository

__all__ = [
    "AlertRepository",
    "BaseRepository",
    "BudgetRepository",
    "CategoryRepository",
    "GoalRepository",
    "ModelRunRepository",
    "PredictionRepository",
    "RefreshTokenRepository",
    "ReportRepository",
    "TransactionRepository",
    "UserRepository",
    "effective_window",
]
