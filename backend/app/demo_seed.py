"""Demo data for local runs (`IN_MEMORY_DB=true`).

Everything is written through the regular service layer, so the seeded database is
indistinguishable from one produced by the API: denormalised date fields, cached
budget spend, contribution metrics and the alert side effects all happen for real.
"""

from __future__ import annotations

import random
from datetime import timedelta
from typing import Any

from pymongo.database import Database

from app.api.deps import Services
from app.core.config import settings
from app.core.constants import TransactionType
from app.core.logging import get_logger
from app.core.money import round_money
from app.core.time_utils import strip_tz, utc_now
from app.models.base import to_object_id

logger = get_logger(__name__)

#: (name, monthly spend, day-to-day spread, typical ticket, budget limit factor)
#: A factor below 1.0 makes the budget tight on purpose so the demo shows real
#: `budget_threshold` / `budget_exceeded` alerts instead of an empty alert tray.
_EXPENSE_PROFILES: tuple[tuple[str, float, float, float, float], ...] = (
    ("Groceries", 620.0, 0.35, 78.0, 1.1),
    ("Rent", 1450.0, 0.0, 1450.0, 1.0),
    ("Utilities", 180.0, 0.25, 64.0, 1.1),
    ("Transport", 240.0, 0.4, 42.0, 1.0),
    ("Dining Out", 310.0, 0.45, 58.0, 0.8),
    ("Entertainment", 150.0, 0.5, 45.0, 1.2),
    ("Healthcare", 95.0, 0.6, 70.0, 1.0),
    ("Shopping", 260.0, 0.55, 96.0, 0.85),
)

_INCOME_PROFILES: tuple[tuple[str, float, int], ...] = (
    ("Salary", 4200.0, 1),
    ("Freelance", 480.0, 1),
)

_GOALS: tuple[tuple[str, float, int, str], ...] = (
    ("Emergency fund", 9000.0, 240, "#38bdf8"),
    ("Japan trip", 3200.0, 150, "#f472b6"),
    ("New laptop", 1800.0, 90, "#34d399"),
)

_MONTHS_OF_HISTORY = 8


def seed_demo_data(db: Database) -> dict[str, Any] | None:
    """Create the demo account with ~8 months of history. Idempotent per database."""
    services = Services(db)
    existing = services.users().find_by_email(settings.demo_email)
    if existing:
        logger.info("Demo account already present (%s); skipping seed", settings.demo_email)
        return None

    rng = random.Random(settings.ml_random_seed)
    auth = services.auth_service()
    user = auth.register(
        {
            "name": "Demo User",
            "email": settings.demo_email,
            "password": settings.demo_password,
            "base_currency": settings.default_currency,
        }
    )
    user_id = user["_id"]

    expense_ids: dict[str, Any] = {}
    for name, _monthly, _spread, _ticket, _factor in _EXPENSE_PROFILES:
        expense_ids[name] = _category_id(services, user_id, name, TransactionType.EXPENSE)
    income_ids: dict[str, Any] = {}
    for name, _amount, _day in _INCOME_PROFILES:
        income_ids[name] = _category_id(services, user_id, name, TransactionType.INCOME)

    for name, monthly, spread, ticket, _factor in _EXPENSE_PROFILES:
        for month_offset in range(-_MONTHS_OF_HISTORY, 1):
            if monthly == ticket:  # fixed cost (rent): one payment per month
                _add(
                    services,
                    user_id,
                    expense_ids[name],
                    TransactionType.EXPENSE,
                    ticket,
                    _month_start(month_offset).replace(day=1),
                    name,
                    recurring=True,
                )
                continue
            purchases = max(1, int(monthly / max(ticket, 1.0) * rng.uniform(0.6, 1.4)))
            for _ in range(purchases):
                amount = round_money(ticket * rng.uniform(1 - spread, 1 + spread))
                day = rng.randint(1, 28)
                when = _month_start(month_offset).replace(day=day) + timedelta(
                    hours=rng.randint(7, 21), minutes=rng.randint(0, 59)
                )
                _add(services, user_id, expense_ids[name], TransactionType.EXPENSE, amount, when, name)

    for name, amount, day_of_month in _INCOME_PROFILES:
        for month_offset in range(-_MONTHS_OF_HISTORY, 1):
            when = _month_start(month_offset).replace(day=day_of_month)
            _add(services, user_id, income_ids[name], TransactionType.INCOME, amount, when, name, recurring=True)

    # Budgets are created after the history so their cached `spent_amount` and the
    # resulting alerts are computed from real spend.
    budgets = []
    for name, monthly, _spread, _ticket, factor in _EXPENSE_PROFILES[:6]:
        budget = services.budget_service().create_budget(
            user_id,
            {
                "limit_amount": round_money(monthly * factor),
                "period": "monthly",
                "category_id": str(expense_ids[name]),
                "name": name,
            },
        )
        budgets.append(budget["_id"])

    # Re-evaluate the current month so the demo account has live alerts waiting.
    services.budget_service().evaluate_budgets(user_id)

    goals = []
    for name, target, days, color in _GOALS:
        goal = services.goal_service().create_goal(
            user_id,
            {
                "name": name,
                "target_amount": target,
                "target_date": strip_tz(utc_now()) + timedelta(days=days),
                "color": color,
                "current_amount": round_money(target * rng.uniform(0.2, 0.75)),
            },
        )
        goals.append(goal["_id"])

    services.auth_service().issue_tokens_for(user_id)

    summary = {
        "email": settings.demo_email,
        "password": settings.demo_password,
        "transactions": db["transactions"].count_documents({"user_id": user_id}),
        "budgets": len(budgets),
        "goals": len(goals),
        "alerts": db["alerts"].count_documents({"user_id": user_id}),
    }
    logger.info(
        "Seeded demo data for %s: %d transactions, %d budgets, %d goals, %d alerts",
        settings.demo_email,
        summary["transactions"],
        summary["budgets"],
        summary["goals"],
        summary["alerts"],
    )
    return summary


# --------------------------------------------------------------------------- #
def _category_id(services: Services, user_id: Any, name: str, txn_type: TransactionType) -> Any:
    """Prefer the seeded system category; create a personal one if it is missing."""
    system = services.db["categories"].find_one({"name": name, "type": txn_type.value, "is_system": True})
    if system:
        return system["_id"]
    existing = services.categories().find_by_name(user_id, name, txn_type.value)
    if existing:
        return existing["_id"]
    created = services.category_service().create_category(
        user_id,
        {"name": name, "type": txn_type.value, "color": "#64748b", "monthly_typical_amount": 0.0},
    )
    return created["_id"]


def _month_start(offset: int):
    from datetime import datetime

    now = strip_tz(utc_now())
    year, month = now.year, now.month + offset
    while month < 1:
        month += 12
        year -= 1
    while month > 12:
        month -= 12
        year += 1
    return datetime(year, month, min(now.day, 28), 9, 0, 0)


def _add(
    services: Services,
    user_id: Any,
    category_id: Any,
    txn_type: TransactionType,
    amount: float,
    when: Any,
    description: str,
    *,
    recurring: bool = False,
) -> None:
    services.transaction_service().create_transaction(
        user_id,
        {
            "amount": round_money(amount),
            "type": txn_type.value,
            "category_id": str(category_id),
            "date": when,
            "description": description,
            "is_recurring": recurring,
        },
    )


__all__ = ["seed_demo_data", "to_object_id"]
