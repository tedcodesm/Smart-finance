"""Synthetic data generator.

Real transaction history does not exist yet, and waiting for it would mean no
model can be built, evaluated, or demoed. This module fabricates a panel that is
*statistically* realistic rather than random:

  * income arrives on a payday with persona-specific jitter and, for freelancers,
    genuinely high month-to-month variance;
  * rent/utilities/subscriptions are genuinely monthly and recurring;
  * groceries and dining are weekly-ish and weekend-weighted, which is what gives
    the K-Means clustering real structure to find;
  * some users are chronic overspenders and some never overspend, so the
    overspend classifier has a learnable signal rather than a coin flip;
  * budgets are set at a random fraction of the user's *actual* average, which
    produces a realistic mix of over- and under-budget periods.

The output is a list of BSON-shaped documents that can be inserted into a real
MongoDB or an in-memory `mongomock` client, so the same fixture drives both the
training pipeline and the test suite.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

from bson import ObjectId

from mercy_ml.periods import month_key, week_key

# --------------------------------------------------------------------------- #
# Expense "behaviours"
# --------------------------------------------------------------------------- #
#: (category, per-month frequency, mean amount, amount spread, weekend weight,
#:  recurring)
EXPENSE_BEHAVIOURS: tuple[tuple[str, float, float, float, float, bool], ...] = (
    ("Groceries", 3.2, 78.0, 0.45, 0.55, False),
    ("Dining Out", 4.5, 34.0, 0.60, 0.70, False),
    ("Transport", 2.4, 26.0, 0.50, 0.20, False),
    ("Entertainment", 1.8, 42.0, 0.65, 0.65, False),
    ("Shopping", 1.5, 88.0, 0.85, 0.60, False),
    ("Personal Care", 1.2, 32.0, 0.50, 0.40, False),
    ("Healthcare", 0.35, 120.0, 0.70, 0.10, False),
    ("Education", 0.45, 140.0, 0.55, 0.20, False),
)

#: Fixed monthly commitments, paid on a stable day each month.
RECURRING_EXPENSES: tuple[tuple[str, int, float, float], ...] = (
    ("Housing", 1, 1450.0, 0.04),
    ("Utilities", 8, 165.0, 0.22),
    ("Subscriptions", 15, 64.0, 0.18),
    ("Insurance", 3, 205.0, 0.05),
)

INCOME_BEHAVIOURS: tuple[tuple[str, int, float, float], ...] = (
    ("Salary", 26, 4200.0, 0.03),
)

PERSONAS: tuple[dict[str, Any], ...] = (
    {
        "key": "steady",
        "weight": 0.35,
        "income_scale": (1.0, 1.4),
        "income_volatility": (0.01, 0.05),
        "expense_scale": (0.85, 1.15),
        "discretionary_bias": (0.8, 1.1),
        "budget_factor": (1.0, 1.35),
        "savings_goal_count": (1, 2),
    },
    {
        "key": "saver",
        "weight": 0.25,
        "income_scale": (0.8, 1.2),
        "income_volatility": (0.01, 0.04),
        "expense_scale": (0.6, 0.9),
        "discretionary_bias": (0.5, 0.8),
        "budget_factor": (1.1, 1.5),
        "savings_goal_count": (1, 3),
    },
    {
        "key": "spender",
        "weight": 0.25,
        "income_scale": (0.9, 1.3),
        "income_volatility": (0.02, 0.06),
        "expense_scale": (1.1, 1.6),
        "discretionary_bias": (1.3, 1.9),
        "budget_factor": (0.7, 1.0),
        "savings_goal_count": (0, 1),
    },
    {
        "key": "freelancer",
        "weight": 0.15,
        "income_scale": (0.7, 1.1),
        "income_volatility": (0.30, 0.75),
        "expense_scale": (0.7, 1.1),
        "discretionary_bias": (0.9, 1.4),
        "budget_factor": (0.85, 1.25),
        "savings_goal_count": (1, 2),
    },
)

FIRST_NAMES = ("Alex", "Blair", "Casey", "Devon", "Elena", "Frankie", "Gray", "Harper",
               "Indigo", "Jamie", "Kai", "Lior", "Morgan", "Noor", "Oakley", "Parker",
               "Quinn", "Rowan", "Sasha", "Tatum")
LAST_NAMES = ("Adler", "Bellamy", "Cardoso", "Duval", "Eriksen", "Fontaine", "Gupta",
              "Halvorsen", "Ibrahim", "Jansen", "Kowalski", "Larkin", "Moreau", "Nakamura")


@dataclass
class _UserProfile:
    """Everything needed to generate one user's history."""

    user_id: ObjectId
    name: str
    email: str
    persona: str
    income: float
    income_cv: float
    expense_scale: float
    discretionary_bias: float
    budget_factor: float
    salary_day: int
    goal_count: int
    created_at: datetime
    category_ids: dict[str, ObjectId] = field(default_factory=dict)
    user_category_ids: dict[str, ObjectId] = field(default_factory=dict)


class SyntheticGenerator:
    """Builds a consistent, multi-collection fixture."""

    def __init__(
        self,
        *,
        n_users: int = 60,
        months: int = 14,
        end: date | None = None,
        seed: int = 20260101,
        currency: str = "USD",
    ) -> None:
        self.rng = random.Random(seed)
        self.n_users = n_users
        self.months = months
        self.currency = currency
        # Anchor the window so data ends on the first of the current month: the
        # trailing partial period is never a complete observation.
        self.end_month = (end or date.today()).replace(day=1)
        self.periods = _month_range(self.end_month, months)
        #: First day of the earliest generated month, as a date.
        first_year, first_month = (int(part) for part in self.periods[0].split("-"))
        self.start = date(first_year, first_month, 1)

    # ----------------------------------------------------------------- build
    def generate(self) -> dict[str, list[dict[str, Any]]]:
        categories = self._build_categories()
        system_ids = {doc["name"]: doc["_id"] for doc in categories}
        profiles = self._build_users()
        for profile in profiles:
            profile.category_ids = dict(system_ids)
        transactions: list[dict[str, Any]] = []
        budgets: list[dict[str, Any]] = []
        goals: list[dict[str, Any]] = []

        for profile in profiles:
            transactions.extend(self._user_transactions(profile))
            budgets.extend(self._user_budgets(profile, categories))
            goals.extend(self._user_goals(profile))

        transactions.sort(key=lambda doc: (doc["user_id"], doc["date"]))
        return {
            "users": [self._user_doc(p) for p in profiles],
            "categories": categories,
            "transactions": transactions,
            "budgets": budgets,
            "savings_goals": goals,
        }

    # ------------------------------------------------------------ categories
    def _build_categories(self) -> list[dict[str, Any]]:
        now = datetime.utcnow()
        expense = [
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
        ]
        income = [
            {"name": "Salary", "icon": "briefcase", "color": "#22c55e", "monthly_typical_amount": 4200.0},
            {"name": "Freelance", "icon": "laptop", "color": "#0ea5e9", "monthly_typical_amount": 600.0},
            {"name": "Investments", "icon": "trending-up", "color": "#8b5cf6", "monthly_typical_amount": 180.0},
            {"name": "Other Income", "icon": "plus-circle", "color": "#94a3b8", "monthly_typical_amount": 100.0},
        ]
        docs: list[dict[str, Any]] = []
        for kind, items in (("expense", expense), ("income", income)):
            for item in items:
                docs.append(
                    {
                        "_id": ObjectId(),
                        "user_id": None,
                        "name": item["name"],
                        "type": kind,
                        "icon": item["icon"],
                        "color": item["color"],
                        "is_system": True,
                        "is_archived": False,
                        "monthly_typical_amount": item["monthly_typical_amount"],
                        "created_at": now,
                        "updated_at": now,
                    }
                )
        return docs

    # ----------------------------------------------------------------- users
    def _build_users(self) -> list[_UserProfile]:
        profiles: list[_UserProfile] = []
        keys = [p["key"] for p in PERSONAS]
        weights = [p["weight"] for p in PERSONAS]
        by_key = {p["key"]: p for p in PERSONAS}

        for index in range(self.n_users):
            key = self.rng.choices(keys, weights=weights, k=1)[0]
            spec = by_key[key]
            first = self.rng.choice(FIRST_NAMES)
            last = self.rng.choice(LAST_NAMES)
            user_id = ObjectId()
            created = self.start - timedelta(days=self.rng.randint(30, 900))
            created = datetime(created.year, created.month, created.day, self.rng.randint(8, 20), self.rng.randint(0, 59))
            profiles.append(
                _UserProfile(
                    user_id=user_id,
                    name=f"{first} {last}",
                    email=f"{first.lower()}.{last.lower()}{index}@example.com",
                    persona=key,
                    income=self.rng.uniform(*spec["income_scale"]) * 4200.0,
                    income_cv=self.rng.uniform(*spec["income_volatility"]),
                    expense_scale=self.rng.uniform(*spec["expense_scale"]),
                    discretionary_bias=self.rng.uniform(*spec["discretionary_bias"]),
                    budget_factor=self.rng.uniform(*spec["budget_factor"]),
                    salary_day=self.rng.choice((24, 25, 26, 27, 28)),
                    goal_count=self.rng.randint(*spec["savings_goal_count"]),
                    created_at=created,
                )
            )
        return profiles

    def _user_doc(self, profile: _UserProfile) -> dict[str, Any]:
        return {
            "_id": profile.user_id,
            "name": profile.name,
            "email": profile.email,
            # A real bcrypt hash of "Password123!" would be misleading; these
            # fixtures are for training only and are never used to authenticate.
            "password_hash": "synthetic-not-a-real-hash",
            "base_currency": self.currency,
            "is_active": True,
            "monthly_income_estimate": round(profile.income, 2),
            "created_at": profile.created_at,
            "updated_at": datetime(self.start.year, self.start.month, 1),
            "last_login_at": datetime(self.start.year, self.start.month, 1),
        }

    # ---------------------------------------------------------- transactions
    def _user_transactions(self, profile: _UserProfile) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for period in self.periods:
            year, month = (int(part) for part in period.split("-"))
            days_in_month = _days_in_month(year, month)
            docs.extend(self._income_for_month(profile, year, month, days_in_month))
            docs.extend(self._expenses_for_month(profile, year, month, days_in_month))
        return docs

    def _income_for_month(
        self, profile: _UserProfile, year: int, month: int, days_in_month: int
    ) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []

        # Salary on the payday.
        day = min(profile.salary_day, days_in_month)
        noise = self.rng.gauss(1.0, profile.income_cv)
        salary = max(0.0, profile.income * max(0.35, noise))
        if salary > 0:
            docs.append(
                self._txn(
                    profile,
                    when=date(year, month, day),
                    amount=salary,
                    kind="income",
                    category="Salary",
                    description="Monthly salary",
                    recurring=True,
                )
            )

        # Freelance work for freelancers (and occasionally everyone else).
        freelance_chance = 0.75 if profile.persona == "freelancer" else 0.18
        if self.rng.random() < freelance_chance:
            invoices = self.rng.randint(1, 3)
            for _ in range(invoices):
                amount = max(
                    80.0, abs(self.rng.gauss(1.0, 0.6)) * profile.income * 0.22
                )
                docs.append(
                    self._txn(
                        profile,
                        when=date(year, month, self.rng.randint(1, days_in_month)),
                        amount=amount,
                        kind="income",
                        category="Freelance",
                        description="Client invoice",
                    )
                )

        # Occasional investment/other income.
        if self.rng.random() < 0.35:
            docs.append(
                self._txn(
                    profile,
                    when=date(year, month, self.rng.randint(1, days_in_month)),
                    amount=abs(self.rng.gauss(0.0, 1.0)) * profile.income * 0.04 + 15.0,
                    kind="income",
                    category=self.rng.choice(("Investments", "Other Income")),
                    description="Interest / dividend",
                )
            )
        return docs

    def _expenses_for_month(
        self, profile: _UserProfile, year: int, month: int, days_in_month: int
    ) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []

        for category, day, amount, spread in RECURRING_EXPENSES:
            jitter = max(0.0, self.rng.gauss(1.0, spread))
            value = amount * profile.expense_scale * jitter
            docs.append(
                self._txn(
                    profile,
                    when=date(year, month, min(day, days_in_month)),
                    amount=value,
                    kind="expense",
                    category=category,
                    description=f"{category} - monthly",
                    recurring=True,
                )
            )

        for category, per_month, mean, spread, weekend_weight, _ in EXPENSE_BEHAVIOURS:
            expected = per_month * (
                1.0 if category in ("Groceries", "Dining Out", "Transport", "Personal Care")
                else profile.discretionary_bias
            )
            count = _poisson(self.rng, expected)
            for _ in range(count):
                when = self._pick_day(days_in_month, weekend_weight)
                bias = profile.discretionary_bias if category not in (
                    "Groceries", "Transport"
                ) else 1.0
                value = abs(self.rng.gauss(mean * bias, mean * spread)) * profile.expense_scale
                if value < 1.0:
                    continue
                docs.append(
                    self._txn(
                        profile,
                        when=date(year, month, when),
                        amount=value,
                        kind="expense",
                        category=category,
                        description=f"{category} purchase",
                    )
                )

        # Rare large travel spike, which is what makes clustering interesting.
        if self.rng.random() < 0.10:
            docs.append(
                self._txn(
                    profile,
                    when=date(year, month, self.rng.randint(1, days_in_month)),
                    amount=abs(self.rng.gauss(900.0, 450.0)) * profile.expense_scale,
                    kind="expense",
                    category="Travel",
                    description="Trip booking",
                )
            )
        return docs

    def _pick_day(self, days_in_month: int, weekend_weight: float) -> int:
        for _ in range(8):
            day = self.rng.randint(1, days_in_month)
            if self.rng.random() < weekend_weight:
                # Nudge towards Sat/Sun.
                shift = self.rng.choice((5, 6)) - (day - 1) % 7
                day = min(days_in_month, max(1, day + shift))
            return day
        return self.rng.randint(1, days_in_month)

    def _txn(
        self,
        profile: _UserProfile,
        *,
        when: date,
        amount: float,
        kind: str,
        category: str,
        description: str,
        recurring: bool = False,
    ) -> dict[str, Any]:
        when_time = datetime(when.year, when.month, when.day, self.rng.randint(7, 22), self.rng.randint(0, 59))
        iso = when.isocalendar()
        return {
            "_id": ObjectId(),
            "user_id": profile.user_id,
            "category_id": profile.category_ids.get(category, ObjectId()),
            "amount": round(max(0.01, amount), 2),
            "type": kind,
            "date": when_time,
            "description": description,
            "notes": None,
            "is_recurring": recurring,
            "year": when.year,
            "month": when.month,
            "day": when.day,
            "day_of_week": when.weekday(),
            "day_of_month": when.day,
            "week_of_year": int(iso[1]),
            "iso_year": int(iso[0]),
            "period_key": month_key(when),
            "week_key": week_key(when),
            "signed_amount": round(amount, 2) if kind == "income" else round(-amount, 2),
            "created_at": when_time,
            "updated_at": when_time,
        }

    # --------------------------------------------------------------- budgets
    def _user_budgets(self, profile: _UserProfile, categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Budgets derived from this user's *own* average spend.

        Setting the limit at a random fraction of actual behaviour is what
        produces a realistic class balance for the overspend classifier instead
        of an all-yes or all-no label column.
        """
        if self.rng.random() < 0.15:
            return []  # some users simply do not budget

        by_name: dict[str, ObjectId] = {c["name"]: c["_id"] for c in categories}
        spend = self._average_monthly_spend(profile)
        start = self.start
        docs: list[dict[str, Any]] = []

        candidates = [
            name
            for name in ("Groceries", "Dining Out", "Transport", "Entertainment", "Shopping", "Utilities")
            if spend.get(name, 0.0) > 0
        ]
        self.rng.shuffle(candidates)
        chosen = candidates[: self.rng.randint(1, min(3, len(candidates)))] or ["Groceries"]

        for name in chosen:
            actual = spend.get(name, 200.0)
            limit = max(20.0, actual * profile.budget_factor * self.rng.uniform(0.85, 1.15))
            docs.append(
                {
                    "_id": ObjectId(),
                    "user_id": profile.user_id,
                    "category_id": by_name.get(name),
                    "name": f"{name} budget",
                    "limit_amount": round(limit, 2),
                    "period": "monthly",
                    "start_date": datetime(start.year, start.month, 1),
                    "end_date": None,
                    "alert_threshold_pct": self.rng.choice((0.7, 0.8, 0.9)),
                    "rollover": False,
                    "is_active": True,
                    "last_alerted_at": None,
                    "created_at": profile.created_at,
                    "updated_at": profile.created_at,
                }
            )

        # A minority also cap total monthly spend.
        if self.rng.random() < 0.25:
            total = sum(spend.values()) or 2000.0
            docs.append(
                {
                    "_id": ObjectId(),
                    "user_id": profile.user_id,
                    "category_id": None,
                    "name": "Overall budget",
                    "limit_amount": round(max(100.0, total * self.rng.uniform(0.95, 1.2)), 2),
                    "period": "monthly",
                    "start_date": datetime(start.year, start.month, 1),
                    "end_date": None,
                    "alert_threshold_pct": 0.85,
                    "rollover": False,
                    "is_active": True,
                    "last_alerted_at": None,
                    "created_at": profile.created_at,
                    "updated_at": profile.created_at,
                }
            )
        return docs

    def _average_monthly_spend(self, profile: _UserProfile) -> dict[str, float]:
        totals: dict[str, float] = {}
        count = 0
        for period in self.periods:
            count += 1
            for category, per_month, mean, spread, _weekend, _rec in EXPENSE_BEHAVIOURS:
                totals[category] = totals.get(category, 0.0) + per_month * mean * (
                    profile.discretionary_bias if category not in ("Groceries", "Transport") else 1.0
                ) * profile.expense_scale
            for category, _day, amount, _spread in RECURRING_EXPENSES:
                totals[category] = totals.get(category, 0.0) + amount * profile.expense_scale
        return {name: value / max(1, count) for name, value in totals.items()}

    # ----------------------------------------------------------------- goals
    def _user_goals(self, profile: _UserProfile) -> list[dict[str, Any]]:
        if profile.goal_count <= 0:
            return []
        templates = (
            ("Emergency fund", 3000.0, 0.35),
            ("New laptop", 1800.0, 0.25),
            ("Holiday in Japan", 4200.0, 0.15),
            ("Car replacement", 9500.0, 0.05),
            ("Course certification", 900.0, 0.30),
            ("Wedding fund", 12000.0, 0.20),
        )
        docs: list[dict[str, Any]] = []
        chosen = self.rng.sample(templates, min(profile.goal_count, len(templates)))
        for name, base, start_ratio in chosen:
            months_out = self.rng.randint(3, 18)
            target_date = self.end_month + timedelta(days=months_out * 30)
            target = base * (0.6 + self.rng.random() * 0.8)
            current = round(target * start_ratio * self.rng.uniform(0.4, 1.0), 2)
            achieved = current >= target
            docs.append(
                {
                    "_id": ObjectId(),
                    "user_id": profile.user_id,
                    "name": name,
                    "target_amount": round(target, 2),
                    "current_amount": current,
                    "target_date": datetime(target_date.year, target_date.month, 15),
                    "description": f"Synthetic goal: {name}",
                    "color": self.rng.choice(("#22c55e", "#6366f1", "#f59e0b", "#a855f7", "#ec4899")),
                    "linked_category_id": None,
                    "status": "achieved" if achieved else "active",
                    "achieved_at": datetime(self.end_month.year, self.end_month.month, 15) if achieved else None,
                    "created_at": profile.created_at,
                    "updated_at": datetime(self.end_month.year, self.end_month.month, 15),
                }
            )
        return docs

    # ------------------------------------------------------------------ seed
    def seed(self, db: Any, *, reset: bool = False, batch_size: int = 1000) -> dict[str, int]:
        """Insert the fixture into any MongoDB-compatible database."""
        data = self.generate()
        counts: dict[str, int] = {}
        for collection, documents in data.items():
            handle = db[collection]
            if reset:
                handle.delete_many({})
            for start in range(0, len(documents), batch_size):
                handle.insert_many(documents[start : start + batch_size], ordered=False)
            counts[collection] = len(documents)
        return counts


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _month_range(anchor: date, months: int) -> list[str]:
    keys: list[str] = []
    year, month = anchor.year, anchor.month
    for _ in range(months):
        keys.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            month, year = 12, year - 1
    return list(reversed(keys))


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - timedelta(days=1)).day


def _poisson(rng: random.Random, mean: float) -> int:
    """Knuth's algorithm - fine for the small means used here."""
    if mean <= 0:
        return 0
    limit = 2.718281828459045 ** -mean
    count, cumulative = 0, rng.random()
    while cumulative > limit and count < 12:
        count += 1
        cumulative *= rng.random()
    return count


def build_fixture(**kwargs: Any) -> dict[str, list[dict[str, Any]]]:
    return SyntheticGenerator(**kwargs).generate()


def seed_mongomock(**kwargs: Any) -> tuple[Any, dict[str, int]]:
    """In-memory fixture - the default for tests and offline experiments."""
    import mongomock

    client = mongomock.MongoClient()
    generator = SyntheticGenerator(**kwargs)
    counts = generator.seed(client["mercy"], reset=True)
    return client, counts


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, ObjectId):
        return str(value)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mercy-ml-generate",
        description="Generate a realistic synthetic finance panel for training and demos.",
    )
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--months", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--uri", default=None, help="MongoDB URI; omit to only write --out")
    parser.add_argument("--db", default=None, help="Database name (defaults to MONGODB_DB_NAME)")
    parser.add_argument("--reset", action="store_true", help="Delete existing documents first")
    parser.add_argument("--out", type=Path, default=None, help="Also write JSON here (for fixtures)")
    args = parser.parse_args(argv)

    generator = SyntheticGenerator(
        n_users=args.users, months=args.months, seed=args.seed, currency=args.currency
    )

    if args.uri or args.db:
        from mercy_ml.data import get_client
        from mercy_ml.config import get_settings

        settings = get_settings()
        client = get_client()
        db = client[args.db or settings.mongo_db_name]
        counts = generator.seed(db, reset=args.reset)
        print(f"Seeded {db.name}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    else:
        data = _to_jsonable(generator.generate())
        counts = {k: len(v) for k, v in data.items()}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {args.out}")

    if not args.uri and not args.db and not args.out:
        print("Generated (not persisted): " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
