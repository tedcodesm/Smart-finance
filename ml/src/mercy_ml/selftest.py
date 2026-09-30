"""Offline smoke test: synthetic panel -> feature frames -> model fit -> inference.

Run with:  .\\.venv\\Scripts\\python.exe -m mercy_ml.selftest
"""

from __future__ import annotations

import logging
import sys
import traceback
from datetime import date

import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def step(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    from mercy_ml import synthetic
    from mercy_ml.data import set_client

    # ---------------------------------------------------------------- fixture
    step("1. Synthetic fixture")
    client, counts = synthetic.seed_mongomock(n_users=40, months=14)
    set_client(client)
    db = client["mercy"]
    for name, n in counts.items():
        print(f"  {name}: {n}")
    check("transactions generated", counts["transactions"] > 2000, f"{counts['transactions']}")
    check("budgets generated", counts["budgets"] > 10, f"{counts['budgets']}")
    check("goals generated", counts["savings_goals"] > 5, f"{counts['savings_goals']}")

    periods = sorted({t["period_key"] for t in db.transactions.find({}, {"period_key": 1})})
    print(f"  periods: {periods[0]} .. {periods[-1]} ({len(periods)})")
    check("multiple periods", len(periods) >= 12)

    # ----------------------------------------------------------------- frames
    step("2. Feature frames")
    from mercy_ml.data import load_budgets, load_categories, load_transactions

    transactions = load_transactions()
    categories = load_categories()
    budgets = load_budgets()
    print(f"  transactions={transactions.shape} categories={categories.shape} budgets={budgets.shape}")

    from mercy_ml.features import (
        build_budget_recommendation_frame,
        build_cluster_frame,
        build_forecast_frame,
        build_overspend_frame,
        build_savings_frame,
    )

    cluster = build_cluster_frame(transactions, categories)
    forecast = build_forecast_frame(transactions, categories)
    overspend = build_overspend_frame(transactions, budgets, categories)
    savings = build_savings_frame(transactions)
    budget_stats = build_budget_recommendation_frame(transactions, categories)

    for name, frame, target in (
        ("cluster", cluster, None),
        ("forecast", forecast, "target_next_expense"),
        ("overspend", overspend, "label_overspend"),
        ("savings", savings, "target_next_net_savings"),
    ):
        n = len(frame)
        check(f"{name} frame non-empty", n > 0, f"{frame.shape}")
        if n:
            nan_ratio = float(frame[frame.columns].isna().mean().mean())
            check(f"{name} frame mostly populated", nan_ratio < 0.6, f"mean NaN ratio {nan_ratio:.2f}")
        if target:
            if target in frame:
                vals = pd.to_numeric(frame[target], errors="coerce")
                check(f"{name} target sane", vals.notna().any(), f"min={vals.min()} max={vals.max()}")
            else:
                check(f"{name} target present", False, f"missing {target}")

    if len(overspend):
        rate = float(overspend["label_overspend"].mean())
        print(f"  overspend positive rate: {rate:.3f}")
        check("overspend labels are mixed", 0.02 < rate < 0.98, f"{rate:.3f}")

    # ----------------------------------------------------------------- models
    step("3. Model training")
    from mercy_ml.models import (
        BudgetRecommender,
        ExpenseForecaster,
        OverspendClassifier,
        SavingsRecommender,
        SpendingClusterer,
    )

    clusterer = SpendingClusterer().fit(cluster)
    check("clusterer trained", clusterer.is_trained)
    if clusterer.report:
        m = clusterer.report.metrics
        print(f"  k={m.get('n_clusters')} silhouette={m.get('silhouette_score')} db={m.get('davies_bouldin')}")
        print(f"  sizes={m.get('cluster_sizes')}")
        for profile in clusterer.describe():
            print(f"   - [{profile['cluster_id']}] {profile['label']} ({profile['size']} txns, {profile['share_pct']}%)")

    forecaster = ExpenseForecaster().fit(forecast)
    check("forecaster trained", forecaster.is_trained)
    if forecaster.report:
        m = forecaster.report.metrics
        print(f"  MAE={m.get('mae')} RMSE={m.get('rmse')} R2={m.get('r2')}")
        print(f"  naive MAE={m.get('naive_lag1_mae')} skill={m.get('mae_skill_vs_naive')}")

    classifier = OverspendClassifier().fit(overspend)
    check("classifier trained", classifier.is_trained)
    if classifier.report:
        m = classifier.report.metrics
        print(f"  ROC-AUC={m.get('roc_auc')} F1={m.get('f1')} precision={m.get('precision')} recall={m.get('recall')}")
        print(f"  confusion={m.get('confusion_matrix')}")
        print(f"  top features={m.get('top_features')}")

    saver = SavingsRecommender().fit(savings)
    check("savings model trained", saver.is_trained)
    if saver.report:
        m = saver.report.metrics
        print(f"  MAE={m.get('mae')} R2={m.get('r2')} skill={m.get('mae_skill_vs_naive')}")

    recommender = BudgetRecommender().fit(budget_stats)
    check("budget recommender ready", recommender.is_trained())

    # ------------------------------------------------------------- inference
    step("4. Inference on a live user")
    from bson import ObjectId

    from mercy_ml.features import (
        build_forecast_inference_row,
        build_overspend_inference_row,
        build_savings_inference_row,
    )

    user = db.users.find_one({})
    user_id = user["_id"]
    per_user = transactions[transactions["user_id"] == user_id]
    print(f"  user={user['email']} history rows={len(per_user)}")
    current_period = periods[-1]
    next_period = synthetic._month_range(date(2026, 1, 1), 1)  # placeholder, replaced below
    from mercy_ml.periods import next_period_key

    next_period = next_period_key("monthly", current_period)
    names = dict(zip(categories["category_id"], categories["category_name"]))

    top_category = (
        per_user[per_user["type"] == "expense"]
        .groupby("category_id")["amount"]
        .sum()
        .idxmax()
    )
    forecast_row = pd.DataFrame(
        [
            build_forecast_inference_row(
                user_id=user_id,
                category_id=top_category,
                category_name=names.get(top_category, "Unknown"),
                period_key=current_period,
                transactions=transactions,
                categories=categories,
            )
        ]
    )
    if forecaster.is_trained:
        result = forecaster.predict_with_interval(forecast_row)
        print(f"  forecast {names.get(top_category)} -> {result}")
        check("forecast is a number", result["predicted_amount"] >= 0)
        check("interval brackets point", result["lower_bound"] <= result["predicted_amount"] + 1e-6,
              f"{result['lower_bound']} <= {result['predicted_amount']} <= {result['upper_bound']}")

    active_budget = db.budgets.find_one({"user_id": user_id, "is_active": True})
    if active_budget and classifier.is_trained:
        spent = float(
            per_user[
                (per_user["period_key"] == current_period)
                & (per_user["type"] == "expense")
                & (
                    (per_user["category_id"] == active_budget["category_id"])
                    if active_budget["category_id"] is not None
                    else pd.Series(True, index=per_user.index)
                )
            ]["amount"].sum()
        )
        risk_row = pd.DataFrame(
            [
                build_overspend_inference_row(
                    user_id=user_id,
                    category_id=active_budget["category_id"],
                    category_name=names.get(active_budget["category_id"], "Overall"),
                    limit_amount=float(active_budget["limit_amount"]),
                    spent_so_far=spent,
                    as_of=date.today(),
                    period="monthly",
                    period_key=current_period,
                    transactions=transactions,
                    categories=categories,
                )
            ]
        )
        risk = classifier.predict_risk(risk_row)
        print(f"  overspend risk -> {risk}")
        print(f"  top contributors -> {classifier.explain(risk_row, top=5)}")
        check("risk probability in range", 0.0 <= risk["probability"] <= 1.0)
        check("risk band assigned", risk["label"] in {"low", "medium", "high"}, risk["label"])

    savings_row = pd.DataFrame(
        [build_savings_inference_row(user_id=user_id, period_key=current_period, transactions=transactions)]
    )
    if saver.is_trained:
        rec = saver.recommend(
            savings_row,
            target_amount=3000.0,
            current_amount=900.0,
            active_goals=1,
            months_remaining=8,
        )
        print(f"  savings -> {rec}")
        check("savings suggestion is sane", rec["suggested_contribution"] >= 0)

    stats = budget_stats[budget_stats["user_id"].astype(str) == str(user_id)]
    if len(stats):
        suggestion = recommender.recommend(stats.iloc[0], current_limit=400.0)
        print(f"  budget -> {suggestion}")
        check("budget suggestion positive", suggestion["recommended_limit"] > 0)

    if clusterer.is_trained and len(per_user):
        profile = clusterer.user_cluster_profile(cluster[cluster["user_id"].astype(str) == str(user_id)], str(user_id))
        print(f"  clusters -> {profile}")
        check("user has cluster assignments", len(profile) > 0)

    # ---------------------------------------------------------------- results
    step("5. Results")
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # pragma: no cover - diagnostic path
        traceback.print_exc()
        sys.exit(2)
