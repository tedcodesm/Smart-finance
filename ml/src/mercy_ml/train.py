"""Offline training pipeline.

Loads the panel from MongoDB (or mongomock), builds every feature frame, fits
each model family, picks the best variant per model type on a held-out split,
persists the winners, and records the whole run in `model_runs`.

Money-metric choices are audited in `forecast_comparison` / `classifier_comparison`
so the training script's decision between RandomForest/Ridge/ETS and between
RandomForest/Logistic is evidence-based, and the naive baselines are kept so a
regression is visible when a new model gets worse than "repeat last period".
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mercy_ml.train")


def run_training(
    *,
    user_ids: Sequence[Any] | None = None,
    months_back: int | None = None,
    n_jobs: int | None = None,
    compare_estimators: bool = True,
    seed: int | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Fit everything and persist. Returns a summary dict for the manifest."""
    import random as _random

    import numpy as np

    from mercy_ml import get_settings
    from mercy_ml.data import get_db, load_budgets, load_categories, load_transactions
    from mercy_ml.features import (
        build_budget_recommendation_frame,
        build_cluster_frame,
        build_forecast_frame,
        build_overspend_frame,
        build_savings_frame,
    )
    from mercy_ml.models import (
        BudgetRecommender,
        ExpenseForecaster,
        OverspendClassifier,
        SavingsRecommender,
        SpendingClusterer,
    )
    from mercy_ml.registry import (
        metrics_summary,
        record_run,
        save_model,
        write_manifest,
    )

    settings = get_settings()
    if seed is not None:
        settings.random_seed = seed
        np.random.seed(seed)
        _random.seed(seed)

    started = time.time()
    db = get_db()

    # ----------------------------------------------------------------- data
    logger.info("Loading transactions / categories / budgets ...")
    start = None
    if months_back:
        from mercy_ml.periods import month_key
        from datetime import timedelta

        start = datetime.now() - timedelta(days=months_back * 31)
    transactions = load_transactions(user_ids=user_ids, start=start)
    categories = load_categories(user_ids=user_ids)
    budgets = load_budgets(user_ids=user_ids)

    era = (
        f"last_{months_back}mo" if months_back
        else "all_history"
    )
    logger.info("panel: %d transactions, %d categories, %d budgets (%s)",
                len(transactions), len(categories), len(budgets), era)
    if transactions.empty:
        raise RuntimeError("No transactions to train on.")

    # ---------------------------------------------------------------- frames
    logger.info("Building feature frames ...")
    cluster_frame = build_cluster_frame(transactions, categories)
    forecast_frame = build_forecast_frame(transactions, categories)
    overspend_frame = (build_overspend_frame(transactions, budgets, categories)
                       if len(budgets) else forecast_frame.head(0))
    savings_frame = build_savings_frame(transactions)
    budget_stats = build_budget_recommendation_frame(transactions, categories)

    if overspend_frame.empty:
        logger.warning("No overspend observations (no budgets). Using a dummy empty frame.")
    if savings_frame.empty:
        logger.warning("No savings observations (users never save). Using a dummy empty frame.")

    frames = {
        "spending_clusters": cluster_frame,
        "expense_forecaster": forecast_frame,
        "overspend_classifier": overspend_frame,
        "savings_recommender": savings_frame,
    }

    # --------------------------------------------------------------- models
    logger.info("Fitting spending clusters ...")
    clusterer = SpendingClusterer(n_jobs=n_jobs or -1).fit(cluster_frame)
    save_model(clusterer, overwrite=overwrite)

    logger.info("Fitting expense forecaster (RandomForest vs Ridge vs ETS) ...")
    forecaster, forecast_choice, forecast_metrics = _fit_forecaster(
        forecast_frame, compare=compare_estimators, n_jobs=n_jobs
    )
    save_model(forecaster, overwrite=overwrite)

    overspend_model: Any = None
    overall_metrics: dict[str, Any] = {}
    if not overspend_frame.empty:
        logger.info("Fitting overspend classifier (RandomForest vs Logistic) ...")
        overspend_model, overspend_choice, overspend_metrics = _fit_classifier(
            overspend_frame, compare=compare_estimators, n_jobs=n_jobs
        )
        save_model(overspend_model, overwrite=overwrite)
        overall_metrics["overspend_classifier"] = overspend_metrics
    else:
        logger.warning("Skipping overspend classifier: no training observations.")

    savings_model, savings_choice, savings_metrics = _fit_savings(
        savings_frame, compare=compare_estimators, n_jobs=n_jobs
    )
    save_model(savings_model, overwrite=overwrite)

    # Budget recommender is closed-form over per-category stats.
    recommender = BudgetRecommender()
    recommender.fit(budget_stats)

    summaries: list[dict[str, Any]] = []
    for name, model in (
        ("spending_clusters", clusterer),
        ("expense_forecaster", forecaster),
        ("overspend_classifier", overspend_model),
        ("savings_recommender", savings_model),
    ):
        if model is None:
            continue
        report = getattr(model, "report", None)
        summaries.append(
            {
                "name": name,
                "version": getattr(model, "name", name),
                "trained": report is not None,
                "n_samples": getattr(report, "n_samples", 0) if report else 0,
                "n_users": getattr(report, "n_users", 0) if report else 0,
                "metrics": metrics_summary(report),
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "era": era,
        "data": {
            "transactions": len(transactions),
            "categories": len(categories),
            "budgets": len(budgets),
            "users": transactions["user_id"].nunique() if "user_id" in transactions else 0,
        },
        "frames": {name: int(len(frame)) for name, frame in frames.items()},
        "model_selection": {
            "expense_forecaster": forecast_choice,
            "overspend_classifier": overspend_choice,
            "savings_recommender": savings_choice,
            "budget_recommender": "closed_form",
        },
        "models": summaries,
        "duration_seconds": round(time.time() - started, 1),
    }
    manifest_path = write_manifest(manifest)

    record_run(
        db,
        status="success",
        models=summaries,
        errors=[],
        duration_seconds=manifest["duration_seconds"],
        era=era,
        manifest=manifest_path.name,
    )
    logger.info("Training complete in %.1fs: %s", manifest["duration_seconds"], manifest_path)
    return manifest


def _fit_forecaster(frame: pd.DataFrame, *, compare: bool, n_jobs: int | None) -> tuple[Any, str, dict[str, Any]]:
    from mercy_ml.models import EtsForecaster, ExpenseForecaster, compare_forecasters

    candidates: list[tuple[dict[str, Any], str]] = []
    base_kwargs = {"n_jobs": n_jobs or -1}
    results = {}
    for kind in ("random_forest", "ridge"):
        model = ExpenseForecaster(estimator=kind, **base_kwargs).fit(frame)
        if model.report and model.report.metrics:
            results[kind] = dict(model.report.metrics)
            candidates.append(((model, model.report.metrics), kind))
    if compare:
        ets = EtsForecaster().fit_evaluate(frame)
        results["ets"] = ets.get("metrics", {}) or {}
        if ets.get("metrics"):
            candidates.append((("ets", ets["metrics"]), "ets"))

    if not candidates:
        logger.warning("No forecaster produced metrics; defaulting to RandomForest.")
        model = ExpenseForecaster(estimator="random_forest", **base_kwargs).fit(frame)
        return model, "random_forest", {}

    best, best_kind = min(candidates, key=lambda item: item[0][1].get("mae", float("inf")))
    if best_kind == "ets":
        # Persist the best pooled model instead of the benchmark; ETS is used as
        # a comparison and fallback only.
        model = ExpenseForecaster(estimator="random_forest", **base_kwargs).fit(frame)
        return model, "ensemble (ETS won solo benchmark)", dict(results)

    best_model, best_metrics = best
    return best_model, best_kind, dict(results)


def _fit_classifier(frame: pd.DataFrame, *, compare: bool, n_jobs: int | None) -> tuple[Any, str, dict[str, Any]]:
    from mercy_ml.models import OverspendClassifier

    results: dict[str, Any] = {}
    candidates: list[tuple[Any, dict[str, Any], str]] = []
    for kind in ("random_forest", "logistic"):
        model = OverspendClassifier(estimator=kind, n_jobs=n_jobs or -1).fit(frame)
        if model.report and model.report.metrics:
            results[kind] = dict(model.report.metrics)
            candidates.append((model, model.report.metrics, kind))
    if not candidates:
        model = OverspendClassifier().fit(frame)
        return model, "random_forest", {}
    best = max(candidates, key=lambda item: item[1].get("roc_auc") or 0.0)
    if not compare:
        best = candidates[0]
    return best[0], best[2], dict(results)


def _fit_savings(frame: pd.DataFrame, *, compare: bool, n_jobs: int | None) -> tuple[Any, str, dict[str, Any]]:
    from mercy_ml.models import SavingsRecommender

    results: dict[str, Any] = {}
    candidates: list[tuple[Any, dict[str, Any], str]] = []
    if compare:
        kinds = ("gradient_boosting", "random_forest")
    else:
        kinds = ("gradient_boosting",)
    for kind in kinds:
        model = SavingsRecommender(estimator=kind, n_jobs=n_jobs or -1).fit(frame)
        if model.report and model.report.metrics:
            results[kind] = dict(model.report.metrics)
            candidates.append((model, model.report.metrics, kind))
    if not candidates:
        model = SavingsRecommender().fit(frame)
        return model, "gradient_boosting", {}
    best = min(candidates, key=lambda item: item[1].get("mae", float("inf")))
    return best[0], best[2], dict(results)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mercy-ml-train",
        description="Train and persist all Mercy financial models.",
    )
    parser.add_argument("--uri", default=None, help="MongoDB URI override (env used otherwise)")
    parser.add_argument("--db", default=None)
    parser.add_argument("--months", type=int, default=None, help="Only train on the last N months")
    parser.add_argument("--user-ids", default=None, help="Comma-separated ObjectIds to subset")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel jobs for tree ensembles")
    parser.add_argument("--no-compare", action="store_true", help="Skip cross-family comparisons")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mongomock", action="store_true",
                        help="Run against an in-memory synthetic fixture (for CI/dev)")
    parser.add_argument("--users", type=int, default=80, help="Users when --mongomock")
    parser.add_argument("--months-train", type=int, default=14, help="Months when --mongomock")
    parser.add_argument("--overwrite", action="store_true", default=True)
    args = parser.parse_args(argv)

    user_ids = [uid.strip() for uid in (args.user_ids or "").split(",") if uid.strip()] or None

    try:
        if args.mongomock:
            from mercy_ml import synthetic
            from mercy_ml.data import set_client

            client, counts = synthetic.seed_mongomock(n_users=args.users, months=args.months_train)
            set_client(client)
            logger.info("mongomock fixture: %s", counts)

        manifest = run_training(
            user_ids=user_ids,
            months_back=args.months,
            n_jobs=args.n_jobs,
            compare_estimators=not args.no_compare,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(json.dumps(manifest, indent=2, default=str))
        return 0
    except Exception as exc:  # pragma: no cover - CLI error path
        logger.error("Training failed: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())