"""Artifact registry: save/load models, write the manifest and run records.

The manifest is a JSON snapshot of *everything* that trained, so a training run
is reproducible and auditable even when nobody remembers which flag produced it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from mercy_ml.config import artifact_name, ensure_artifact_dir, get_settings, manifest_name
from mercy_ml.schema import contract_digest

logger = logging.getLogger(__name__)

#: name -> class used to reload an artifact.
LOADERS: dict[str, type] = {}


def register(name: str):
    def decorator(cls: type) -> type:
        LOADERS[name] = cls
        return cls

    return decorator


def save_model(model: Any, *, overwrite: bool = True) -> Path:
    """Persist a fitted model to the artifact directory."""
    settings = get_settings()
    ensure_artifact_dir()
    path = settings.artifact(artifact_name(model.name))
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
    return model.save(path)


def load_model(name: str) -> Any | None:
    """Reload a persisted model by its `name` (e.g. 'expense_forecaster')."""
    settings = get_settings()
    path = settings.artifact(artifact_name(name))
    if not path.exists():
        return None
    cls = LOADERS.get(name, _infer_loader(name))
    if cls is None:
        logger.warning("No registered loader for artifact %r; returning the raw bundle", name)
        return joblib.load(path)
    return cls.load(path)


def _infer_loader(name: str) -> type | None:
    from mercy_ml.models import (
        BudgetRecommender,
        ExpenseForecaster,
        OverspendClassifier,
        SavingsRecommender,
        SpendingClusterer,
    )

    table = {
        "spending_clusters": SpendingClusterer,
        "expense_forecaster": ExpenseForecaster,
        "overspend_classifier": OverspendClassifier,
        "budget_recommender": BudgetRecommender,
        "savings_recommender": SavingsRecommender,
    }
    return table.get(name)


def available_models() -> list[dict[str, Any]]:
    settings = get_settings()
    ensure_artifact_dir()
    out: list[dict[str, Any]] = []
    for path in sorted(settings.artifact_path.glob("*.joblib")):
        try:
            bundle = joblib.load(path)
        except Exception as exc:  # pragma: no cover - corrupt artifact
            logger.warning("Skipping corrupt artifact %s: %s", path, exc)
            continue
        version = bundle.get("version", "unknown")
        report = bundle.get("report") or {}
        metrics = report.get("metrics") or {}
        out.append(
            {
                "name": bundle.get("name", path.stem),
                "version": version,
                "path": str(path),
                "trained_at": report.get("trained_at"),
                "n_samples": report.get("n_samples"),
                "n_users": report.get("n_users"),
                "metrics": {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
            }
        )
    return out


def write_manifest(contents: dict[str, Any]) -> Path:
    settings = get_settings()
    ensure_artifact_dir()
    path = settings.artifact(manifest_name())
    payload = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "feature_contracts": contract_digest(),
        **contents,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote manifest %s", path)
    return path


def record_run(db: Any, *, status: str, models: list[dict[str, Any]], errors: list[str], **extra: Any) -> Any:
    """Write one `model_runs` document so the backend can surface run health."""
    doc: dict[str, Any] = {
        "status": status,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "models": models,
        "errors": errors,
        "meta": extra,
        "created_at": datetime.now(timezone.utc),
    }
    result = db["model_runs"].insert_one(doc)
    logger.info("Recorded model run %s (%s)", result.inserted_id, status)
    return result.inserted_id


def metrics_summary(report: Any) -> dict[str, Any]:
    if report is None:
        return {}
    return {k: v for k, v in (report.metrics or {}).items() if not isinstance(v, (list, dict))}


__all__ = [
    "available_models",
    "load_model",
    "manifest_name",
    "metrics_summary",
    "record_run",
    "register",
    "save_model",
    "write_manifest",
]