#!/usr/bin/env python3
"""
Production build helper: seed synthetic training data into the real MongoDB,
train all mercy_ml models, and write the artifacts to ML_ARTIFACT_DIR.

Run during the Render build step (before the app starts) so that trained
artifacts are in place when the first request hits the predictions endpoints.

Usage (from repo root):
    python scripts/build_train.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_train")

# ---------------------------------------------------------------------------
# Resolve paths regardless of CWD
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ML_DIR = REPO_ROOT / "ml"
ARTIFACT_DIR = Path(os.environ.get("ML_ARTIFACT_DIR", str(ML_DIR / "artifacts")))

logger.info("Repo root   : %s", REPO_ROOT)
logger.info("Artifact dir: %s", ARTIFACT_DIR)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Point mercy_ml at the right artifact directory before importing anything
# ---------------------------------------------------------------------------
os.environ.setdefault("ML_ARTIFACT_DIR", str(ARTIFACT_DIR))

# Ensure both backend and ml/src are importable
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(ML_DIR / "src"))

# ---------------------------------------------------------------------------
# Synthetic data → real MongoDB → train
# ---------------------------------------------------------------------------
try:
    import pymongo
    from mercy_ml.synthetic import generate_panel
    from mercy_ml.data import set_client
    from mercy_ml.train import run_training

    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    mongo_db  = os.environ.get("MONGO_DB_NAME", "mercy")

    logger.info("Connecting to MongoDB at %s (db=%s) …", mongo_uri, mongo_db)
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
    db = client[mongo_db]

    # Wire mercy_ml to use the same client
    set_client(client)

    # Check if we already have enough transaction history to train on
    tx_count = db["transactions"].count_documents({})
    logger.info("Existing transactions in DB: %d", tx_count)

    if tx_count < 200:
        logger.info("Not enough data — generating synthetic panel (20 users × 18 months) …")
        panel = generate_panel(n_users=20, months=18, seed=42)
        for collection, docs in panel.items():
            if docs:
                db[collection].insert_many(docs)
                logger.info("  Inserted %d docs into '%s'", len(docs), collection)
    else:
        logger.info("Sufficient transaction history found — skipping synthetic generation.")

    # Run full training
    logger.info("Starting model training …")
    summary = run_training(overwrite=True)
    logger.info("Training complete: %s", summary)

    # Verify artifacts were written
    artifacts = list(ARTIFACT_DIR.glob("*.joblib"))
    if artifacts:
        logger.info("Artifacts written (%d files):", len(artifacts))
        for a in artifacts:
            logger.info("  %s  (%.1f MB)", a.name, a.stat().st_size / 1e6)
    else:
        logger.warning("No .joblib artifacts found in %s after training!", ARTIFACT_DIR)
        sys.exit(1)

except Exception:
    logger.exception("Build training failed")
    # Don't hard-fail the build if training errors — the app can still serve
    # non-ML features. Log the error loudly so it's visible in Render logs.
    logger.error(
        "⚠️  ML training failed. Predictions endpoints will return 503 until "
        "models are available. Check the build log for details."
    )
    sys.exit(0)  # exit 0 so the Render build continues
