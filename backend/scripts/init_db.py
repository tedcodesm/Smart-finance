"""CLI helper to initialise MongoDB indexes and seed system data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Running this file directly puts `scripts/` on sys.path, not `backend/`, so the
# `app` package would be unimportable without this bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.client import close_client, get_db  # noqa: E402
from app.migrations import ensure_indexes, ensure_indexes_and_seeds, seed_system_categories  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialise Mercy database")
    parser.add_argument("--seed-only", action="store_true", help="Only seed system categories")
    parser.add_argument("--indexes-only", action="store_true", help="Only ensure indexes")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Also seed the demo account with ~8 months of history. Works against any database.",
    )
    args = parser.parse_args(argv)

    db = get_db()
    try:
        if args.indexes_only:
            result = {"indexes": ensure_indexes(db)}
        elif args.seed_only:
            result = {"system_categories_seeded": seed_system_categories(db)}
        else:
            result = ensure_indexes_and_seeds(db)
        if args.demo:
            from app.demo_seed import seed_demo_data

            demo = seed_demo_data(db)
            result["demo"] = demo if demo is not None else "already present; skipped"
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"init-db failed: {exc}")
        return 1
    finally:
        close_client()


if __name__ == "__main__":
    raise SystemExit(main())