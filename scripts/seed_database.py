"""Create data/tech.db from the rules in data/db_Tech.sql.

Run once::

    python scripts/seed_database.py
    python scripts/seed_database.py --overwrite    # rebuild from scratch

The seed is fixed (42) so the committed database is reproducible — an unseeded
run would produce a different file every time and pollute the diff.

STATUS: scaffolding. The CLI is final; the seeding logic is not implemented yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.modules.database.seeder import seed, summarize  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing database.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed. Keep it fixed.")
    args = parser.parse_args(argv)

    db_path = Path(settings.db_url.removeprefix("sqlite:///"))
    print(f"Seeding {db_path} ...")

    rows = seed(db_path, seed_value=args.seed, overwrite=args.overwrite)
    print(f"Inserted {rows} rows.")
    print(summarize(db_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
