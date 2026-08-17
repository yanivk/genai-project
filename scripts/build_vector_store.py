"""Build the Chroma index from the job description PDF.

The spec's offline embedding step. Run once::

    python scripts/build_vector_store.py

Writes into data/chroma/, which is committed so Streamlit Community Cloud — which
cannot run this script — still has an index to query.

Costs a small number of embedding tokens. Nothing else in the project triggers it.

STATUS: scaffolding. The CLI is final; the indexing logic is not implemented yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.modules.embedding.indexer import build_index  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Append to the existing collection instead of rebuilding it. Risks duplicates.",
    )
    args = parser.parse_args(argv)

    settings.require_api_key()

    print(f"PDF        : {settings.job_description_pdf}")
    print(f"Model      : {settings.embedding_model}")
    print(f"Chroma path: {settings.chroma_path}")
    print(f"Collection : {settings.chroma_collection}")

    count = build_index(reset=not args.keep)
    print(f"Indexed {count} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
