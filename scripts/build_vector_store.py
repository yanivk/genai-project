"""Build the vector store from the job description PDF.

The spec's offline embedding step. Run once::

    python scripts/build_vector_store.py

Writes data/vector_store.json, which IS committed so Streamlit Community Cloud —
which cannot run this script — still has an index to query. The retriever rebuilds
an ephemeral Chroma collection from it at first use.

Costs a small number of embedding tokens. Nothing else in the project triggers it.
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
    parser.parse_args(argv)

    settings.require_api_key()

    print(f"PDF        : {settings.job_description_pdf}")
    print(f"Model      : {settings.embedding_model}")
    print(f"Output     : {settings.vector_store_json}")
    print(f"Collection : {settings.chroma_collection}")

    count = build_index()
    size_kb = settings.vector_store_json.stat().st_size / 1024
    print(f"Indexed {count} chunks -> {size_kb:,.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
