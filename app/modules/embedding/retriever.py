"""Runtime retrieval from the Chroma vector database.

Chroma is the vector store, as the spec requires. This module resolves the
collection in two steps:

1. **Use the persisted database** at ``settings.chroma_path``, written by the
   offline step. This is the normal local path.
2. **Rebuild it from the committed JSON seed** when that directory is absent —
   a fresh checkout, or Streamlit Community Cloud, which cannot run the offline
   script. The rebuild goes through ``chromadb.PersistentClient`` and
   ``collection.add``, so what gets queried is a real Chroma database either way.

Query pattern from ``Course23/Embedding & Retrieval.ipynb``: embed the query with
the SAME model used for indexing, then ``collection.query(query_embeddings=...)``
and slice the results out of the returned dict.

Chroma returns cosine **distance**, not similarity: range 0-2, lower is better,
``distance = 1 - cosine_similarity``. Any threshold written here must respect that
direction (ENGINEERING.md pitfall 7).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from functools import lru_cache

import chromadb

from app.config import settings
from app.modules.common import CHROMA_SETTINGS, get_openai_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned by a similarity search.

    Attributes:
        chunk_id: The id assigned at indexing time.
        text: The chunk content.
        distance: Cosine distance, 0-2. Lower is more similar.
    """

    chunk_id: str
    text: str
    distance: float


#: Building a Chroma client is not thread-safe: concurrent constructions race on
#: Chroma's Rust bindings and fail with a bogus "tenant default_tenant does not
#: exist" error. The evaluation notebook fans out over decision points, so
#: serialise the first construction. `lru_cache` alone does not help — it does not
#: lock, so several threads can enter the factory at once.
_COLLECTION_LOCK = threading.Lock()


def get_collection():
    """Return the Chroma collection, opening or rebuilding it on first use.

    Thread-safe and cached.

    Raises:
        RuntimeError: If neither the persisted database nor the JSON seed exists,
            or if the seed was built with a different embedding model.
    """
    with _COLLECTION_LOCK:
        return _open_collection()


@lru_cache(maxsize=1)
def _open_collection():
    """Resolve the collection. Always called with :data:`_COLLECTION_LOCK` held."""
    collection = _open_persisted()
    if collection is not None:
        return collection

    logger.info(
        "No Chroma database at %s; rebuilding it from %s",
        settings.chroma_path,
        settings.vector_store_json,
    )
    return _rebuild_from_seed()


def _open_persisted():
    """Open the persisted collection, or return None when it is unusable."""
    if not settings.chroma_path.exists():
        return None
    try:
        client = chromadb.PersistentClient(
            path=str(settings.chroma_path), settings=CHROMA_SETTINGS
        )
        collection = client.get_collection(settings.chroma_collection)
    except Exception:  # noqa: BLE001 - chroma raises several types for "absent"
        return None
    return collection if collection.count() else None


def _load_seed() -> dict:
    """Read and validate the committed JSON seed."""
    path = settings.vector_store_json
    if not path.is_file():
        raise RuntimeError(
            f"No Chroma database at {settings.chroma_path} and no seed at {path}. "
            "Run: python scripts/build_vector_store.py"
        )

    seed = json.loads(path.read_text(encoding="utf-8"))
    if not seed.get("documents"):
        raise RuntimeError(f"{path} contains no documents. Rebuild it.")

    indexed_model = seed.get("model")
    if indexed_model and indexed_model != settings.embedding_model:
        # Mixing models produces silently meaningless distances, so fail loudly
        # instead (ENGINEERING.md pitfall 6).
        raise RuntimeError(
            f"{path} was built with {indexed_model!r} but OPENAI_EMBEDDING_MODEL is "
            f"{settings.embedding_model!r}. Rebuild the index or restore the model."
        )
    return seed


def _rebuild_from_seed():
    """Populate a persisted Chroma database from the JSON seed and return it."""
    seed = _load_seed()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(settings.chroma_path), settings=CHROMA_SETTINGS
    )
    collection = client.get_or_create_collection(name=settings.chroma_collection)
    if collection.count() == 0:
        collection.add(
            ids=seed["ids"],
            documents=seed["documents"],
            embeddings=seed["embeddings"],
        )
    return collection


def retrieve(query: str, n_results: int = 3) -> list[RetrievedChunk]:
    """Return the chunks most relevant to a query.

    Args:
        query: The candidate's question, or a rewritten form of it.
        n_results: How many chunks to return.

    Returns:
        Chunks ordered by increasing distance (most relevant first). Empty when
        the query is blank — the embeddings endpoint rejects an empty string.
    """
    if not query or not query.strip():
        return []

    collection = get_collection()
    # Same embedding model as indexing — enforced when loading the seed.
    query_embedding = (
        get_openai_client()
        .embeddings.create(input=[query], model=settings.embedding_model)
        .data[0]
        .embedding
    )
    results = collection.query(
        query_embeddings=[list(query_embedding)],
        n_results=min(n_results, collection.count()),
        include=["documents", "distances"],
    )
    return [
        RetrievedChunk(chunk_id=cid, text=doc, distance=dist)
        for cid, doc, dist in zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        )
    ]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as the context block to stuff into a prompt.

    Mirrors the course's XML source formatting
    (``<sources><result id=...><content>...</content></result></sources>``), which
    keeps the boundary between retrieved text and instructions unambiguous to the
    model.
    """
    if not chunks:
        return "<sources></sources>"
    body = "".join(
        f"<result id='{c.chunk_id}'><content>{c.text}</content></result>" for c in chunks
    )
    return f"<sources>{body}</sources>"
