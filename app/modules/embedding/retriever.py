"""Runtime retrieval against the persisted Chroma collection.

Query pattern from ``Course23/Embedding & Retrieval.ipynb``: embed the query with
the SAME model used for indexing, then ``collection.query(query_embeddings=...)``
and slice the results out of the returned dict.

Chroma returns cosine **distance**, not similarity: range 0-2, lower is better,
``distance = 1 - cosine_similarity``. Any threshold written here must respect
that direction (CLAUDE.md pitfall 7).

STATUS: scaffolding. Signatures are final; bodies are not implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass


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


def get_collection():
    """Open the persisted Chroma collection.

    Returns:
        The ``settings.chroma_collection`` collection from
        ``settings.chroma_path``.

    Raises:
        RuntimeError: If the index does not exist yet, pointing the caller at
            ``python scripts/build_vector_store.py``.
    """
    raise NotImplementedError


def retrieve(query: str, n_results: int = 3) -> list[RetrievedChunk]:
    """Return the chunks most relevant to a query.

    Args:
        query: The candidate's question, or a rewritten form of it.
        n_results: How many chunks to return.

    Returns:
        Chunks ordered by increasing distance (most relevant first).
    """
    raise NotImplementedError


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as the context block to stuff into a prompt.

    Mirrors the course's XML source formatting
    (``<sources><result id=...><content>...</content></result></sources>``), which
    keeps the boundary between retrieved text and instructions unambiguous to the
    model.
    """
    raise NotImplementedError
