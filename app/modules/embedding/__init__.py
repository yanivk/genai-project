"""Offline embedding pipeline and runtime retrieval.

    indexer    PDF -> text -> chunks -> OpenAI embeddings -> Chroma. Offline.
    retriever  Query-time lookup against the persisted collection.

The spec calls for the embedding step to run "separately from the main process",
so indexing is only ever triggered by ``scripts/build_vector_store.py``.
Importing this package must not build anything (ENGINEERING.md section 2, rule 5).
"""

from app.modules.embedding.retriever import retrieve

__all__ = ["retrieve"]
