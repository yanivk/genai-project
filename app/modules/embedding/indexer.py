"""Offline: build the Chroma index from the job description PDF.

Follows the course pattern from ``Course23/Embedding & Retrieval.ipynb``: raw
``chromadb`` with explicitly computed OpenAI embeddings, and ``PyPDF2.PdfReader``
for text extraction. No LangChain vectorstore wrapper, no ``PyPDFLoader``, no
``RecursiveCharacterTextSplitter`` — none of those were taught (CLAUDE.md 4.6).

One deliberate deviation: ``chromadb.PersistentClient`` instead of the course's
ephemeral ``chromadb.Client()``. The spec requires an offline embedding step
whose index is reused by the running app, and Streamlit Community Cloud needs the
index to ship inside the repo.

Run via ``python scripts/build_vector_store.py``.

STATUS: scaffolding. Signatures are final; bodies are not implemented yet.
"""

from __future__ import annotations

from pathlib import Path

#: Chunk size in characters. The course never chunked, so this is ours to pick:
#: large enough to keep a requirement and its context together, small enough
#: that a retrieved chunk is mostly relevant.
CHUNK_SIZE = 1000

#: Overlap in characters, so a requirement split across a boundary still appears
#: whole in at least one chunk.
CHUNK_OVERLAP = 200


def extract_text(pdf_path: Path) -> str:
    """Extract all text from the PDF.

    Uses ``PyPDF2.PdfReader``, the course's PDF loader, skipping pages that
    extract to nothing.

    Args:
        pdf_path: Path to the job description PDF.
    """
    raise NotImplementedError


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries.

    Args:
        text: Full document text.
        chunk_size: Target chunk length in characters.
        overlap: Characters shared between consecutive chunks.

    Returns:
        The chunks, in document order.
    """
    raise NotImplementedError


def embed(chunks: list[str]) -> list[list[float]]:
    """Compute embeddings for chunks with ``settings.embedding_model``.

    The query side must use this same model — mixing ``-small`` and ``-large``
    produces silently meaningless distances (CLAUDE.md pitfall 6).
    """
    raise NotImplementedError


def build_index(pdf_path: Path | None = None, reset: bool = True) -> int:
    """Build the persisted Chroma collection end to end.

    Extracts, chunks, embeds and writes into ``settings.chroma_path`` under
    ``settings.chroma_collection``.

    Args:
        pdf_path: PDF to index. Defaults to ``settings.job_description_pdf``.
        reset: Drop an existing collection first. Without this, re-running
            appends duplicates.

    Returns:
        The number of chunks indexed.
    """
    raise NotImplementedError
