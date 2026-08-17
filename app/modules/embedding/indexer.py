"""Offline: build the Chroma index from the job description PDF.

Follows the course pattern from ``Course23/Embedding & Retrieval.ipynb``: raw
``chromadb`` with explicitly computed OpenAI embeddings, and ``PyPDF2.PdfReader``
for text extraction. No LangChain vectorstore wrapper, no ``PyPDFLoader``, no
``RecursiveCharacterTextSplitter`` — none of those were taught (CLAUDE.md 4.6).

The offline step writes **two** artifacts:

1. ``data/chroma/`` — a real persisted Chroma database, via
   ``chromadb.PersistentClient``. This is what the spec asks for and what the app
   queries locally.
2. ``data/vector_store.json`` — the same chunks and vectors as plain JSON, ~20 KB.

The second exists only because Chroma pre-allocates its HNSW index for 10,000
vectors and so writes **60 MB** for our three chunks — too heavy to commit. So
``data/chroma/`` is gitignored and the JSON seed is committed; on a fresh checkout
(Streamlit Community Cloud, which cannot run this script) the retriever rebuilds
the Chroma database from the seed on first use.

Chroma is the vector database in every path. The seed only changes how it gets
populated — never whether it is used.

Run via ``python scripts/build_vector_store.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb
from PyPDF2 import PdfReader

from app.config import settings
from app.modules.common import CHROMA_SETTINGS, get_openai_client

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
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() for page in reader.pages]
    text = "\n\n".join(p.strip() for p in pages if p and p.strip())
    # PDF extraction leaves ragged whitespace that inflates chunk sizes without
    # adding meaning; collapse it but keep paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        # A single paragraph longer than the budget gets split on its own.
        while len(paragraph) > chunk_size:
            cut = paragraph.rfind(" ", 0, chunk_size)
            cut = cut if cut > chunk_size // 2 else chunk_size
            chunks.append(paragraph[:cut].strip())
            paragraph = paragraph[max(0, cut - overlap) :].strip()
        current = paragraph

    if current:
        chunks.append(current)
    return [c for c in chunks if c]


def embed(chunks: list[str]) -> list[list[float]]:
    """Compute embeddings for chunks with ``settings.embedding_model``.

    The query side must use this same model — mixing ``-small`` and ``-large``
    produces silently meaningless distances (CLAUDE.md pitfall 6).
    """
    response = get_openai_client().embeddings.create(
        input=chunks, model=settings.embedding_model
    )
    return [list(d.embedding) for d in response.data]


def write_chroma(ids: list[str], documents: list[str], embeddings: list[list[float]]) -> None:
    """Populate the persisted Chroma database at ``settings.chroma_path``.

    Drops any existing collection first, because ``collection.add`` appends and
    re-running the build would otherwise duplicate every chunk. The reset goes
    through Chroma's own ``delete_collection`` rather than deleting the directory:
    on Windows the store keeps file handles open, so ``rmtree`` raises
    ``PermissionError`` and leaves a half-removed directory behind.
    """
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(settings.chroma_path), settings=CHROMA_SETTINGS
    )

    try:
        client.delete_collection(settings.chroma_collection)
    except Exception:  # noqa: BLE001 - nothing to delete on a first run
        pass

    collection = client.create_collection(name=settings.chroma_collection)
    collection.add(ids=ids, documents=documents, embeddings=embeddings)


def write_seed(
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    source: str,
) -> None:
    """Write the committed JSON seed.

    Records the embedding model alongside the vectors, so the retriever can refuse
    to query an index built with a different model rather than returning
    meaningless distances.
    """
    settings.vector_store_json.parent.mkdir(parents=True, exist_ok=True)
    settings.vector_store_json.write_text(
        json.dumps(
            {
                "model": settings.embedding_model,
                "collection": settings.chroma_collection,
                "source": source,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
            }
        ),
        encoding="utf-8",
    )


def build_index(pdf_path: Path | None = None) -> int:
    """Run the offline embedding step end to end.

    Extracts, chunks and embeds the PDF, then writes both the persisted Chroma
    database and the committed JSON seed.

    Args:
        pdf_path: PDF to index. Defaults to ``settings.job_description_pdf``.

    Returns:
        The number of chunks indexed.
    """
    pdf_path = Path(pdf_path or settings.job_description_pdf)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Job description PDF not found: {pdf_path}")

    chunks = chunk_text(extract_text(pdf_path))
    if not chunks:
        raise RuntimeError(f"No text could be extracted from {pdf_path}.")

    embeddings = embed(chunks)
    ids = [f"chunk-{i}" for i in range(len(chunks))]

    write_chroma(ids, chunks, embeddings)
    write_seed(ids, chunks, embeddings, source=pdf_path.name)
    return len(chunks)
