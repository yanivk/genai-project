"""Conversation Info Advisor.

Answers the candidate's questions about the position, grounded in the Chroma
vector store built offline from ``data/Python Developer Job Description.pdf``,
and helps formulate the next message.

Per the spec this advisor **also drives the conversation toward the end goal:
scheduling an interview**. A grounded answer that leaves the candidate with
nothing to do next is a half-done answer.

RAG is hand-wired the way the course taught it (``Course23/Embedding & Retrieval.ipynb``):
retrieve with ``collection.query`` -> stuff the context into the prompt -> call
the LLM. No ``RetrievalQA``, no ``create_retrieval_chain`` (CLAUDE.md 4.6).
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.modules.advisors.schemas import InfoVerdict
from app.modules.common import fill, get_llm, load_prompt
from app.modules.embedding.retriever import format_context, retrieve

logger = logging.getLogger(__name__)


def build_info_chain():
    """Build the Info Advisor chain.

    Taught LCEL pipe ``prompt | llm | JsonOutputParser()``, system prompt from
    ``app/prompts/info_advisor.txt``, ``temperature=0``.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", "{system_text}"), ("user", "{input}")]
    )
    return prompt | get_llm() | JsonOutputParser()


def _retrieve_context(conversation_text: str, n_results: int) -> str:
    """Retrieve job-description context for the latest candidate message.

    Uses the last candidate line as the query rather than the whole transcript:
    embedding the full history dilutes the actual question and retrieves noise.
    Returns an empty ``<sources>`` block when the index is missing, so a missing
    offline step degrades the answer instead of breaking the turn.
    """
    candidate_lines = [
        line.split(":", 1)[1].strip()
        for line in conversation_text.splitlines()
        if line.startswith("Candidate:")
    ]
    query = (candidate_lines[-1] if candidate_lines else conversation_text).strip()
    if not query:
        # The opening turn of a conversation: the candidate has not spoken yet, so
        # there is nothing to retrieve against.
        return "<sources></sources>"
    try:
        return format_context(retrieve(query, n_results=n_results))
    except Exception:  # noqa: BLE001 - index missing or unreachable
        logger.exception("Retrieval failed; answering without grounding")
        return "<sources></sources>"


def advise(conversation_text: str, n_results: int = 3) -> InfoVerdict:
    """Answer the candidate's question from the job description, if there is one.

    Args:
        conversation_text: The full history rendered as plain text.
        n_results: How many chunks to retrieve from Chroma.

    Returns:
        The validated verdict. ``info_needed`` is False when the candidate did
        not ask anything answerable, in which case ``answer`` is empty.
        Degrades to "no info needed" on failure rather than raising.
    """
    system_text = fill(
        load_prompt("info_advisor"),
        conversation=conversation_text,
        sources=_retrieve_context(conversation_text, n_results),
    )
    try:
        raw = build_info_chain().invoke(
            {
                "system_text": system_text,
                "input": "Return your JSON verdict for the conversation above.",
            }
        )
        return InfoVerdict.model_validate(raw)
    except Exception:  # noqa: BLE001 - degrade instead of killing the turn
        logger.exception("Info Advisor failed; defaulting to info_needed=False")
        return InfoVerdict(info_needed=False, answer="", sources=[])
