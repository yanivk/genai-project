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

STATUS: scaffolding. Signatures are final; bodies are not implemented yet.
"""

from __future__ import annotations

from app.modules.advisors.schemas import InfoVerdict


def build_info_chain():
    """Build the Info Advisor chain.

    Taught LCEL pipe ``prompt | llm | JsonOutputParser()``, system prompt from
    ``app/prompts/info_advisor.txt``, ``temperature=0``.
    """
    raise NotImplementedError


def advise(conversation_text: str, n_results: int = 3) -> InfoVerdict:
    """Answer the candidate's question from the job description, if there is one.

    Args:
        conversation_text: The full history rendered as plain text.
        n_results: How many chunks to retrieve from Chroma.

    Returns:
        The validated verdict. ``info_needed`` is False when the candidate did
        not ask anything answerable, in which case ``answer`` is empty.
    """
    raise NotImplementedError
