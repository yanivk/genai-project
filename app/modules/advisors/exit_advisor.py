"""Conversation Exit Advisor.

Decides whether ending the conversation now makes sense, so uninterested
candidates are not chased and a booked candidate is not asked more questions.

This is the advisor the spec requires to be **fine-tuned**. The model id comes
from ``settings.exit_advisor_model``, which returns the fine-tuned model when
``FT_EXIT_ADVISOR_MODEL`` is set and falls back to ``OPENAI_MODEL`` with few-shot
prompting otherwise. The fallback must always keep working: the app never
hard-fails because a fine-tuning job is missing or still running
(CLAUDE.md 11.7).

Critical semantics: ``should_end`` is True for BOTH terminal outcomes — the
interview is confirmed, and the candidate opted out. In the dataset, ``end`` is
the last recruiter turn of every conversation: 11 happy endings and 4 opt-outs.
Treating ``end`` as a rejection signal gets most of the class wrong
(CLAUDE.md 6.1).

STATUS: scaffolding. Signatures are final; bodies are not implemented yet.
"""

from __future__ import annotations

from app.modules.advisors.schemas import ExitVerdict


def build_exit_chain():
    """Build the Exit Advisor chain.

    Shape is the taught LCEL pipe, ``prompt | llm | JsonOutputParser()``
    (CLAUDE.md 4.4), with the system prompt loaded from
    ``app/prompts/exit_advisor.txt`` and ``temperature=0``.

    Returns:
        A runnable producing the raw :class:`ExitVerdict` dict.
    """
    raise NotImplementedError


def should_end(conversation_text: str) -> ExitVerdict:
    """Judge whether the conversation should close now.

    Args:
        conversation_text: The full history rendered as plain text, one
            ``Speaker: message`` line per turn.

    Returns:
        The validated verdict.
    """
    raise NotImplementedError
