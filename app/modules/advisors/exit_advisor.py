"""Conversation Exit Advisor.

Decides whether ending the conversation now makes sense, so uninterested
candidates are not chased and a booked candidate is not asked more questions.

This is the advisor the spec requires to be **fine-tuned**. The model id comes
from ``settings.exit_advisor_model``, which returns the fine-tuned model when
``FT_EXIT_ADVISOR_MODEL`` is set and falls back to ``OPENAI_MODEL`` with few-shot
prompting otherwise. The fallback must always keep working: the app never
hard-fails because a fine-tuning job is missing or still running
(ENGINEERING.md 11.7).

The prompt follows the model: few-shot examples for the base model, a condensed
file for the fine-tuned one. See :data:`PROMPT_FEWSHOT` and
:data:`PROMPT_FINETUNED`.

Critical semantics: ``should_end`` is True for BOTH terminal outcomes — the
interview is confirmed, and the candidate opted out. In the dataset, ``end`` is
the last recruiter turn of every conversation: 11 happy endings and 4 opt-outs.
Treating ``end`` as a rejection signal gets most of the class wrong
(ENGINEERING.md 6.1).
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.modules.advisors.schemas import ExitVerdict
from app.modules.common import fill, get_llm, load_prompt

logger = logging.getLogger(__name__)

#: Short directive in the user turn. The conversation itself lives in the system
#: prompt's `# Context` section, following the Course20 layout.
DIRECTIVE = "Return your JSON verdict for the conversation above."

#: Two prompt files, one per model.
#:
#: The few-shot file carries seven worked examples that teach the base model the
#: `end`-is-terminal semantics. Replacing those examples with learned behaviour is
#: the whole point of fine-tuning, so the fine-tuned model gets a condensed file:
#: same identity, same output contract, no examples.
#:
#: Which one is used follows `settings.is_finetuned`, and the fine-tuning dataset
#: builder calls :func:`build_system_text` with ``finetuned=True`` — so the rows
#: the model trains on and the messages it later receives are identical by
#: construction, not by two places being kept in sync.
PROMPT_FEWSHOT = "exit_advisor"
PROMPT_FINETUNED = "exit_advisor_finetuned"


def build_system_text(conversation_text: str, finetuned: bool | None = None) -> str:
    """Render the Exit Advisor system prompt for one conversation.

    Args:
        conversation_text: History rendered as plain text, one
            ``Speaker: message`` line per turn.
        finetuned: Force a prompt variant. Defaults to ``settings.is_finetuned``,
            which is what every runtime call wants; the fine-tuning dataset
            builder passes True explicitly.
    """
    if finetuned is None:
        finetuned = settings.is_finetuned
    name = PROMPT_FINETUNED if finetuned else PROMPT_FEWSHOT
    return fill(load_prompt(name), conversation=conversation_text)


def build_exit_chain():
    """Build the Exit Advisor chain.

    Shape is the taught LCEL pipe, ``prompt | llm | JsonOutputParser()``
    (ENGINEERING.md 4.4), with the system prompt loaded from
    ``app/prompts/exit_advisor.txt`` and ``temperature=0``.

    The system text is passed in as a *value*, not as a template, so the literal
    JSON braces in the few-shot examples are never read as template fields.

    Returns:
        A runnable producing the raw :class:`ExitVerdict` dict.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", "{system_text}"), ("user", "{input}")]
    )
    return prompt | get_llm(settings.exit_advisor_model) | JsonOutputParser()


def should_end(conversation_text: str) -> ExitVerdict:
    """Judge whether the conversation should close now.

    Args:
        conversation_text: The full history rendered as plain text, one
            ``Speaker: message`` line per turn.

    Returns:
        The validated verdict. On any model or parsing failure this returns a
        conservative "keep talking" verdict rather than raising — one flaky
        advisor must not end a candidate's conversation.
    """
    system_text = build_system_text(conversation_text)
    try:
        raw = build_exit_chain().invoke(
            {"system_text": system_text, "input": DIRECTIVE}
        )
        return ExitVerdict.model_validate(raw)
    except Exception:  # noqa: BLE001 - degrade instead of killing the turn
        logger.exception("Exit Advisor failed; defaulting to should_end=False")
        return ExitVerdict(should_end=False, reason="Exit Advisor unavailable.")
