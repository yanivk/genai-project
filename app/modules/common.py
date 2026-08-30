"""Shared helpers for prompts, models and history rendering.

Small internal utilities used by every agent module, so the course patterns from
ENGINEERING.md section 4 are written once rather than copy-pasted four times.

Prompt-file placeholders are substituted with plain ``str.replace``, never
``str.format``: the prompt files contain literal JSON braces in their few-shot
examples, and ``format`` would try to read those as template fields.

The rendered system text is then passed into the chain as the *value* of a single
``{system_text}`` variable. LangChain substitutes variables without re-parsing
what it substituted, so the JSON braces inside survive untouched.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from chromadb.config import Settings as ChromaSettings
from langchain_openai import ChatOpenAI
from openai import OpenAI

from app.config import settings

#: Chroma phones home on startup and prints a stack trace when the call fails,
#: which buries real output. The env var alone is not enough in chromadb 1.0.x —
#: pass this to every client constructor.
CHROMA_SETTINGS = ChromaSettings(anonymized_telemetry=False)


@lru_cache(maxsize=8)
def load_prompt(name: str) -> str:
    """Load a prompt file from ``app/prompts``.

    Args:
        name: File name, with or without the ``.txt`` suffix.

    Returns:
        The raw prompt text.
    """
    return settings.prompt_path(name).read_text(encoding="utf-8")


def fill(template: str, **values: str) -> str:
    """Substitute ``{placeholder}`` tokens without touching literal JSON braces.

    Args:
        template: Prompt text containing ``{placeholder}`` tokens.
        **values: Placeholder name -> replacement text.
    """
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def get_llm(model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    """Build a chat model.

    ``temperature=0`` by default: every advisor and the router are making
    decisions, and those must be reproducible (ENGINEERING.md section 8).

    Args:
        model: Model id. Defaults to ``settings.openai_model``.
        temperature: Sampling temperature.
    """
    settings.require_api_key()
    return ChatOpenAI(
        model=model or settings.openai_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
    )


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Return the raw OpenAI client, used for embeddings and fine-tuning."""
    return OpenAI(api_key=settings.require_api_key())


#: How each speaker is labelled when history is rendered for a model. The same
#: labels are used at fine-tuning time, so training and inference see identical
#: input (ENGINEERING.md section 11).
SPEAKER_LABELS = {"recruiter": "Recruiter", "candidate": "Candidate"}


def render_history(history: list[dict[str, str]]) -> str:
    """Render turns as the plain text every advisor receives.

    One ``Speaker: message`` line per turn.

    Args:
        history: Turns, each with ``speaker`` and ``text`` keys.
    """
    return "\n".join(
        f"{SPEAKER_LABELS.get(turn['speaker'], turn['speaker'].capitalize())}: {turn['text']}"
        for turn in history
    )


def parse_json_output(text: str) -> dict:
    """Parse a JSON object out of a model response.

    Models occasionally wrap JSON in a code fence or add a sentence around it
    despite being told not to. This tolerates both rather than failing the turn.

    Args:
        text: The raw model output.

    Raises:
        ValueError: If no JSON object can be recovered.
    """
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} span.
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON from model output: {text!r}") from exc
    raise ValueError(f"Could not parse JSON from model output: {text!r}")
