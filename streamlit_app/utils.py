"""Rendering and session helpers for the Streamlit app.

Presentation only — nothing here decides anything about the conversation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from app.config import settings


def default_conversation_start() -> str:
    """Return now as the conversation anchor, in the dataset's timestamp format.

    Relative dates resolve against this. Defaulting to now is what you want for a
    live chat; override it in the sidebar to replay a dataset conversation. The
    seeded schedule runs to the end of 2027, so today always lands inside it
    (CLAUDE.md 6.3).
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


#: Anchor offered in the sidebar on first render.
DEFAULT_CONVERSATION_START = default_conversation_start()

#: Opening message, matching the phrasing used throughout the labeled dataset.
GREETING = (
    "Hi, thanks for submitting your application for our Python Developer role. "
    "Could you share a bit about your Python experience?"
)

#: Streamlit session key -> agent memory key. One browser session, one thread.
SESSION_ID = "streamlit"

#: Badge shown next to each action in the transcript.
ACTION_BADGES = {
    "continue": ("💬", "Continuing"),
    "schedule": ("📅", "Proposing slots"),
    "end": ("🏁", "Conversation closed"),
}


def _seed_agent_memory() -> None:
    """Put the greeting into the agent's memory so turn 1 has context."""
    from app.modules.main_agent.orchestrator import get_history, reset_session

    reset_session(SESSION_ID)
    get_history(SESSION_ID).add_ai_message(GREETING)


def init_session_state() -> None:
    """Seed ``st.session_state`` on first render.

    Sets up the message list, the session id used to key the agent's memory, and
    the conversation-start anchor.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": GREETING}]
        _seed_agent_memory()

    st.session_state.setdefault("session_id", SESSION_ID)
    st.session_state.setdefault("conversation_start", DEFAULT_CONVERSATION_START)
    st.session_state.setdefault("finished", False)
    st.session_state.setdefault("last_verdicts", None)


def render_history() -> None:
    """Replay ``st.session_state.messages`` into the chat area."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            action = message.get("action")
            if action in ACTION_BADGES:
                icon, label = ACTION_BADGES[action]
                st.caption(f"{icon} {label}")


def append_message(role: str, content: str, action: str | None = None) -> None:
    """Append a message to the session transcript.

    Args:
        role: ``"user"`` for the candidate, ``"assistant"`` for the bot.
        content: The message text.
        action: The action the Main Agent chose, shown as a badge.
    """
    entry: dict[str, str] = {"role": role, "content": content}
    if action:
        entry["action"] = action
    st.session_state.messages.append(entry)


def reset_conversation() -> None:
    """Clear both the UI transcript and the agent's memory for this session."""
    _seed_agent_memory()
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]
    st.session_state.finished = False
    st.session_state.last_verdicts = None


def health_check() -> dict[str, tuple[bool, str]]:
    """Report whether each offline artifact is present.

    Checks the SQLite database, the Chroma index and the API key, so a missing
    setup step shows up in the UI rather than as a stack trace mid-conversation.
    The Exit Advisor row reports which model backs it, but never fails: running
    on the few-shot fallback is a valid state, not a missing step.

    Returns:
        ``{name: (ok, human readable detail)}``.
    """
    db_file = Path(settings.db_url.removeprefix("sqlite:///"))
    # Either the persisted Chroma DB or the committed seed it is rebuilt from.
    persisted = settings.chroma_path.exists() and any(settings.chroma_path.iterdir())
    seeded = settings.vector_store_json.is_file()
    return {
        "OpenAI API key": (
            bool(settings.openai_api_key),
            "set" if settings.openai_api_key else "missing — fill in .env",
        ),
        "Schedule database": (
            db_file.exists(),
            db_file.name if db_file.exists() else "run: python scripts/seed_database.py",
        ),
        "Chroma index": (
            persisted or seeded,
            f"{settings.chroma_collection} (persisted)"
            if persisted
            else f"{settings.chroma_collection} (rebuilt from seed on first use)"
            if seeded
            else "run: python scripts/build_vector_store.py",
        ),
        # Not a setup step, and not a warning: OpenAI closed self-serve
        # fine-tuning in May 2026, so few-shot is the only reachable state and
        # the one the reported evaluation measures. Flagging it orange would
        # tell the reader something is missing when nothing is.
        "Exit Advisor": (
            True,
            "fine-tuned"
            if settings.is_finetuned
            else f"few-shot on {settings.openai_model}",
        ),
    }
