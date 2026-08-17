"""Rendering and session helpers for the Streamlit app.

Presentation only — nothing here decides anything about the conversation.

STATUS: scaffolding. Signatures are final; bodies are not implemented yet.
"""

from __future__ import annotations

#: Default conversation date offered in the sidebar. Anchored inside 2024 because
#: that is the only year the seeded database covers; today's date would find no
#: slots at all (CLAUDE.md 6.2).
DEFAULT_CONVERSATION_START = "2024-04-03T15:12:00Z"

#: Opening message, matching the phrasing used throughout the labeled dataset.
GREETING = (
    "Hi, thanks for submitting your application for our Python Developer role. "
    "Could you share a bit about your Python experience?"
)


def init_session_state() -> None:
    """Seed ``st.session_state`` on first render.

    Sets up the message list, the session id used to key the agent's memory, and
    the conversation-start anchor.
    """
    raise NotImplementedError


def render_history() -> None:
    """Replay ``st.session_state.messages`` into the chat area.

    Course24's pattern: iterate the stored messages and write each one inside
    ``st.chat_message(role)``.
    """
    raise NotImplementedError


def append_message(role: str, content: str) -> None:
    """Append a message to the session transcript.

    Args:
        role: ``"user"`` for the candidate, ``"assistant"`` for the bot.
        content: The message text.
    """
    raise NotImplementedError


def reset_conversation() -> None:
    """Clear both the UI transcript and the agent's memory for this session."""
    raise NotImplementedError


def render_sidebar() -> dict:
    """Draw the sidebar and return the chosen settings.

    Exposes the conversation date anchor, a reset button, and a read-only status
    panel showing whether the fine-tuned Exit Advisor, the database and the
    vector index are available.

    Returns:
        The selected settings, e.g. ``{"conversation_start": "..."}``.
    """
    raise NotImplementedError


def health_check() -> dict[str, tuple[bool, str]]:
    """Report whether each offline artifact is present.

    Checks the SQLite database, the Chroma index, the API key and the fine-tuned
    model id, so a missing setup step shows up in the UI rather than as a stack
    trace mid-conversation.

    Returns:
        ``{name: (ok, human readable detail)}``.
    """
    raise NotImplementedError
