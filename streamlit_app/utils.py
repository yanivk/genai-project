"""Rendering and session helpers for the Streamlit app.

Presentation only — nothing here decides anything about the conversation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import streamlit as st


def default_conversation_start() -> str:
    """Return now as the conversation anchor, in the dataset's timestamp format.

    Relative dates resolve against this. Defaulting to now is what you want for a
    live chat; override it in the sidebar to replay a dataset conversation. The
    seeded schedule runs to the end of 2027, so today always lands inside it
    (ENGINEERING.md 6.3).
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


#: Anchor offered in the sidebar on first render.
DEFAULT_CONVERSATION_START = default_conversation_start()

#: Opening message, matching the phrasing used throughout the labeled dataset.
GREETING = (
    "Hi, thanks for submitting your application for our Python Developer role. "
    "Could you share a bit about your Python experience?"
)

#: Prefix for the agent-memory key. The id itself must be UNIQUE PER BROWSER
#: SESSION: ``orchestrator.store`` is process-global, so a constant here makes
#: every tab and every viewer of a deployed app share one conversation — and
#: ``_seed_agent_memory`` resets it, so a second tab wipes the first one's memory
#: while its transcript still shows the whole exchange.
SESSION_PREFIX = "streamlit"


def new_session_id() -> str:
    """Return a fresh agent-memory key for one browser session."""
    return f"{SESSION_PREFIX}-{uuid.uuid4().hex}"


#: Sidebar width, in pixels. Streamlit's default is too narrow for the
#: availability calendar: seven weekday columns wrap and the grid stops being
#: scannable. The sidebar stays user-resizable — this only sets where it opens.
SIDEBAR_WIDTH = 350


def apply_sidebar_width(width: int = SIDEBAR_WIDTH) -> None:
    """Widen the sidebar to fit the seven-column calendar grid.

    Streamlit has no page-config option for this, so it goes in as CSS. The rule
    is marked ``!important`` deliberately: the resize handle writes an inline
    ``width`` on the same element, and an author ``!important`` declaration is
    the one thing that outranks a plain inline style.

    Args:
        width: Sidebar width in pixels.
    """
    st.markdown(
        f"""
        <style>
          [data-testid="stSidebar"] {{
              width: {width}px !important;
              min-width: {width}px !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _seed_agent_memory(session_id: str) -> None:
    """Put the greeting into the agent's memory so turn 1 has context.

    Args:
        session_id: This browser session's key into the agent's memory. Passed in
            rather than read from a constant — resetting a shared key is exactly
            how one tab used to erase another's conversation.
    """
    from app.modules.main_agent.orchestrator import get_history, reset_session

    reset_session(session_id)
    get_history(session_id).add_ai_message(GREETING)


def init_session_state() -> None:
    """Seed ``st.session_state`` on first render.

    Sets up the message list, the session id used to key the agent's memory, and
    the conversation-start anchor.

    The session id is minted first and the memory seeded against it, in that
    order: seeding is a reset, so it must never run against an id another
    session could be using.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = new_session_id()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": GREETING}]
        _seed_agent_memory(st.session_state.session_id)

    st.session_state.setdefault("conversation_start", DEFAULT_CONVERSATION_START)
    st.session_state.setdefault("finished", False)
    st.session_state.setdefault("last_verdicts", None)


def render_history() -> None:
    """Replay ``st.session_state.messages`` into the chat area.

    The transcript shows the conversation and nothing else — the action behind
    each bot message is stored but never rendered, so the chat reads the way an
    SMS thread would.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def append_message(role: str, content: str, action: str | None = None) -> None:
    """Append a message to the session transcript.

    Args:
        role: ``"user"`` for the candidate, ``"assistant"`` for the bot.
        content: The message text.
        action: The action the Main Agent chose. Stored as part of the
            transcript entry for the terminal-state check (``end`` closes the
            conversation) — deliberately not displayed to the candidate.
    """
    entry: dict[str, str] = {"role": role, "content": content}
    if action:
        entry["action"] = action
    st.session_state.messages.append(entry)


def reset_conversation() -> None:
    """Clear both the UI transcript and the agent's memory for this session."""
    _seed_agent_memory(st.session_state.session_id)
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]
    st.session_state.finished = False
    st.session_state.last_verdicts = None
