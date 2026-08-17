"""Streamlit app — the proof of concept that stands in for SMS.

Run with::

    streamlit run streamlit_app/streamlit_main.py

Chat layout follows ``Course24/streamlit_1_app.py``: ``st.session_state.messages``
holds the transcript, ``st.chat_message`` renders it, ``st.chat_input`` collects
the next candidate message, and ``st.spinner`` covers the model call.

UI only — every decision is made in ``app/`` (CLAUDE.md section 2, rule 4).
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` puts this file's directory on sys.path, not the repository
# root, so `import app` would fail. Fix it before importing anything local.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from streamlit_app.utils import (  # noqa: E402
    ACTION_BADGES,
    DEFAULT_CONVERSATION_START,
    append_message,
    health_check,
    init_session_state,
    render_history,
    reset_conversation,
)

st.set_page_config(page_title="Python Developer — Recruiting Bot", page_icon="💬")

init_session_state()

# --- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("Settings ⚙️")

    st.session_state.conversation_start = st.text_input(
        "Conversation date",
        value=st.session_state.get("conversation_start", DEFAULT_CONVERSATION_START),
        help=(
            "Relative dates like 'next Friday' resolve against this. Defaults to "
            "now; change it to replay a conversation from the dataset."
        ),
    )

    if st.button("Reset conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.divider()
    st.subheader("Setup status")
    for name, (ok, detail) in health_check().items():
        st.write(f"{'✅' if ok else '⚠️'} **{name}** — {detail}")

    if st.session_state.get("last_verdicts"):
        st.divider()
        with st.expander("Last advisor verdicts"):
            st.json(st.session_state.last_verdicts)


# --- Chat ----------------------------------------------------------------
st.title("Python Developer — Recruiting Bot")
st.caption("SMS-style screening chat. Ask about the role, or arrange an interview.")

render_history()

if st.session_state.get("finished"):
    st.info("This conversation is closed. Use **Reset conversation** to start again.")

user_prompt = st.chat_input(
    "Type your message...", disabled=bool(st.session_state.get("finished"))
)

if user_prompt:
    append_message("user", user_prompt)
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                from app.modules.main_agent.orchestrator import handle_turn

                result = handle_turn(
                    session_id=st.session_state.session_id,
                    candidate_message=user_prompt,
                    conversation_start=st.session_state.conversation_start,
                )
                st.write(result.message)
                icon, label = ACTION_BADGES.get(result.action, ("", result.action))
                st.caption(f"{icon} {label} — {result.reason}")

                append_message("assistant", result.message, action=result.action)
                st.session_state.last_verdicts = result.verdicts
                if result.action == "end":
                    st.session_state.finished = True
                    st.rerun()
            except Exception as exc:  # noqa: BLE001 - surface it in the UI
                st.error(f"The agent failed on this turn: {exc}")
                st.caption(
                    "Check the sidebar status panel — a missing database or "
                    "vector index is the usual cause."
                )
