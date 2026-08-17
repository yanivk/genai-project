"""Streamlit app — the proof of concept that stands in for SMS.

Run with::

    streamlit run streamlit_app/streamlit_main.py

Chat layout follows ``Course24/streamlit_1_app.py``: ``st.session_state.messages``
holds the transcript, ``st.chat_message`` renders it, ``st.chat_input`` collects
the next candidate message, and ``st.spinner`` covers the model call.

STATUS: scaffolding. The page renders and reports setup status; the chat loop is
wired but the turn handler is not implemented yet.
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

from app.config import settings  # noqa: E402
from streamlit_app.utils import DEFAULT_CONVERSATION_START, GREETING  # noqa: E402

st.set_page_config(page_title="Python Developer — Recruiting Bot", page_icon="💬")


def _status_rows() -> list[tuple[str, bool, str]]:
    """Check that each offline artifact exists, without calling the API."""
    db_file = settings.db_url.removeprefix("sqlite:///")
    return [
        (
            "OpenAI API key",
            bool(settings.openai_api_key),
            "set" if settings.openai_api_key else "missing — copy .env.example to .env",
        ),
        (
            "Schedule database",
            Path(db_file).exists(),
            db_file if Path(db_file).exists() else "run: python scripts/seed_database.py",
        ),
        (
            "Chroma index",
            settings.chroma_path.exists() and any(settings.chroma_path.iterdir()),
            str(settings.chroma_path)
            if settings.chroma_path.exists() and any(settings.chroma_path.iterdir())
            else "run: python scripts/build_vector_store.py",
        ),
        (
            "Exit Advisor model",
            settings.is_finetuned,
            settings.ft_exit_advisor_model
            if settings.is_finetuned
            else f"not fine-tuned yet — falling back to {settings.openai_model}",
        ),
    ]


# --- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("Settings ⚙️")

    conversation_start = st.text_input(
        "Conversation date",
        value=DEFAULT_CONVERSATION_START,
        help=(
            "Relative dates like 'next Friday' resolve against this, not today. "
            "The interview schedule only covers 2024."
        ),
    )

    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Setup status")
    for name, ok, detail in _status_rows():
        st.write(f"{'✅' if ok else '⚠️'} **{name}** — {detail}")


# --- Chat ----------------------------------------------------------------
st.title("Python Developer — Recruiting Bot")
st.caption("SMS-style screening chat. Ask about the role, or arrange an interview.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]

if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                from app.modules.main_agent.orchestrator import handle_turn

                result = handle_turn(
                    session_id=st.session_state.session_id,
                    candidate_message=user_prompt,
                    conversation_start=conversation_start,
                )
                reply = result.message
            except NotImplementedError:
                reply = (
                    "⚠️ The agent is not implemented yet — this is the project "
                    "scaffolding. See CLAUDE.md for the architecture."
                )
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
