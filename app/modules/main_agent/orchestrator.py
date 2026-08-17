"""Turn orchestration — the Main Agent.

Implements the flow from the spec's workflow diagram (``Page 6.png``): receive
the candidate message, consult the advisors, then either consult again (bounded
by ``settings.max_advisor_rounds``) or emit the reply.

Follows the course agent pattern from ``Course22/LangChain - Agents & Tools.ipynb``:
``create_openai_tools_agent`` + ``AgentExecutor``, memory via
``RunnableWithMessageHistory`` over a module-level ``store`` dict. Multi-agent
routing is plain Python, as taught in that notebook's Example 5 — no framework.

See CLAUDE.md sections 4.1, 4.2 and 4.5.

STATUS: scaffolding. Signatures are final; bodies are not implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_community.chat_message_histories import ChatMessageHistory

from app.modules.main_agent.actions import Action

#: session_id -> conversation history. The course's memory pattern
#: (Course22/LangChain - Chains & Memory.ipynb). Process-local by design: the
#: Streamlit layer keys it by its own session id.
store: dict[str, ChatMessageHistory] = {}


def get_history(session_id: str) -> ChatMessageHistory:
    """Return the history for a session, creating it on first use.

    This is the ``get_session_history`` callback passed to
    ``RunnableWithMessageHistory``.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def reset_session(session_id: str) -> None:
    """Clear one session's history, leaving the session registered."""
    if session_id in store:
        store[session_id].clear()


@dataclass(frozen=True)
class TurnResult:
    """What the Main Agent decided for one turn.

    Attributes:
        action: One of ``continue`` / ``schedule`` / ``end``.
        message: The text to send to the candidate.
        reason: Short rationale, for logging and for the evaluation error table.
        advisor_rounds: How many advisor consultation rounds this turn used.
    """

    action: Action
    message: str
    reason: str
    advisor_rounds: int = 1


def build_main_agent():
    """Build the Main Agent executor.

    Prompt layout is mandatory and ordered ``system`` -> ``history`` ->
    ``agent_scratchpad`` -> ``user`` (CLAUDE.md 4.1). The system text is loaded
    from ``app/prompts/main_agent.txt``.

    Returns:
        A ``RunnableWithMessageHistory`` wrapping an ``AgentExecutor``.
    """
    raise NotImplementedError


def handle_turn(
    session_id: str,
    candidate_message: str,
    conversation_start: str | None = None,
) -> TurnResult:
    """Process one candidate message and decide the bot's response.

    Consults the Exit, Scheduling and Info advisors, then resolves their verdicts
    into a single action and a candidate-facing message. May consult again, up to
    ``settings.max_advisor_rounds`` times, before replying.

    Args:
        session_id: Conversation key into :data:`store`.
        candidate_message: The candidate's latest message.
        conversation_start: ISO timestamp the conversation started. Relative
            dates ("next Friday") resolve against this, NOT against today —
            see CLAUDE.md section 6.2. Defaults to now when omitted.

    Returns:
        The decided :class:`TurnResult`.
    """
    raise NotImplementedError


def predict_action(
    history: list[dict[str, str]],
    conversation_start: str | None = None,
) -> Action:
    """Predict the action for the next recruiter turn, given prior history.

    This is the evaluation entry point (CLAUDE.md section 10): it returns only the
    label, without generating a candidate-facing message, so
    ``tests/test_evals.ipynb`` can score the system against
    ``data/sms_conversations.json``.

    Args:
        history: Turns so far, each ``{"speaker": ..., "text": ...}``, ending on
            a candidate turn.
        conversation_start: The conversation's ``start_time_utc``.

    Returns:
        One of ``continue`` / ``schedule`` / ``end``.
    """
    raise NotImplementedError
