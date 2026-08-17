"""Turn orchestration — the Main Agent.

Implements the flow from the spec's workflow diagram (``Page 6.png``): receive
the candidate message, consult the three advisors, then emit the reply.

Multi-agent routing is plain Python, as taught in
``Course22/LangChain - Agents & Tools.ipynb`` Example 5 — no framework.

Memory uses the course's objects — a module-level ``store`` dict of
``ChatMessageHistory``, replayed through ``MessagesPlaceholder("history")``
(CLAUDE.md 4.2). It does **not** use ``RunnableWithMessageHistory``: that wrapper
appends the chain's raw output to the history, and this chain outputs a JSON
decision object. The candidate would then "remember" the bot speaking JSON, which
poisons every later turn. Instead the store is appended to explicitly, with the
natural-language message only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import settings
from app.modules.advisors import exit_advisor, info_advisor, scheduling_advisor
from app.modules.advisors.schemas import (
    ExitVerdict,
    InfoVerdict,
    MainDecision,
    SchedulingVerdict,
)
from app.modules.common import fill, get_llm, load_prompt, parse_json_output
from app.modules.main_agent.actions import CONTINUE, END, SCHEDULE, Action

logger = logging.getLogger(__name__)

#: session_id -> conversation history. The course's memory pattern
#: (Course22/LangChain - Chains & Memory.ipynb). Process-local by design: the
#: Streamlit layer keys it by its own session id.
store: dict[str, ChatMessageHistory] = {}


def get_history(session_id: str) -> ChatMessageHistory:
    """Return the history for a session, creating it on first use."""
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
        verdicts: The raw advisor verdicts, exposed so the UI can show its work.
    """

    action: Action
    message: str
    reason: str
    advisor_rounds: int = 1
    verdicts: dict = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def render_messages(messages, latest: str | None = None) -> str:
    """Render stored chat messages as the plain text the advisors receive.

    Args:
        messages: ``HumanMessage`` / ``AIMessage`` objects from the store.
        latest: A candidate message not yet in the store, appended last.
    """
    lines = [
        f"{'Candidate' if isinstance(m, HumanMessage) else 'Recruiter'}: {m.content}"
        for m in messages
    ]
    if latest:
        lines.append(f"Candidate: {latest}")
    return "\n".join(lines)


def consult_advisors(
    conversation_text: str, conversation_start: str
) -> tuple[ExitVerdict, SchedulingVerdict, InfoVerdict]:
    """Run the three advisors over the same conversation snapshot.

    Each advisor degrades to a safe default on failure rather than raising, so a
    single flaky call cannot break the turn.
    """
    exit_verdict = exit_advisor.should_end(conversation_text)
    scheduling_verdict = scheduling_advisor.advise(conversation_text, conversation_start)
    info_verdict = info_advisor.advise(conversation_text)
    return exit_verdict, scheduling_verdict, info_verdict


def _format_verdicts(
    exit_verdict: ExitVerdict,
    scheduling_verdict: SchedulingVerdict,
    info_verdict: InfoVerdict,
) -> str:
    """Render the advisor verdicts in the shape the prompt's examples use."""
    return (
        f"exit: {exit_verdict.model_dump_json()}\n"
        f"scheduling: {scheduling_verdict.model_dump_json()}\n"
        f"info: {info_verdict.model_dump_json()}"
    )


def resolve_action(
    exit_verdict: ExitVerdict, scheduling_verdict: SchedulingVerdict
) -> Action:
    """Resolve the advisor verdicts into the turn's action.

    Precedence: **exit beats scheduling, scheduling beats continue.**

    This is deliberately deterministic Python rather than an LLM judgement call.
    Asked to weigh the verdicts itself, the model reliably picked ``schedule``
    over ``end`` whenever both advisors fired — which is exactly the case where a
    candidate has just accepted a time, so the correct answer is always ``end``.
    Routing in code also matches the course's own multi-agent example, where
    dispatch is a plain Python function (CLAUDE.md 4.5).
    """
    if exit_verdict.should_end:
        return END
    if scheduling_verdict.should_schedule:
        return SCHEDULE
    return CONTINUE


def build_main_agent():
    """Build the Main Agent chain.

    Prompt layout is mandatory and ordered ``system`` -> ``history`` -> ``user``
    (CLAUDE.md 4.1). The system text is loaded from ``app/prompts/main_agent.txt``
    and passed in as a value, so the literal JSON braces in its few-shot examples
    are never read as template fields.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_text}"),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}"),
        ]
    )
    return prompt | get_llm() | StrOutputParser()


def decide(
    conversation_text: str,
    candidate_message: str,
    conversation_start: str,
    history_messages: list | None = None,
) -> tuple[MainDecision, dict]:
    """Consult the advisors and resolve them into one decision.

    The action comes from :func:`resolve_action` — deterministic. The Main Agent
    is then told that action and writes the candidate-facing message for it, so
    the wording can never contradict the decision.

    Shared by :func:`handle_turn` (stateful, for chat) and :func:`predict_action`
    (stateless, for evaluation), so both paths exercise identical logic.

    Returns:
        The decision, and the raw advisor verdicts as a dict.
    """
    exit_verdict, scheduling_verdict, info_verdict = consult_advisors(
        conversation_text, conversation_start
    )
    action = resolve_action(exit_verdict, scheduling_verdict)
    verdicts = {
        "exit": exit_verdict.model_dump(),
        "scheduling": scheduling_verdict.model_dump(),
        "info": info_verdict.model_dump(),
        "resolved_action": action,
    }

    system_text = fill(
        load_prompt("main_agent"),
        conversation_start=conversation_start[:10],
        action=action,
        advisors=_format_verdicts(exit_verdict, scheduling_verdict, info_verdict),
    )

    try:
        raw = build_main_agent().invoke(
            {
                "system_text": system_text,
                "history": history_messages or [],
                "input": candidate_message,
            }
        )
        decision = MainDecision.model_validate(parse_json_output(raw))
        # The action is ours, not the model's — it only writes the message.
        decision = decision.model_copy(update={"action": action})
    except Exception:  # noqa: BLE001 - the decision still stands without a message
        logger.exception("Main Agent failed; using advisor reason as the message")
        decision = MainDecision(
            action=action,
            message=info_verdict.answer
            or "Thanks for your message — let me get back to you shortly.",
            reason="Main Agent unavailable; action from advisor precedence.",
        )

    return decision, verdicts


def handle_turn(
    session_id: str,
    candidate_message: str,
    conversation_start: str | None = None,
) -> TurnResult:
    """Process one candidate message and decide the bot's response.

    Args:
        session_id: Conversation key into :data:`store`.
        candidate_message: The candidate's latest message.
        conversation_start: ISO timestamp the conversation started. Relative
            dates ("next Friday") resolve against this, NOT against today —
            see CLAUDE.md section 6.3. Defaults to now.

    Returns:
        The decided :class:`TurnResult`.
    """
    conversation_start = conversation_start or _now_iso()
    history = get_history(session_id)

    conversation_text = render_messages(history.messages, latest=candidate_message)
    decision, verdicts = decide(
        conversation_text=conversation_text,
        candidate_message=candidate_message,
        conversation_start=conversation_start,
        history_messages=list(history.messages),
    )

    # Store the natural-language exchange only — never the JSON envelope.
    history.add_user_message(candidate_message)
    history.add_ai_message(decision.message)

    return TurnResult(
        action=decision.action,
        message=decision.message,
        reason=decision.reason,
        verdicts=verdicts,
    )


def predict_action(
    history: list[dict[str, str]],
    conversation_start: str | None = None,
) -> Action:
    """Predict the action for the next recruiter turn, given prior history.

    The evaluation entry point (CLAUDE.md section 10): returns only the label,
    without touching :data:`store`, so scoring a conversation cannot leak state
    into the next one.

    Args:
        history: Turns so far, each ``{"speaker": ..., "text": ...}``, ending on
            a candidate turn.
        conversation_start: The conversation's ``start_time_utc``.

    Returns:
        One of ``continue`` / ``schedule`` / ``end``.
    """
    conversation_start = conversation_start or _now_iso()
    conversation_text = "\n".join(
        f"{'Candidate' if t['speaker'] == 'candidate' else 'Recruiter'}: {t['text']}"
        for t in history
    )
    latest = history[-1]["text"] if history else ""

    # Replay prior turns as chat messages so the Main Agent sees the same shape
    # it sees in a live conversation.
    messages = [
        HumanMessage(content=t["text"])
        if t["speaker"] == "candidate"
        else AIMessage(content=t["text"])
        for t in history[:-1]
    ]

    decision, _ = decide(
        conversation_text=conversation_text,
        candidate_message=latest,
        conversation_start=conversation_start,
        history_messages=messages,
    )
    return decision.action


def describe_turn(result: TurnResult) -> str:
    """Render a turn's advisor verdicts for logging or a debug panel."""
    return json.dumps(result.verdicts, indent=2)
