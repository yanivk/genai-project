"""Interview Scheduling Advisor.

Decides whether it is the right moment to schedule, and — when it is — reads the
recruiter's availability from the SQL database through a LangChain ``@tool``
(the spec's "function calling to interact with the SQL database") and proposes
the **3 nearest available slots**.

Relative expressions in the candidate's message ("next Friday", "Tuesday at 10")
are resolved against the conversation's ``start_time_utc``, NOT against today.
Using ``datetime.now()`` for a historical conversation puts every lookup in the
wrong week and returns the wrong slots (CLAUDE.md 6.3 and pitfall 2).

The resolution happens **inside the tool**, deterministically, rather than being
left to the model: LLMs are unreliable at date arithmetic, and a wrong date here
fails silently as "no slots available". The model supplies the candidate's own
phrasing and the conversation date; Python does the calendar maths.

The database contains no Monday or Saturday slots and nothing outside
09:00-17:00. Only ever propose slots the database actually returns; a
plausible-sounding time can simply not exist (CLAUDE.md pitfall 9).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.modules.advisors.schemas import SchedulingVerdict
from app.modules.common import fill, get_llm, load_prompt, parse_json_output
from app.modules.database.engine import get_available_slots as db_get_available_slots

logger = logging.getLogger(__name__)

#: The only position this bot recruits for, as stored in the Schedule table.
POSITION = "Python Dev"

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def resolve_relative_date(expression: str, conversation_start: str) -> str:
    """Resolve a relative date expression against the conversation's date.

    Args:
        expression: Candidate phrasing, e.g. "next Friday", "tomorrow".
            An empty or unrecognised expression resolves to the conversation date
            itself, which makes the caller search forward from "now".
        conversation_start: The conversation's ``start_time_utc``, ISO 8601.

    Returns:
        An ISO date (YYYY-MM-DD).
    """
    base = datetime.fromisoformat(conversation_start.replace("Z", "+00:00")).date()
    expr = (expression or "").strip().lower()

    if not expr or "today" in expr or "tonight" in expr:
        return base.isoformat()
    if "tomorrow" in expr:
        return (base + timedelta(days=1)).isoformat()

    for name, index in _WEEKDAYS.items():
        if name in expr:
            # Always the coming occurrence: "Tuesday" said on a Tuesday means the
            # next one, not today. This matches how the dataset uses "next X".
            ahead = (index - base.weekday()) % 7 or 7
            if "week after" in expr or "in two weeks" in expr:
                ahead += 7
            return (base + timedelta(days=ahead)).isoformat()

    if "next week" in expr:
        return (base + timedelta(days=7)).isoformat()

    return base.isoformat()


@tool
def get_available_slots(
    conversation_date: str,
    when: str = "",
    limit: int = 3,
    position: str = POSITION,
) -> str:
    """Find the nearest available interview slots in the recruiter's calendar.

    Always call this before proposing any time to a candidate. Never invent a
    slot: the calendar has no Monday or Saturday availability and only runs
    09:00-17:00, so a reasonable-sounding time may simply not exist.

    Args:
        conversation_date: The date this conversation is taking place, as
            YYYY-MM-DD. Take it from the Context section of your instructions.
        when: The candidate's own wording for when they want to meet, e.g.
            "next Friday", "tomorrow", "Tuesday". Leave empty to search forward
            from the conversation date.
        limit: How many slots to return. Defaults to 3.
        position: Role name as stored in the schedule. Defaults to "Python Dev".

    Returns:
        One slot per line as "YYYY-MM-DD HH:MM:SS (Weekday)", or a message
        saying nothing is available.
    """
    try:
        from_date = resolve_relative_date(when, f"{conversation_date}T00:00:00Z")
    except (ValueError, TypeError):
        from_date = conversation_date

    frame = db_get_available_slots(from_date=from_date, position=position, limit=limit)
    if frame.empty:
        return f"No available {position} slots on or after {from_date}."

    lines = [
        f"{row['date']} {row['time']} "
        f"({datetime.fromisoformat(row['date']).strftime('%A')})"
        for _, row in frame.iterrows()
    ]
    return f"Available {position} slots from {from_date}:\n" + "\n".join(lines)


def build_scheduling_agent() -> AgentExecutor:
    """Build the Scheduling Advisor agent executor.

    Uses ``create_openai_tools_agent`` + ``AgentExecutor`` with
    :func:`get_available_slots` bound as its only tool, and the system prompt
    from ``app/prompts/scheduling_advisor.txt`` (CLAUDE.md 4.1, 4.3).

    Placeholder order is mandatory: system -> agent_scratchpad -> user.
    """
    tools = [get_available_slots]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_text}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("user", "{input}"),
        ]
    )
    agent = create_openai_tools_agent(get_llm(), tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=4)


def advise(conversation_text: str, conversation_start: str) -> SchedulingVerdict:
    """Decide whether to schedule, and with which slots.

    Args:
        conversation_text: The full history rendered as plain text.
        conversation_start: The conversation's ``start_time_utc``, used to anchor
            every relative date.

    Returns:
        The validated verdict, with up to 3 slots when ``should_schedule`` is
        True. Degrades to "don't schedule" on any failure rather than raising.
    """
    conversation_date = conversation_start[:10]
    system_text = fill(
        load_prompt("scheduling_advisor"),
        conversation=conversation_text,
        conversation_start=conversation_date,
    )
    try:
        output = build_scheduling_agent().invoke(
            {
                "system_text": system_text,
                "input": "Return your JSON verdict for the conversation above.",
            }
        )["output"]
        return SchedulingVerdict.model_validate(parse_json_output(output))
    except Exception:  # noqa: BLE001 - degrade instead of killing the turn
        logger.exception("Scheduling Advisor failed; defaulting to should_schedule=False")
        return SchedulingVerdict(
            should_schedule=False, slots=[], reason="Scheduling Advisor unavailable."
        )
