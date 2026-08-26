"""Interview Scheduling Advisor.

Decides whether it is the right moment to schedule, and — when it is — reads the
recruiter's availability from the SQL database through a LangChain ``@tool``
(the spec's "function calling to interact with the SQL database").

**The candidate names the window; the database names the times.** The bot never
opens with dates of its own. Once the screening gate is passed it asks *when the
candidate is free* — pointing them at the availability calendar in the app's left
sidebar, which is rendered from this same schedule — and only then queries the
schedule for that window. That first turn is what ``needs_availability`` on the
verdict carries: schedule this turn, but with no slots, because there is no date
to search on yet. The turn after, the candidate's own words ("Tuesday afternoon",
"next week", "the 19th") become the ``when`` / ``until`` / ``time_of_day``
arguments of the tool, and only slots the query actually returned are ever
offered.

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
import re
from datetime import datetime, timedelta

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.modules.advisors.schemas import SchedulingVerdict, Slot
from app.modules.common import fill, get_llm, load_prompt, parse_json_output
from app.modules.database.engine import get_available_slots as db_get_available_slots
from app.modules.database.engine import is_slot_available

logger = logging.getLogger(__name__)

#: The only position this bot recruits for, as stored in the Schedule table.
POSITION = "Python Dev"

#: An explicit date the advisor computed itself, rather than candidate phrasing.
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

#: Candidate phrasing -> inclusive ``(time_from, time_to)`` bounds. The schedule
#: only ever runs 09:00-17:00, so these carve up that range rather than a whole
#: day: an "afternoon" that ended at 23:59 would be a lie about the calendar.
TIME_OF_DAY = {
    "morning": ("09:00:00", "11:00:00"),
    "midday": ("11:00:00", "13:00:00"),
    "lunch": ("11:00:00", "13:00:00"),
    "afternoon": ("12:00:00", "17:00:00"),
    "evening": ("15:00:00", "17:00:00"),
    "late": ("15:00:00", "17:00:00"),
    "early": ("09:00:00", "11:00:00"),
}

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
        expression: Candidate phrasing, e.g. "next Friday", "tomorrow". Also
            accepts an explicit ``YYYY-MM-DD`` date, which is passed straight
            through — that is how the advisor searches past slots the candidate
            has already turned down.
            An empty or unrecognised expression resolves to the conversation date
            itself, which makes the caller search forward from "now".
        conversation_start: The conversation's ``start_time_utc``, ISO 8601.

    Returns:
        An ISO date (YYYY-MM-DD).
    """
    base = datetime.fromisoformat(conversation_start.replace("Z", "+00:00")).date()
    expr = (expression or "").strip().lower()

    # An explicit date wins over any phrase matching. Without this the advisor has
    # no way to say "start from the 19th" after the candidate rejected the 18th,
    # and every retry returns the same three slots.
    if _ISO_DATE.fullmatch(expr):
        return expr

    # Before "tomorrow": the substring is inside "day after tomorrow", which means
    # +2, not +1. Ordering these the other way round silently answers the wrong day.
    if "day after tomorrow" in expr or "overmorrow" in expr:
        return (base + timedelta(days=2)).isoformat()
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
    until: str = "",
    time_of_day: str = "",
    limit: int = 3,
    position: str = POSITION,
) -> str:
    """Look up free interview slots inside the window the candidate said they are free.

    Always call this before mentioning any time to a candidate. Never invent a
    slot: the calendar has no Monday or Saturday availability and only runs
    09:00-17:00, so a reasonable-sounding time may simply not exist.

    Call it once the candidate has told you when they are free, and pass their own
    words through. Do NOT call it just to have dates to offer before asking them —
    asking is what needs_availability is for.

    Args:
        conversation_date: The date this conversation is taking place, as
            YYYY-MM-DD. Take it from the Context section of your instructions.
        when: The start of the window the candidate named. Either their own wording
            ("next Friday", "tomorrow", "day after tomorrow", "Tuesday") or an
            explicit YYYY-MM-DD date. Leave empty to search forward from the
            conversation date.
            If the candidate turned down slots you already proposed, pass the
            explicit date of the day AFTER the last one you offered — otherwise
            this returns the same slots again.
        until: The end of that window, inclusive — same formats as ``when``. Pass
            the SAME value as ``when`` when the candidate named one single day, so
            the answer stays inside the day they actually offered. Leave it empty
            when they were open-ended ("sometime next week", "whenever suits") and
            you want the nearest slots on or after ``when``.
        time_of_day: Which part of the day they asked for — "morning", "midday",
            "afternoon" or "evening". Leave empty when they did not say.
        limit: How many slots to return. Defaults to 3. Pass a larger number, up to
            9, when you are looking over one whole day.
        position: Role name as stored in the schedule. Defaults to "Python Dev".

    Returns:
        One slot per line as "YYYY-MM-DD HH:MM:SS (Weekday)", or a message saying
        nothing is available in that window — in which case call again with a wider
        one rather than making a time up.
    """
    anchor = f"{conversation_date}T00:00:00Z"
    try:
        from_date = resolve_relative_date(when, anchor)
    except (ValueError, TypeError):
        from_date = conversation_date
    try:
        to_date = resolve_relative_date(until, anchor) if (until or "").strip() else None
    except (ValueError, TypeError):
        to_date = None
    # A window that runs backwards returns nothing at all, which reads to the model
    # as "the candidate's day is fully booked". Treat it as one single day instead.
    if to_date and to_date < from_date:
        to_date = from_date

    time_from, time_to = TIME_OF_DAY.get(_time_of_day_key(time_of_day), (None, None))

    frame = db_get_available_slots(
        from_date=from_date,
        position=position,
        limit=limit,
        to_date=to_date,
        time_from=time_from,
        time_to=time_to,
    )
    window = describe_window(from_date, to_date, time_of_day)
    if frame.empty:
        return f"No available {position} slots {window}."

    lines = [
        f"{row['date']} {row['time']} "
        f"({datetime.fromisoformat(row['date']).strftime('%A')})"
        for _, row in frame.iterrows()
    ]
    return f"Available {position} slots {window}:\n" + "\n".join(lines)


def _time_of_day_key(expression: str) -> str:
    """Return the :data:`TIME_OF_DAY` key an expression names, or ``""``.

    Substring matching, because the model passes the candidate's phrasing more
    often than a bare keyword. The longest match wins rather than the first found,
    so "early afternoon" lands on afternoon instead of on early.
    """
    expr = (expression or "").strip().lower()
    matches = [key for key in TIME_OF_DAY if key in expr]
    return max(matches, key=len) if matches else ""


def describe_window(from_date: str, to_date: str | None, time_of_day: str = "") -> str:
    """Phrase the searched window, so the model can quote it back accurately."""
    if to_date == from_date:
        span = f"on {from_date}"
    elif to_date:
        span = f"between {from_date} and {to_date}"
    else:
        span = f"from {from_date}"
    key = _time_of_day_key(time_of_day)
    return f"{span} ({key})" if key else span


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


def validate_slots(
    verdict: SchedulingVerdict,
    conversation_date: str,
    position: str = POSITION,
) -> SchedulingVerdict:
    """Drop any slot that is not actually free in the schedule, in code.

    The prompt tells the advisor never to invent a time, and it mostly obeys —
    but "mostly" is not a guarantee, and the failure is invisible: asked for a
    Monday, it answered with three plausible morning slots on the Tuesday after,
    none of which were free. The candidate is then offered an interview that does
    not exist, and the calendar cannot honour it.

    So every slot is re-read from the database before the verdict leaves this
    module (CLAUDE.md pitfall 9). What survives is real. If nothing survives, the
    nearest genuine openings from the earliest date the advisor was aiming at are
    substituted, which keeps the candidate's own window as close as the schedule
    allows.

    Args:
        verdict: The advisor's verdict, after JSON validation.
        conversation_date: ISO date the conversation is anchored on — the fallback
            anchor when the advisor named no usable date at all.
        position: Role name as stored in the schedule.
    """
    if not verdict.should_schedule or verdict.needs_availability:
        return verdict

    real = [slot for slot in verdict.slots if is_slot_available(slot.date, slot.time, position)]
    if len(real) == len(verdict.slots):
        return verdict

    logger.warning(
        "Scheduling Advisor proposed %d slot(s) the schedule does not have free: %s",
        len(verdict.slots) - len(real),
        [f"{s.date} {s.time}" for s in verdict.slots if s not in real],
    )
    if real:
        return verdict.model_copy(update={"slots": real})

    # Aim at the day the advisor was aiming at: it read the candidate's window,
    # and only the times inside it were wrong.
    anchor = min((slot.date for slot in verdict.slots), default="") or conversation_date
    frame = db_get_available_slots(from_date=anchor, position=position, limit=3)
    substituted = [
        Slot(date=row["date"], time=row["time"]) for _, row in frame.iterrows()
    ]
    if not substituted:
        return verdict.model_copy(update={"slots": []})
    return verdict.model_copy(
        update={
            "slots": substituted,
            "reason": f"{verdict.reason} (Times not free in the calendar were replaced "
            f"with its real openings from {anchor}.)",
        }
    )


def reconcile(verdict: SchedulingVerdict) -> SchedulingVerdict:
    """Keep ``needs_availability`` and ``slots`` from contradicting each other.

    The two are mutually exclusive by construction: either we are asking the
    candidate when they are free, or we are offering what the query returned for
    the window they gave. The model can still hand back both, or neither:

    * ``should_schedule`` with no slots is an *ask* — whatever the model called
      the field. Left as-is, the Main Agent has a schedule turn with nothing to
      propose, and the reliable way it fills that gap is by inventing a time.
    * ``needs_availability`` alongside real slots is stale: the window is known,
      so asking again would ignore what the candidate already said.
    """
    if not verdict.should_schedule:
        return verdict.model_copy(update={"needs_availability": False, "slots": []})
    if verdict.slots:
        return verdict.model_copy(update={"needs_availability": False})
    return verdict.model_copy(update={"needs_availability": True})


def advise(conversation_text: str, conversation_start: str) -> SchedulingVerdict:
    """Decide whether to schedule, and — once the candidate said when — with which slots.

    Args:
        conversation_text: The full history rendered as plain text.
        conversation_start: The conversation's ``start_time_utc``, used to anchor
            every relative date.

    Returns:
        The validated verdict. When ``should_schedule`` is True it carries either
        ``needs_availability`` (ask the candidate for their window) or slots that
        have each been re-read from the schedule — never both, never neither.
        Degrades to "don't schedule" on any failure rather than raising.
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
        verdict = SchedulingVerdict.model_validate(parse_json_output(output))
        # Validate first, reconcile second: a verdict left with no real slots is an
        # ask for availability, not a scheduling turn with nothing to offer.
        return reconcile(validate_slots(verdict, conversation_date))
    except Exception:  # noqa: BLE001 - degrade instead of killing the turn
        logger.exception("Scheduling Advisor failed; defaulting to should_schedule=False")
        return SchedulingVerdict(
            should_schedule=False, slots=[], reason="Scheduling Advisor unavailable."
        )
