"""Interview Scheduling Advisor.

Decides whether it is the right moment to schedule, and — when it is — reads the
recruiter's availability from the SQL database through a LangChain ``@tool``
(the spec's "function calling to interact with the SQL database") and proposes
the **3 nearest available slots**.

Relative expressions in the candidate's message ("next Friday", "Tuesday at 10")
are resolved against the conversation's ``start_time_utc``, NOT against today.
Using ``datetime.now()`` for a historical conversation puts every lookup in the
wrong week and returns the wrong slots (CLAUDE.md 6.3 and pitfall 2).

The database contains no Monday or Saturday slots and nothing outside
09:00-17:00. Only ever propose slots the database actually returns; a
plausible-sounding time can simply not exist (CLAUDE.md pitfall 9).

STATUS: scaffolding. Signatures are final; bodies are not implemented yet.
"""

from __future__ import annotations

from langchain.tools import tool

from app.modules.advisors.schemas import SchedulingVerdict

#: The only position this bot recruits for, as stored in the Schedule table.
POSITION = "Python Dev"


@tool
def get_available_slots(from_date: str, limit: int = 3, position: str = POSITION) -> str:
    """Find the nearest available interview slots for a position.

    Searches the recruiter's schedule forward from a date and returns the
    earliest slots that are still free. Use this whenever you need to propose
    interview times to a candidate.

    Args:
        from_date: ISO date (YYYY-MM-DD) to search forward from. Derive it from
            the conversation date plus whatever the candidate asked for, e.g.
            "next Friday".
        limit: How many slots to return. Defaults to 3.
        position: Role name as stored in the schedule. Defaults to "Python Dev".

    Returns:
        A human-readable list of slots, one per line, or a message saying that
        nothing is available in the searched range.
    """
    raise NotImplementedError


def resolve_relative_date(expression: str, conversation_start: str) -> str:
    """Resolve a relative date expression against the conversation's date.

    Args:
        expression: Candidate phrasing, e.g. "next Friday", "tomorrow", "the 12th".
        conversation_start: The conversation's ``start_time_utc``, ISO 8601.

    Returns:
        An ISO date (YYYY-MM-DD).
    """
    raise NotImplementedError


def build_scheduling_agent():
    """Build the Scheduling Advisor agent executor.

    Uses ``create_openai_tools_agent`` + ``AgentExecutor`` with
    :func:`get_available_slots` bound as its only tool, and the system prompt
    from ``app/prompts/scheduling_advisor.txt`` (CLAUDE.md 4.1, 4.3).
    """
    raise NotImplementedError


def advise(conversation_text: str, conversation_start: str) -> SchedulingVerdict:
    """Decide whether to schedule, and with which slots.

    Args:
        conversation_text: The full history rendered as plain text.
        conversation_start: The conversation's ``start_time_utc``, used to anchor
            every relative date.

    Returns:
        The validated verdict, with up to 3 slots when ``should_schedule`` is True.
    """
    raise NotImplementedError
