"""Advisor output contracts.

Each advisor returns a small JSON object through ``prompt | llm | JsonOutputParser()``
(the parser the course taught) and the raw dict is then validated against the
matching model below. ``JsonOutputParser`` gives us the taught pattern; Pydantic
is the guard that stops a malformed LLM response from propagating into the
router.

``with_structured_output`` is deliberately not used — it was not taught in the
course and is not part of the pinned stack. See ENGINEERING.md section 9.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.modules.main_agent.actions import Action


class Slot(BaseModel):
    """One interview slot as stored in the Schedule table."""

    date: str = Field(description="ISO date, YYYY-MM-DD.")
    time: str = Field(description="24h time, HH:MM:SS.")


class ExitVerdict(BaseModel):
    """Conversation Exit Advisor verdict.

    ``should_end`` is True for BOTH terminal cases — the interview is confirmed,
    and the candidate opted out. It is not a "candidate is uninterested" flag.
    See ENGINEERING.md section 6.1.
    """

    should_end: bool = Field(description="True when the conversation should close now.")
    reason: str = Field(description="Short rationale.")


class SchedulingVerdict(BaseModel):
    """Interview Scheduling Advisor verdict.

    ``slots`` holds the available slots read from the SQL database for the window
    the candidate named, and is empty when ``should_schedule`` is False.

    It is also empty in one case where ``should_schedule`` is True:
    ``needs_availability``. The bot does not hand out dates unprompted — it asks
    the candidate when they are free, then looks that window up in the schedule.
    So the first scheduling turn carries no slots at all: there is no date to
    search on yet. See ENGINEERING.md section 4.12.
    """

    should_schedule: bool = Field(description="True when it is the right moment to schedule.")
    needs_availability: bool = Field(
        default=False,
        description="True when the candidate has not said when they are free yet, so the "
        "turn asks for their availability instead of proposing times.",
    )
    slots: list[Slot] = Field(default_factory=list, description="Free slots in the named window.")
    reason: str = Field(description="Short rationale.")


class InfoVerdict(BaseModel):
    """Conversation Info Advisor verdict.

    ``answer`` is grounded in the Chroma vector store when ``info_needed`` is
    True, and should also nudge the candidate toward scheduling — that is the
    advisor's stated secondary goal in the spec.
    """

    info_needed: bool = Field(description="True when the candidate asked something answerable.")
    answer: str = Field(default="", description="Grounded answer, empty when not needed.")
    sources: list[str] = Field(default_factory=list, description="Retrieved chunk ids.")


class MainDecision(BaseModel):
    """The Main Agent's decision for one turn."""

    action: Action = Field(description="One of continue / schedule / end.")
    message: str = Field(description="Text sent to the candidate.")
    reason: str = Field(default="", description="Short rationale, for logs and error analysis.")
