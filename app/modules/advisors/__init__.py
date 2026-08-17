"""Supporting agents ("Advisors") consulted by the Main Agent each turn.

    exit_advisor        Should the conversation end? Fine-tuned.
    scheduling_advisor  Is it time to schedule? Reads the SQL DB via a @tool.
    info_advisor        Answers position questions from the Chroma vector DB.
    schemas             Pydantic output contracts for all three.

Advisors are stateless. They receive the rendered conversation history as plain
text; only the Main Agent owns memory. See CLAUDE.md sections 4.2 and 9.
"""

from app.modules.advisors.schemas import (
    ExitVerdict,
    InfoVerdict,
    MainDecision,
    SchedulingVerdict,
    Slot,
)

__all__ = ["ExitVerdict", "InfoVerdict", "MainDecision", "SchedulingVerdict", "Slot"]
