"""The three actions the Main Agent can take.

Defined once, imported everywhere. Never retype the string literals — a typo in
one branch silently disables it, and these same strings are the class labels in
``data/sms_conversations.json`` and in the evaluation notebook.

See ENGINEERING.md sections 1 and 6.1.
"""

from __future__ import annotations

from typing import Final, Literal

#: Keep the dialogue going — ask or answer a question, exchange information.
CONTINUE: Final = "continue"

#: Propose or renegotiate interview time slots.
SCHEDULE: Final = "schedule"

#: The conversation is over. Terminal in BOTH directions: the interview is
#: confirmed, or the candidate opted out. This is the single most misread label
#: in this project — see ENGINEERING.md section 6.1.
END: Final = "end"

Action = Literal["continue", "schedule", "end"]

#: Canonical order, used for confusion-matrix axes so every report lines up.
ACTIONS: Final[tuple[Action, ...]] = (CONTINUE, SCHEDULE, END)


def is_valid_action(value: str) -> bool:
    """Return True when ``value`` is one of the three canonical actions."""
    return value in ACTIONS


def parse_action(value: str) -> Action:
    """Normalise and validate a model-produced action string.

    Args:
        value: Raw action emitted by an LLM, possibly padded or miscased.

    Returns:
        The canonical lowercase action.

    Raises:
        ValueError: If ``value`` is not one of the three actions.
    """
    normalised = value.strip().lower()
    if not is_valid_action(normalised):
        raise ValueError(f"Invalid action {value!r}; expected one of {ACTIONS}.")
    return normalised  # type: ignore[return-value]
