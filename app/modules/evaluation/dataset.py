"""Load the labeled dataset and turn it into evaluable decision points.

``data/sms_conversations.json`` holds 15 conversations / 103 turns, 59 of them
labeled. Only ``recruiter`` turns carry a label; candidate turns are always
``null``.

The evaluation task: given the history up to and including a candidate turn,
predict the label of the **next recruiter turn**.

Label distribution: ``continue`` 25, ``schedule`` 19, ``end`` 15 — majority class
about 42.4%, which is the baseline every accuracy number must be read against.

STATUS: scaffolding. Signatures are final; bodies are not implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.modules.main_agent.actions import Action

#: Share of the majority class (`continue`, 25/59). Predicting it for everything
#: scores this. Report it next to accuracy or the number is unreadable.
MAJORITY_BASELINE = 25 / 59


@dataclass(frozen=True)
class DecisionPoint:
    """One evaluable prediction.

    Attributes:
        conversation_id: Source conversation, and the unit the split works on.
        turn_id: The recruiter turn being predicted.
        history: Turns before it, each ``{"speaker": ..., "text": ...}``, ending
            on a candidate turn (except the very first, which has no history).
        conversation_start: The conversation's ``start_time_utc``. Relative dates
            anchor here, not to today.
        label: The ground-truth action.
    """

    conversation_id: int
    turn_id: int
    history: list[dict[str, str]]
    conversation_start: str
    label: Action


def load_conversations(path=None) -> list[dict]:
    """Load the raw conversations JSON.

    Args:
        path: Override the default ``settings.conversations_json``.
    """
    raise NotImplementedError


def build_decision_points(conversations: list[dict]) -> list[DecisionPoint]:
    """Expand conversations into one decision point per labeled recruiter turn.

    Returns:
        59 points for the full dataset.
    """
    raise NotImplementedError


def split_by_conversation(
    conversations: list[dict],
    test_size: int = 5,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split conversation ids into train and test.

    Splitting happens at the **conversation** level, never the turn level: turns
    in one conversation share a history prefix, so a turn-level split leaks.

    With only 15 conversations the split is also stratified by ending flavour, so
    both opt-out endings and booked endings appear on each side — otherwise the
    test set can end up with no opt-outs at all and the ``end`` class becomes
    untestable.

    Args:
        conversations: Raw conversations.
        test_size: How many conversations to hold out.
        seed: RNG seed, fixed so the split is reproducible across runs and
            across the fine-tuning pipeline.

    Returns:
        ``(train_ids, test_ids)``.
    """
    raise NotImplementedError


def render_history(history: list[dict[str, str]]) -> str:
    """Render turns as the plain text the advisors receive.

    One ``Speaker: message`` line per turn, matching the format used in the
    fine-tuning JSONL so training and inference see identical inputs.
    """
    raise NotImplementedError
