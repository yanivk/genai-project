"""Load the labeled dataset and turn it into evaluable decision points.

``data/sms_conversations.json`` holds 15 conversations / 103 turns, 59 of them
labeled. Only ``recruiter`` turns carry a label; candidate turns are always
``null``.

The evaluation task: given the history up to and including a candidate turn,
predict the label of the **next recruiter turn**.

Label distribution: ``continue`` 25, ``schedule`` 19, ``end`` 15 — majority class
about 42.4%, which is the baseline every accuracy number must be read against.

"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from app.modules.main_agent.actions import END, Action

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


def load_conversations(path: Path | None = None) -> list[dict]:
    """Load the raw conversations JSON.

    Args:
        path: Override the default ``settings.conversations_json``.
    """
    source = Path(path) if path else settings.conversations_json
    return json.loads(source.read_text(encoding="utf-8"))


def build_decision_points(conversations: list[dict]) -> list[DecisionPoint]:
    """Expand conversations into one decision point per labeled recruiter turn.

    Returns:
        59 points for the full dataset.
    """
    points: list[DecisionPoint] = []
    for conversation in conversations:
        turns = conversation["turns"]
        for index, turn in enumerate(turns):
            if turn["speaker"] != "recruiter" or not turn["label"]:
                continue
            points.append(
                DecisionPoint(
                    conversation_id=conversation["conversation_id"],
                    turn_id=turn["turn_id"],
                    history=[
                        {"speaker": t["speaker"], "text": t["text"]}
                        for t in turns[:index]
                    ],
                    conversation_start=conversation["start_time_utc"],
                    label=turn["label"],
                )
            )
    return points


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
    # Ending flavour: an `end` turn that follows a candidate accepting a slot is
    # a booking; anything else is an opt-out. With only 4 opt-outs in the whole
    # dataset, an unstratified split can leave the test set with none.
    opt_out, booked = [], []
    for conversation in conversations:
        (opt_out if _is_opt_out(conversation) else booked).append(
            conversation["conversation_id"]
        )

    rng = random.Random(seed)
    rng.shuffle(opt_out)
    rng.shuffle(booked)

    # Take opt-outs proportionally, but always at least one when any exist.
    n_opt_out = min(len(opt_out), max(1, round(test_size * len(opt_out) / len(conversations))))
    n_booked = test_size - n_opt_out

    test_ids = opt_out[:n_opt_out] + booked[:n_booked]
    train_ids = [
        c["conversation_id"] for c in conversations if c["conversation_id"] not in test_ids
    ]
    return train_ids, test_ids


#: Phrases that mark a conversation as ending because the candidate withdrew,
#: rather than because an interview was booked. Also read by the fine-tuning
#: dataset builder, which needs the same distinction per decision point.
OPT_OUT_MARKERS = (
    "no longer interested",
    "not interested",
    "remove me",
    "stop texting",
    "found a job",
    "found another",
)


def _is_opt_out(conversation: dict) -> bool:
    """True when the conversation ends with the candidate withdrawing."""
    return any(
        marker in turn["text"].lower()
        for turn in conversation["turns"]
        if turn["speaker"] == "candidate"
        for marker in OPT_OUT_MARKERS
    )


def ending_flavour(conversation: dict) -> str:
    """Return ``"opt-out"`` or ``"booked"`` for a conversation.

    Exposed so the notebook can show that both flavours survived the split — the
    `end` class is meaningless if the test set only contains one of them.
    """
    return "opt-out" if _is_opt_out(conversation) else "booked"


def render_history(history: list[dict[str, str]]) -> str:
    """Render turns as the plain text the advisors receive.

    One ``Speaker: message`` line per turn, matching the format used in the
    fine-tuning JSONL so training and inference see identical inputs.
    """
    return "\n".join(
        f"{'Candidate' if t['speaker'] == 'candidate' else 'Recruiter'}: {t['text']}"
        for t in history
    )
