"""Build the Exit Advisor fine-tuning dataset.

Turns ``data/sms_conversations.json`` into OpenAI chat-format JSONL. Each row is
one decision point: the history up to a candidate turn, and the ``should_end``
verdict implied by the label of the recruiter turn that followed.

Two rules govern this module, both from CLAUDE.md section 11:

1. **Training rows come from the training split only.** The held-out
   conversations never appear here. Building the JSONL from all 15 conversations
   and then evaluating on 5 of them is the easiest way to silently invalidate the
   whole evaluation.
2. **The assistant target is the exact JSON contract from CLAUDE.md section 9**,
   so the fine-tuned model is a drop-in replacement for the prompted one.

Both ``end`` cases must be represented: booked-and-confirmed AND
candidate-opted-out. In the dataset that is 11 vs 4.

The system message comes from
:func:`app.modules.advisors.exit_advisor.build_system_text` with
``finetuned=True``, and the user message is the same ``DIRECTIVE`` the advisor
sends at inference time. Both are called rather than re-typed, so a training row
is the same message list the model will see in production.

Output goes to ``data/*.jsonl``, which is gitignored — it is derived, and a JSONL
built from the wrong split is easy to commit by accident.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.modules.advisors.exit_advisor import DIRECTIVE, build_system_text
from app.modules.common import render_history
from app.modules.evaluation.dataset import (
    OPT_OUT_MARKERS,
    DecisionPoint,
    build_decision_points,
    load_conversations,
)
from app.modules.main_agent.actions import END

#: Minimum rows the OpenAI fine-tuning API accepts for a job.
MIN_EXAMPLES = 10


def _is_opt_out_point(point: DecisionPoint) -> bool:
    """True when the candidate withdrew in the last turn of this history.

    Distinguishes the two flavours of ``end`` so the generated ``reason`` says
    which one it is, and so :func:`describe` can refuse to launch a job that has
    only learned one of them.
    """
    candidate_turns = [t for t in point.history if t["speaker"] == "candidate"]
    if not candidate_turns:
        return False
    last = candidate_turns[-1]["text"].lower()
    return any(marker in last for marker in OPT_OUT_MARKERS)


def _reason_for(point: DecisionPoint) -> str:
    """Write the ``reason`` field of a training target.

    The label gives ``should_end``; ``reason`` has no ground truth in the
    dataset, so it is generated deterministically from the situation. It is
    explanatory only — :func:`~app.modules.main_agent.orchestrator.resolve_action`
    reads the boolean and ignores this string — but the model is trained to
    produce it, so it has to be consistent rather than arbitrary.
    """
    if point.label == END:
        if _is_opt_out_point(point):
            return "The candidate opted out; following up further would be unwelcome."
        return "A time has been agreed; the interview is settled."
    if point.history and point.history[-1]["text"].strip().endswith("?"):
        return "The candidate is asking a question and remains engaged."
    return "Nothing is settled yet; the conversation should continue."


def build_examples(conversation_ids: list[int]) -> list[dict]:
    """Build chat-format training rows from the given conversations.

    Args:
        conversation_ids: The TRAINING split. Passing the full set of ids is a
            bug, not a shortcut.

    Returns:
        Rows shaped ``{"messages": [system, user, assistant]}``, where the
        assistant content is a JSON string matching
        :class:`~app.modules.advisors.schemas.ExitVerdict`.
    """
    wanted = set(conversation_ids)
    points = [
        p for p in build_decision_points(load_conversations()) if p.conversation_id in wanted
    ]

    examples: list[dict] = []
    for point in points:
        conversation_text = render_history(point.history)
        target = {
            "should_end": point.label == END,
            "reason": _reason_for(point),
        }
        examples.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": build_system_text(conversation_text, finetuned=True),
                    },
                    {"role": "user", "content": DIRECTIVE},
                    # Compact separators: the model should learn to emit the
                    # contract, not a particular amount of whitespace.
                    {
                        "role": "assistant",
                        "content": json.dumps(target, separators=(",", ":")),
                    },
                ]
            }
        )
    return examples


def write_jsonl(examples: list[dict], out_path: Path) -> Path:
    """Write rows to a JSONL file, one compact JSON object per line.

    Args:
        examples: Rows from :func:`build_examples`.
        out_path: Destination, expected under ``data/`` so it stays gitignored.

    Returns:
        The path written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return out_path


def describe(examples: list[dict]) -> str:
    """Summarize a built dataset before spending money on a job.

    Reports the row count and the ``should_end`` True/False balance, and flags
    the case where only one flavour of ``end`` (booked vs opted-out) is present —
    a model trained on opt-outs alone learns that ``end`` means rejection, which
    is wrong for 11 of the 15 conversations (CLAUDE.md 6.1).
    """
    targets = [json.loads(e["messages"][-1]["content"]) for e in examples]
    ending = [t for t in targets if t["should_end"]]
    opted_out = sum(1 for t in ending if "opted out" in t["reason"])
    booked = len(ending) - opted_out

    lines = [
        f"{len(examples)} training rows",
        f"  should_end=True : {len(ending)}  (booked {booked}, opted out {opted_out})",
        f"  should_end=False: {len(targets) - len(ending)}",
    ]
    if len(examples) < MIN_EXAMPLES:
        lines.append(
            f"  WARNING: OpenAI requires at least {MIN_EXAMPLES} rows; this job will be rejected."
        )
    if not ending:
        lines.append("  WARNING: no should_end=True rows — the model cannot learn to end.")
    elif not booked or not opted_out:
        missing = "booked-and-confirmed" if not booked else "candidate-opted-out"
        lines.append(
            f"  WARNING: no {missing} endings. `end` is terminal in BOTH directions; "
            "training on one flavour teaches the wrong rule (CLAUDE.md 6.1)."
        )
    return "\n".join(lines)
