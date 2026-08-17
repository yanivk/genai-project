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

Output goes to ``data/*.jsonl``, which is gitignored — it is derived, and a JSONL
built from the wrong split is easy to commit by accident.

STATUS: scaffolding. Signatures are final; bodies are not implemented yet.
"""

from __future__ import annotations

from pathlib import Path


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
    raise NotImplementedError


def write_jsonl(examples: list[dict], out_path: Path) -> Path:
    """Write rows to a JSONL file, one compact JSON object per line.

    Args:
        examples: Rows from :func:`build_examples`.
        out_path: Destination, expected under ``data/`` so it stays gitignored.

    Returns:
        The path written.
    """
    raise NotImplementedError


def describe(examples: list[dict]) -> str:
    """Summarize a built dataset before spending money on a job.

    Reports the row count and the ``should_end`` True/False balance, and flags
    the case where only one flavour of ``end`` (booked vs opted-out) is present.
    """
    raise NotImplementedError
