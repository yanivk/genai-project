"""Evaluation of the multi-agent system against the labeled dataset.

    dataset  Load data/sms_conversations.json, build decision points, split.
    metrics  Accuracy, majority baseline, confusion matrix, error table.

The notebook ``tests/test_evals.ipynb`` is the spec deliverable; this package
holds the reusable logic so the notebook stays readable.

Non-negotiable: **split by conversation_id, never by turn**. Turns inside one
conversation share a history prefix, so a turn-level split leaks the test set
into both training and few-shot examples. See ENGINEERING.md section 10.
"""

from app.modules.evaluation.dataset import (
    DecisionPoint,
    build_decision_points,
    ending_flavour,
    load_conversations,
    split_by_conversation,
)

__all__ = [
    "DecisionPoint",
    "build_decision_points",
    "ending_flavour",
    "load_conversations",
    "split_by_conversation",
]
