"""Scoring: accuracy, baseline, confusion matrix, error analysis.

Follows the course evaluation idiom from
``Course17/ML - Evaluation & Interpretation.ipynb`` and ``Course23/Nlp.ipynb``:
``accuracy_score`` + ``confusion_matrix`` + a seaborn heatmap
(``annot=True, fmt='d', cmap='Blues'``) + a labeled DataFrame, always alongside a
majority-class benchmark, and always with the misclassified rows printed out.

Four things every report must contain (CLAUDE.md section 10):

1. accuracy **and** the majority baseline next to it
2. the confusion matrix, with real class names on both axes
3. per-class metrics — ``end`` is the smallest class at 15 examples, and overall
   accuracy can hide a total failure on it
4. the misclassified turns, with history, true label and prediction

STATUS: scaffolding. Signatures are final; bodies are not implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.modules.evaluation.dataset import DecisionPoint
from app.modules.main_agent.actions import Action


@dataclass(frozen=True)
class EvalResult:
    """Outcome of one evaluation run.

    Attributes:
        accuracy: Share of correct predictions.
        baseline_accuracy: Majority-class accuracy on the same points.
        confusion: Confusion matrix, rows = true, columns = predicted, ordered by
            ``ACTIONS``.
        report: Per-class precision / recall / f1.
        errors: The misclassified decision points.
    """

    accuracy: float
    baseline_accuracy: float
    confusion: pd.DataFrame
    report: pd.DataFrame
    errors: pd.DataFrame


def predict_all(
    points: list[DecisionPoint],
    cache_path: Path | None = None,
) -> list[Action]:
    """Run the system over decision points and collect predicted actions.

    Args:
        points: Decision points to score.
        cache_path: JSON file used to memoize predictions keyed by
            ``(conversation_id, turn_id)``. Re-running the notebook should not
            re-spend tokens (CLAUDE.md 10.6).

    Returns:
        Predicted actions, aligned index-wise with ``points``.
    """
    raise NotImplementedError


def evaluate(points: list[DecisionPoint], predictions: list[Action]) -> EvalResult:
    """Score predictions against ground truth.

    Computes accuracy, the majority baseline, the confusion matrix, the per-class
    report and the error table in one pass.
    """
    raise NotImplementedError


def plot_confusion(result: EvalResult, title: str = "Confusion matrix"):
    """Draw the confusion matrix as a seaborn heatmap.

    ``annot=True, fmt='d', cmap='Blues'`` with class names on both axes — the
    course's plot.

    Returns:
        The matplotlib Axes, so the notebook can adjust it.
    """
    raise NotImplementedError


def error_table(points: list[DecisionPoint], predictions: list[Action]) -> pd.DataFrame:
    """Build the misclassification table.

    Columns: conversation id, turn id, the last candidate message, the true label
    and the predicted one. This table is worth more than the headline score — it
    is where a systematic ``schedule``/``end`` confusion becomes visible.
    """
    raise NotImplementedError
