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

"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.config import settings
from app.modules.evaluation.dataset import DecisionPoint
from app.modules.main_agent.actions import ACTIONS, CONTINUE, Action


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


#: Reserved key inside a cache file; never a decision point.
_FINGERPRINT_KEY = "__fingerprint__"

#: Everything that changes what the system predicts. A cached run made under a
#: different value is not comparable to the current one.
_FINGERPRINTED_PROMPTS = ("main_agent", "exit_advisor", "scheduling_advisor", "info_advisor")


def prediction_fingerprint() -> str:
    """Hash the prompts and models that decide a prediction.

    Predictions are cached so re-running the notebook costs nothing, but the
    cache key is the decision point alone. Without this, editing a prompt and
    re-running would silently replay the old predictions and report the previous
    score as if it were the new one — the kind of mistake that invalidates a
    result without leaving a trace.
    """
    digest = hashlib.sha256()
    for name in _FINGERPRINTED_PROMPTS:
        digest.update(settings.prompt_path(name).read_bytes())
    digest.update(settings.openai_model.encode())
    digest.update(settings.exit_advisor_model.encode())
    return digest.hexdigest()[:16]


def predict_all(
    points: list[DecisionPoint],
    cache_path: Path | None = None,
    max_workers: int = 4,
) -> list[Action]:
    """Run the system over decision points and collect predicted actions.

    Args:
        points: Decision points to score.
        cache_path: JSON file used to memoize predictions keyed by
            ``(conversation_id, turn_id)``. Re-running the notebook should not
            re-spend tokens (CLAUDE.md 10.6). The cache is discarded whenever the
            prompts or the models change — see :func:`prediction_fingerprint`.

    Returns:
        Predicted actions, aligned index-wise with ``points``.
    """
    from app.modules.main_agent.orchestrator import predict_action

    fingerprint = prediction_fingerprint()
    cache: dict[str, str] = {}
    if cache_path and Path(cache_path).is_file():
        stored = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        if stored.get(_FINGERPRINT_KEY) == fingerprint:
            cache = {k: v for k, v in stored.items() if k != _FINGERPRINT_KEY}

    def key(point: DecisionPoint) -> str:
        return f"{point.conversation_id}:{point.turn_id}"

    todo = [p for p in points if key(p) not in cache]
    if todo:
        # Each point is an independent multi-call chain; running them serially
        # makes a full sweep take minutes for no reason.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(
                pool.map(
                    lambda p: predict_action(p.history, p.conversation_start), todo
                )
            )
        cache.update({key(p): r for p, r in zip(todo, results)})
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(
                json.dumps({_FINGERPRINT_KEY: fingerprint, **cache}, indent=2),
                encoding="utf-8",
            )

    return [cache[key(p)] for p in points]  # type: ignore[misc]


def evaluate(points: list[DecisionPoint], predictions: list[Action]) -> EvalResult:
    """Score predictions against ground truth.

    Computes accuracy, the majority baseline, the confusion matrix, the per-class
    report and the error table in one pass.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    truth = [p.label for p in points]
    labels = list(ACTIONS)

    accuracy = accuracy_score(truth, predictions)

    # Majority baseline: predict the most frequent class in the truth for
    # everything. Reported alongside accuracy so the number is interpretable.
    majority = Counter(truth).most_common(1)[0][0] if truth else CONTINUE
    baseline_accuracy = accuracy_score(truth, [majority] * len(truth))

    matrix = confusion_matrix(truth, predictions, labels=labels)
    confusion = pd.DataFrame(
        matrix,
        index=[f"true: {label}" for label in labels],
        columns=[f"pred: {label}" for label in labels],
    )

    report = pd.DataFrame(
        classification_report(
            truth, predictions, labels=labels, output_dict=True, zero_division=0
        )
    ).transpose()

    return EvalResult(
        accuracy=float(accuracy),
        baseline_accuracy=float(baseline_accuracy),
        confusion=confusion,
        report=report,
        errors=error_table(points, predictions),
    )


def plot_confusion(result: EvalResult, title: str = "Confusion matrix"):
    """Draw the confusion matrix as a seaborn heatmap.

    ``annot=True, fmt='d', cmap='Blues'`` with class names on both axes — the
    course's plot.

    Returns:
        The matplotlib Axes, so the notebook can adjust it.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    axes = sns.heatmap(
        result.confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=list(ACTIONS),
        yticklabels=list(ACTIONS),
    )
    axes.set_xlabel("Predicted")
    axes.set_ylabel("True")
    axes.set_title(title)
    plt.tight_layout()
    return axes


def error_table(points: list[DecisionPoint], predictions: list[Action]) -> pd.DataFrame:
    """Build the misclassification table.

    Columns: conversation id, turn id, the last candidate message, the true label
    and the predicted one. This table is worth more than the headline score — it
    is where a systematic ``schedule``/``end`` confusion becomes visible.
    """
    rows = []
    for point, predicted in zip(points, predictions):
        if predicted == point.label:
            continue
        last_candidate = next(
            (t["text"] for t in reversed(point.history) if t["speaker"] == "candidate"),
            "",
        )
        rows.append(
            {
                "conversation_id": point.conversation_id,
                "turn_id": point.turn_id,
                "last_candidate_message": last_candidate,
                "true_label": point.label,
                "predicted_label": predicted,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "conversation_id",
            "turn_id",
            "last_candidate_message",
            "true_label",
            "predicted_label",
        ],
    )
