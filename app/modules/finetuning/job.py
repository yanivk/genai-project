"""Launch and monitor the Exit Advisor fine-tuning job.

Thin wrapper over the OpenAI SDK: upload the JSONL with ``purpose="fine-tune"``,
create the job on ``settings.ft_base_model``, poll until it finishes, and print
the resulting model id.

Fine-tuning jobs cost money and take time. Nothing here runs implicitly — the
only caller is ``scripts/run_finetuning.py``, invoked by hand.

Once the job succeeds, write the model id into ``.env`` as
``FT_EXIT_ADVISOR_MODEL``. Until then the Exit Advisor keeps running on the base
model with few-shot prompting (CLAUDE.md 11.7).

**OpenAI closed self-serve fine-tuning on 2026-05-07.** Organizations that had
never run a job before that date can no longer create one: the API answers 403
``training_not_available``. Every remaining organization loses the ability by
2027-01-06. Inference on already-trained models keeps working until the base
model is retired.

This project's organization is on the wrong side of that date, so
:func:`create_job` is unreachable in practice. The code stays — it is the
requested deliverable and it is exercised as far as the API allows (the upload in
:func:`upload_training_file` succeeds) — and the failure is translated into
:class:`FineTuningUnavailable` so the script explains itself instead of printing a
stack trace. See CLAUDE.md 11.9.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from openai import PermissionDeniedError

from app.config import settings
from app.modules.common import get_openai_client

logger = logging.getLogger(__name__)

#: OpenAI's error code when the organization may not create fine-tuning jobs.
_PLATFORM_CLOSED_CODE = "training_not_available"


class FineTuningUnavailable(RuntimeError):
    """OpenAI refused to create the job because the platform is closed.

    Distinct from a bad request or a missing key: nothing about the JSONL, the
    split or the base model would change the outcome.
    """


@dataclass(frozen=True)
class JobStatus:
    """Snapshot of a fine-tuning job.

    Attributes:
        job_id: The OpenAI job id.
        status: ``validating_files`` / ``queued`` / ``running`` / ``succeeded`` /
            ``failed`` / ``cancelled``.
        model: The resulting model id, set once the job succeeds.
        error: Failure message when the job failed.
    """

    job_id: str
    status: str
    model: str | None = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        """True when the job will not change state again."""
        return self.status in {"succeeded", "failed", "cancelled"}


def upload_training_file(jsonl_path: Path) -> str:
    """Upload a JSONL file for fine-tuning.

    Args:
        jsonl_path: The file from
            :func:`~app.modules.finetuning.dataset.write_jsonl`.

    Returns:
        The uploaded file id.
    """
    path = Path(jsonl_path)
    with path.open("rb") as handle:
        uploaded = get_openai_client().files.create(file=handle, purpose="fine-tune")
    logger.info("Uploaded %s as %s", path.name, uploaded.id)
    return uploaded.id


def create_job(training_file_id: str, suffix: str = "exit-advisor") -> str:
    """Create a fine-tuning job on ``settings.ft_base_model``.

    Args:
        training_file_id: Id returned by :func:`upload_training_file`.
        suffix: Tag appended to the resulting model id, to keep runs apart.

    Returns:
        The job id.

    Raises:
        FineTuningUnavailable: When OpenAI refuses because the self-serve
            fine-tuning platform is closed to this organization. Retrying, or
            changing the data or the base model, will not help.
    """
    try:
        job = get_openai_client().fine_tuning.jobs.create(
            training_file=training_file_id,
            model=settings.ft_base_model,
            suffix=suffix,
        )
    except PermissionDeniedError as exc:
        if _is_platform_closed(exc):
            raise FineTuningUnavailable(str(exc)) from exc
        raise
    logger.info("Created fine-tuning job %s on %s", job.id, settings.ft_base_model)
    return job.id


def _is_platform_closed(exc: PermissionDeniedError) -> bool:
    """True when a 403 is the platform wind-down rather than a key problem."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("code") == _PLATFORM_CLOSED_CODE:
            return True
    # The SDK does not always populate `body`; the message is the fallback.
    return _PLATFORM_CLOSED_CODE in str(exc)


def get_status(job_id: str) -> JobStatus:
    """Fetch the current status of a job."""
    job = get_openai_client().fine_tuning.jobs.retrieve(job_id)
    error = getattr(job, "error", None)
    # The SDK returns an error object whose fields are all None while the job is
    # healthy, so only surface a message when one is actually set.
    message = getattr(error, "message", None) if error else None
    return JobStatus(
        job_id=job.id,
        status=job.status,
        model=job.fine_tuned_model,
        error=message,
    )


def wait_for_completion(job_id: str, poll_seconds: int = 30) -> JobStatus:
    """Poll a job until it reaches a terminal state.

    Args:
        job_id: Id returned by :func:`create_job`.
        poll_seconds: Delay between polls.

    Returns:
        The terminal status. On success, ``.model`` holds the id to put into
        ``FT_EXIT_ADVISOR_MODEL``.
    """
    while True:
        status = get_status(job_id)
        if status.is_terminal:
            return status
        logger.info("Job %s: %s", job_id, status.status)
        print(f"  {status.status}...", flush=True)
        time.sleep(poll_seconds)
