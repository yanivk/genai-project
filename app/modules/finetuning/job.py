"""Launch and monitor the Exit Advisor fine-tuning job.

Thin wrapper over the OpenAI SDK: upload the JSONL with ``purpose="fine-tune"``,
create the job on ``settings.ft_base_model``, poll until it finishes, and print
the resulting model id.

Fine-tuning jobs cost money and take time. Nothing here runs implicitly — the
only caller is ``scripts/run_finetuning.py``, invoked by hand.

Once the job succeeds, write the model id into ``.env`` as
``FT_EXIT_ADVISOR_MODEL``. Until then the Exit Advisor keeps running on the base
model with few-shot prompting (CLAUDE.md 11.7).

STATUS: scaffolding. Signatures are final; bodies are not implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    raise NotImplementedError


def create_job(training_file_id: str, suffix: str = "exit-advisor") -> str:
    """Create a fine-tuning job on ``settings.ft_base_model``.

    Args:
        training_file_id: Id returned by :func:`upload_training_file`.
        suffix: Tag appended to the resulting model id, to keep runs apart.

    Returns:
        The job id.
    """
    raise NotImplementedError


def get_status(job_id: str) -> JobStatus:
    """Fetch the current status of a job."""
    raise NotImplementedError


def wait_for_completion(job_id: str, poll_seconds: int = 30) -> JobStatus:
    """Poll a job until it reaches a terminal state.

    Args:
        job_id: Id returned by :func:`create_job`.
        poll_seconds: Delay between polls.

    Returns:
        The terminal status. On success, ``.model`` holds the id to put into
        ``FT_EXIT_ADVISOR_MODEL``.
    """
    raise NotImplementedError
