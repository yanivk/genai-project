"""Fine-tune the Conversation Exit Advisor.

Builds the training JSONL from the TRAINING split only, uploads it, launches the
OpenAI job and polls until it finishes::

    python scripts/run_finetuning.py --dry-run   # build and inspect, spend nothing
    python scripts/run_finetuning.py             # build, upload, launch, wait

This costs money. Nothing here runs implicitly — it is a manual command.

When the job succeeds, copy the printed model id into .env as
FT_EXIT_ADVISOR_MODEL. Until you do, the Exit Advisor keeps running on the base
model with few-shot prompting.

The training split must match the one the evaluation notebook holds out, or the
reported accuracy is meaningless. Both come from
``app.modules.evaluation.dataset.split_by_conversation`` with the same seed.

STATUS: scaffolding. The CLI is final; the pipeline steps are not implemented yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.modules.evaluation.dataset import load_conversations, split_by_conversation  # noqa: E402
from app.modules.finetuning.dataset import build_examples, describe, write_jsonl  # noqa: E402
from app.modules.finetuning.job import create_job, upload_training_file, wait_for_completion  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and describe the JSONL, then stop. Spends nothing.",
    )
    parser.add_argument("--test-size", type=int, default=5, help="Conversations held out.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed. Must match the notebook.")
    args = parser.parse_args(argv)

    conversations = load_conversations()
    train_ids, test_ids = split_by_conversation(
        conversations, test_size=args.test_size, seed=args.seed
    )
    print(f"Train conversations: {sorted(train_ids)}")
    print(f"Held out (never trained on): {sorted(test_ids)}")

    examples = build_examples(train_ids)
    print(describe(examples))

    jsonl_path = write_jsonl(examples, settings.data_dir / "exit_advisor_train.jsonl")
    print(f"Wrote {jsonl_path}")

    if args.dry_run:
        print("Dry run — stopping before upload.")
        return 0

    settings.require_api_key()

    file_id = upload_training_file(jsonl_path)
    print(f"Uploaded file: {file_id}")

    job_id = create_job(file_id)
    print(f"Created job: {job_id} on base {settings.ft_base_model}")

    status = wait_for_completion(job_id)
    if status.status != "succeeded":
        print(f"Job {status.status}: {status.error}")
        return 1

    print(f"\nSuccess. Add this to your .env:\n\nFT_EXIT_ADVISOR_MODEL={status.model}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
