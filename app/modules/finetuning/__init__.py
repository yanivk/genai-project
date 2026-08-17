"""Fine-tuning pipeline for the Conversation Exit Advisor.

    dataset  Build the training JSONL from the labeled conversations.
    job      Upload the file, launch the OpenAI job, poll its status.

No course precedent — this module is written against the OpenAI SDK directly.

The one rule that matters: **train on the training split only**. The held-out
conversations from the evaluation split must never enter the JSONL, or the
reported accuracy is meaningless. See CLAUDE.md section 11.
"""
