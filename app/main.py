"""CLI entry point — a terminal chat loop against the recruiting bot.

The lightweight way to exercise the agents without starting Streamlit::

    python -m app.main
    python -m app.main --start 2026-07-15T15:12:00Z   # replay a dataset conversation

``--start`` pins the conversation date, which is what relative expressions like
"next Friday" resolve against. Leave it off to talk to the bot as of today; pass
it to reproduce a conversation from the dataset (CLAUDE.md 6.3).

STATUS: scaffolding. The argument parsing and the loop shape are final; the turn
handling is not implemented yet.
"""

from __future__ import annotations

import argparse

from app.config import settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="app.main",
        description="Chat with the recruiting bot in the terminal.",
    )
    parser.add_argument(
        "--session",
        default="cli",
        help="Session id used to key the conversation memory.",
    )
    parser.add_argument(
        "--start",
        default=None,
        metavar="ISO8601",
        help=(
            "Conversation start time, e.g. 2026-07-15T15:12:00Z. Relative dates "
            "resolve against this. Defaults to now."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print the resolved configuration and exit, without calling the API.",
    )
    return parser.parse_args(argv)


def run_chat(session_id: str, conversation_start: str | None) -> None:
    """Run the interactive loop until the bot ends the conversation.

    Reads a candidate message, calls
    :func:`~app.modules.main_agent.orchestrator.handle_turn`, prints the reply,
    and stops once the decided action is ``end``.
    """
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)

    if args.check:
        print(settings)
        return 0

    settings.require_api_key()
    run_chat(args.session, args.start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
