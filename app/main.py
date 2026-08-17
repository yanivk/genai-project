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


GREETING = (
    "Hi, thanks for submitting your application for our Python Developer role. "
    "Could you share a bit about your Python experience?"
)


def run_chat(session_id: str, conversation_start: str | None) -> None:
    """Run the interactive loop until the bot ends the conversation.

    Reads a candidate message, calls
    :func:`~app.modules.main_agent.orchestrator.handle_turn`, prints the reply,
    and stops once the decided action is ``end``.
    """
    from app.modules.main_agent.orchestrator import get_history, handle_turn

    print(f"\nRecruiter: {GREETING}")
    print("\n(type 'quit' to leave, 'why' to see the last advisor verdicts)\n")

    # Seed the bot's own greeting so the first turn has context.
    get_history(session_id).add_ai_message(GREETING)
    last: object = None

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not message:
            continue
        if message.lower() in {"quit", "exit"}:
            print("Bye.")
            return
        if message.lower() == "why":
            from app.modules.main_agent.orchestrator import describe_turn

            print(describe_turn(last) if last else "No turn yet.\n")
            continue

        last = handle_turn(session_id, message, conversation_start)
        print(f"\nRecruiter: {last.message}")
        print(f"   [{last.action}] {last.reason}\n")

        if last.action == "end":
            print("-- conversation closed --")
            return


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
