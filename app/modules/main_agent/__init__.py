"""Main Agent — orchestrates one conversation turn.

Receives the candidate's message, consults the three advisors, and decides the
action (``continue`` / ``schedule`` / ``end``) plus the message to send back.
This is the only component that owns conversation memory; the advisors are
stateless.

Modules:
    actions      The three action literals. Import these, never the strings.
    orchestrator The turn router and the agent executor.
"""

from app.modules.main_agent.actions import ACTIONS, CONTINUE, END, SCHEDULE, Action

__all__ = ["ACTIONS", "CONTINUE", "END", "SCHEDULE", "Action"]
