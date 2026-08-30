"""GenAI recruiting chatbot — application package.

An SMS-style multi-agent chatbot that converses with candidates for a Python
Developer position and either schedules an interview or politely closes the
conversation.

Layout::

    app.config      Environment, model ids and paths. Single source of truth.
    app.main        CLI entry point (terminal chat loop).
    app.modules     Main agent, advisors, database, embedding, fine-tuning, evaluation.
    app.prompts     System prompts as .txt files.

Importing this package must never touch the network or rebuild an index. Offline
work lives in ``scripts/``. See ENGINEERING.md section 2.
"""

__version__ = "0.1.0"
