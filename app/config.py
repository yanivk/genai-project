"""Central configuration — the single source of truth for this project.

No other module calls ``os.getenv``, hardcodes a model name, or builds a path by
hand. Everything comes from here::

    from app.config import settings

    llm = ChatOpenAI(model=settings.openai_model, temperature=0)

Resolution order for every value is ``st.secrets`` -> environment (``.env``) ->
default. That single rule is what lets identical code run locally and on
Streamlit Community Cloud, where there is no ``.env`` file and secrets come from
the app's Secrets UI.

Paths are absolute and derived from the repository root, so behaviour does not
depend on the current working directory.

Importing this module must never touch the network. See CLAUDE.md section 7.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Repository root: app/config.py -> app/ -> repo root.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Chroma phones home on startup and logs a stack trace when the call fails, which
# buries real output. Opt out before chromadb is imported anywhere, and silence the
# logger too: in chromadb 1.0.x the flag alone does not stop every event.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

load_dotenv(ROOT_DIR / ".env")


def _from_secrets(key: str) -> str | None:
    """Read ``key`` from ``st.secrets``, or return None when unavailable.

    Streamlit is an optional runtime here: the CLI, the scripts and the tests all
    run without it. Accessing ``st.secrets`` outside a Streamlit process raises,
    so every failure mode is swallowed and treated as "not configured".
    """
    try:
        import streamlit as st  # noqa: PLC0415 - optional, only present in the UI process
    except ImportError:
        return None
    try:
        value = st.secrets[key]
    except Exception:  # noqa: BLE001 - no secrets file, not running in Streamlit, ...
        return None
    return str(value) if value is not None else None


def get_setting(key: str, default: str = "") -> str:
    """Return a configuration value: ``st.secrets`` -> environment -> ``default``."""
    value = _from_secrets(key)
    if value is None or value == "":
        value = os.getenv(key, "")
    return value if value != "" else default


def _resolve(path_value: str) -> Path:
    """Make a configured path absolute, relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else (ROOT_DIR / path)


def _resolve_db_url(url: str) -> str:
    """Turn a relative SQLite URL into an absolute one.

    ``sqlite:///data/tech.db`` resolves against the process working directory,
    which breaks as soon as the app is started from anywhere but the repo root.
    Non-SQLite URLs are passed through untouched.
    """
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url
    raw = url[len(prefix):]
    if Path(raw).is_absolute():
        return url
    return f"{prefix}{(ROOT_DIR / raw).as_posix()}"


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for the whole application.

    Attributes mirror the keys documented in ``.env.example`` and CLAUDE.md
    section 7.
    """

    # --- Paths -----------------------------------------------------------
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    prompts_dir: Path = ROOT_DIR / "app" / "prompts"
    conversations_json: Path = ROOT_DIR / "data" / "sms_conversations.json"
    job_description_pdf: Path = ROOT_DIR / "data" / "Python Developer Job Description.pdf"
    source_sql: Path = ROOT_DIR / "data" / "db_Tech.sql"
    #: The committed output of the offline embedding step: chunks + their vectors.
    #: Around 20 KB, versus the 60 MB Chroma writes for the same three chunks.
    vector_store_json: Path = ROOT_DIR / "data" / "vector_store.json"

    # --- OpenAI ----------------------------------------------------------
    openai_api_key: str = field(default_factory=lambda: get_setting("OPENAI_API_KEY"))
    openai_model: str = field(
        default_factory=lambda: get_setting("OPENAI_MODEL", "gpt-4o-2024-11-20")
    )
    embedding_model: str = field(
        default_factory=lambda: get_setting("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # --- Fine-tuning -----------------------------------------------------
    ft_base_model: str = field(
        default_factory=lambda: get_setting("FT_BASE_MODEL", "gpt-4o-mini-2024-07-18")
    )
    #: Empty until the fine-tuning job completes. The Exit Advisor falls back to
    #: ``openai_model`` with few-shot prompting while this is unset.
    ft_exit_advisor_model: str = field(
        default_factory=lambda: get_setting("FT_EXIT_ADVISOR_MODEL")
    )

    # --- Database --------------------------------------------------------
    db_url: str = field(
        default_factory=lambda: _resolve_db_url(get_setting("DB_URL", "sqlite:///data/tech.db"))
    )

    # --- Vector store ----------------------------------------------------
    chroma_path: Path = field(
        default_factory=lambda: _resolve(get_setting("CHROMA_PATH", "data/chroma"))
    )
    chroma_collection: str = field(
        default_factory=lambda: get_setting("CHROMA_COLLECTION", "python_dev_job")
    )

    # --- Orchestration ---------------------------------------------------
    max_advisor_rounds: int = field(
        default_factory=lambda: int(get_setting("MAX_ADVISOR_ROUNDS", "2"))
    )

    @property
    def exit_advisor_model(self) -> str:
        """Model the Exit Advisor should use — fine-tuned when available.

        Falls back to :attr:`openai_model` so the app never hard-fails because a
        fine-tuning job is missing, expired or still running (CLAUDE.md 11.7).
        """
        return self.ft_exit_advisor_model or self.openai_model

    @property
    def is_finetuned(self) -> bool:
        """True when a fine-tuned Exit Advisor model is configured."""
        return bool(self.ft_exit_advisor_model)

    def require_api_key(self) -> str:
        """Return the OpenAI API key, raising a readable error when it is missing.

        Call this at the edge of any code path that is about to hit the API,
        rather than letting the OpenAI SDK fail with a less obvious message.
        """
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in, "
                "or set it in the Streamlit Secrets UI when deploying."
            )
        return self.openai_api_key

    def prompt_path(self, name: str) -> Path:
        """Return the path to a prompt file in ``app/prompts``.

        Args:
            name: Prompt file name, with or without the ``.txt`` suffix.
        """
        filename = name if name.endswith(".txt") else f"{name}.txt"
        return self.prompts_dir / filename

    def __str__(self) -> str:  # pragma: no cover - diagnostic helper
        """Readable summary that never reveals the API key."""
        key_state = "set" if self.openai_api_key else "MISSING"
        return (
            "Settings(\n"
            f"  root_dir            = {self.root_dir}\n"
            f"  openai_api_key      = <{key_state}>\n"
            f"  openai_model        = {self.openai_model}\n"
            f"  embedding_model     = {self.embedding_model}\n"
            f"  exit_advisor_model  = {self.exit_advisor_model}"
            f"{'  (fine-tuned)' if self.is_finetuned else '  (fallback: not fine-tuned yet)'}\n"
            f"  db_url              = {self.db_url}\n"
            f"  chroma_path         = {self.chroma_path}\n"
            f"  chroma_collection   = {self.chroma_collection}\n"
            f"  max_advisor_rounds  = {self.max_advisor_rounds}\n"
            ")"
        )


#: Import this, not the class.
settings = Settings()
