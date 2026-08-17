"""Unit tests for the application scaffolding.

These tests cover what exists today: configuration resolution, the action
literals, and the shape of the labeled dataset. They run offline — no test here
may call the OpenAI API.

As modules get implemented, add tests next to the relevant class below.
"""

from __future__ import annotations

import json

import pytest

from app.config import ROOT_DIR, settings
from app.modules.main_agent.actions import (
    ACTIONS,
    CONTINUE,
    END,
    SCHEDULE,
    is_valid_action,
    parse_action,
)


class TestActions:
    """The three action literals."""

    def test_canonical_set(self):
        assert ACTIONS == (CONTINUE, SCHEDULE, END)
        assert ACTIONS == ("continue", "schedule", "end")

    def test_validation(self):
        assert is_valid_action("schedule")
        assert not is_valid_action("Schedule")
        assert not is_valid_action("stop")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("continue", CONTINUE), ("  SCHEDULE ", SCHEDULE), ("End", END)],
    )
    def test_parse_normalises(self, raw, expected):
        assert parse_action(raw) == expected

    def test_parse_rejects_unknown(self):
        with pytest.raises(ValueError, match="Invalid action"):
            parse_action("hang_up")


class TestConfig:
    """Configuration resolution."""

    def test_paths_are_absolute(self):
        assert settings.root_dir.is_absolute()
        assert settings.chroma_path.is_absolute()
        assert settings.prompts_dir.is_absolute()

    def test_root_dir_holds_the_project(self):
        assert (ROOT_DIR / "app").is_dir()
        assert (ROOT_DIR / "CLAUDE.md").is_file()

    def test_sqlite_url_is_absolute(self):
        # A relative sqlite URL resolves against the working directory, which
        # breaks as soon as the app starts from anywhere but the repo root.
        if settings.db_url.startswith("sqlite:///"):
            assert "/data/tech.db" in settings.db_url.replace("\\", "/")

    def test_exit_advisor_falls_back(self):
        # The app must never hard-fail because fine-tuning has not run yet.
        assert settings.exit_advisor_model
        if not settings.is_finetuned:
            assert settings.exit_advisor_model == settings.openai_model

    def test_str_never_leaks_the_key(self):
        assert settings.openai_api_key not in str(settings) or not settings.openai_api_key

    def test_prompt_path_accepts_both_forms(self):
        assert settings.prompt_path("exit_advisor") == settings.prompt_path("exit_advisor.txt")


class TestPrompts:
    """Prompt files follow the Course20 layout."""

    NAMES = ["main_agent", "exit_advisor", "scheduling_advisor", "info_advisor"]

    @pytest.mark.parametrize("name", NAMES)
    def test_exists(self, name):
        assert settings.prompt_path(name).is_file()

    @pytest.mark.parametrize("name", NAMES)
    def test_section_order(self, name):
        text = settings.prompt_path(name).read_text(encoding="utf-8")
        positions = [text.index(h) for h in ("# Identity", "# Instructions", "# Examples", "# Context")]
        # Context goes last: it changes per request, the rest is static.
        assert positions == sorted(positions)


class TestDataset:
    """Shape and labels of data/sms_conversations.json."""

    @pytest.fixture(scope="class")
    def conversations(self):
        return json.loads(settings.conversations_json.read_text(encoding="utf-8"))

    def test_size(self, conversations):
        assert len(conversations) == 15
        assert sum(len(c["turns"]) for c in conversations) == 103

    def test_only_recruiter_turns_are_labeled(self, conversations):
        for conversation in conversations:
            for turn in conversation["turns"]:
                if turn["speaker"] == "candidate":
                    assert turn["label"] is None
                else:
                    assert turn["label"] in ACTIONS

    def test_label_distribution(self, conversations):
        counts = {action: 0 for action in ACTIONS}
        for conversation in conversations:
            for turn in conversation["turns"]:
                if turn["label"]:
                    counts[turn["label"]] += 1
        assert counts == {CONTINUE: 25, SCHEDULE: 19, END: 15}

    def test_end_is_terminal(self, conversations):
        # `end` is the last recruiter turn of every conversation, covering both
        # confirmed bookings and opt-outs. See CLAUDE.md section 6.1.
        for conversation in conversations:
            labeled = [t for t in conversation["turns"] if t["label"]]
            assert labeled[-1]["label"] == END
            assert all(t["label"] != END for t in labeled[:-1])

    def test_conversations_fall_inside_the_seeded_year(self, conversations):
        # Relative dates anchor on start_time_utc, and the schedule only covers
        # 2024. A conversation outside that range would find no slots.
        for conversation in conversations:
            assert conversation["start_time_utc"].startswith("2024-")
