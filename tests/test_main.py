"""Unit tests for the application.

Everything here runs **offline** — no test may call the OpenAI API. That keeps the
suite fast and free, so it can run on every change. The parts that need a model
(the advisors, the Main Agent) are exercised by ``tests/test_evals.ipynb``.

Covered: configuration resolution, the action literals and routing precedence,
the dataset invariants, date arithmetic, chunking, JSON recovery, and the seeded
database.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pytest
from sqlalchemy import text as sa_text

from app.config import ROOT_DIR, settings
from app.modules.advisors.scheduling_advisor import resolve_relative_date
from app.modules.advisors.schemas import ExitVerdict, SchedulingVerdict
from app.modules.common import parse_json_output, render_history
from app.modules.database.engine import (
    get_available_slots,
    get_engine,
    is_slot_available,
)
from app.modules.database.seeder import END_DATE, START_DATE
from app.modules.embedding.indexer import chunk_text
from app.modules.evaluation.dataset import (
    build_decision_points,
    ending_flavour,
    load_conversations,
    split_by_conversation,
)
from app.modules.main_agent.actions import (
    ACTIONS,
    CONTINUE,
    END,
    SCHEDULE,
    is_valid_action,
    parse_action,
)
from app.modules.main_agent.orchestrator import resolve_action


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

    def test_conversations_fall_inside_the_seeded_range(self, conversations):
        # Invariant 1 of CLAUDE.md 6.3: relative dates anchor on start_time_utc
        # and are looked up in the Schedule table, so a conversation outside the
        # seeded range would silently find no slots.
        for conversation in conversations:
            start = dt.datetime.fromisoformat(
                conversation["start_time_utc"].replace("Z", "+00:00")
            ).date()
            assert START_DATE <= start <= END_DATE, (
                f"conversation {conversation['conversation_id']} starts {start}, "
                f"outside the seeded range {START_DATE}..{END_DATE}"
            )

    def test_no_message_proposes_an_unavailable_weekday(self, conversations):
        # Invariant 2 of CLAUDE.md 6.3: the schedule has no Monday or Saturday
        # rows, so no message may propose one.
        excluded = {"Monday", "Saturday"}
        offenders = [
            (c["conversation_id"], t["turn_id"], day)
            for c in conversations
            for t in c["turns"]
            for day in re.findall(r"\b(\w+day)\b", t["text"])
            if day in excluded
        ]
        assert not offenders, f"unavailable weekdays proposed: {offenders}"

    def test_timestamps_are_ordered_and_anchored(self, conversations):
        for conversation in conversations:
            stamps = [
                dt.datetime.fromisoformat(t["timestamp_utc"].replace("Z", "+00:00"))
                for t in conversation["turns"]
            ]
            assert stamps == sorted(stamps)
            assert stamps[0].isoformat() == dt.datetime.fromisoformat(
                conversation["start_time_utc"].replace("Z", "+00:00")
            ).isoformat()


class TestDecisionPoints:
    """Expanding conversations into evaluable prediction points."""

    @pytest.fixture(scope="class")
    def points(self):
        return build_decision_points(load_conversations())

    def test_one_point_per_labeled_recruiter_turn(self, points):
        assert len(points) == 59

    def test_history_stops_before_the_predicted_turn(self, points):
        # The point must not contain the turn it is predicting, or the task is
        # trivially solved by reading the answer.
        for point in points:
            assert len(point.history) == point.turn_id - 1

    def test_history_ends_on_a_candidate_turn_except_the_opener(self, points):
        for point in points:
            if point.history:
                assert point.history[-1]["speaker"] == "candidate"

    def test_labels_are_canonical(self, points):
        assert {p.label for p in points} == set(ACTIONS)


class TestSplit:
    """Conversation-level splitting, stratified by ending flavour."""

    @pytest.fixture(scope="class")
    def conversations(self):
        return load_conversations()

    def test_no_overlap_and_full_coverage(self, conversations):
        train, test = split_by_conversation(conversations, test_size=5, seed=42)
        assert not set(train) & set(test)
        assert len(train) + len(test) == len(conversations)
        assert len(test) == 5

    def test_both_ending_flavours_survive_the_split(self, conversations):
        # With only 4 opt-outs, an unstratified split can leave the test set with
        # none — and then the `end` class is untestable (CLAUDE.md 10.1).
        train, test = split_by_conversation(conversations, test_size=5, seed=42)
        by_id = {c["conversation_id"]: c for c in conversations}
        for ids in (train, test):
            assert "opt-out" in {ending_flavour(by_id[i]) for i in ids}
            assert "booked" in {ending_flavour(by_id[i]) for i in ids}

    def test_split_is_reproducible(self, conversations):
        # The fine-tuning script and the notebook must agree on the split, or the
        # held-out set leaks into training.
        first = split_by_conversation(conversations, test_size=5, seed=42)
        second = split_by_conversation(conversations, test_size=5, seed=42)
        assert first == second


class TestRouting:
    """The deterministic action precedence: exit > schedule > continue."""

    @staticmethod
    def _verdicts(should_end: bool, should_schedule: bool):
        return (
            ExitVerdict(should_end=should_end, reason=""),
            SchedulingVerdict(should_schedule=should_schedule, slots=[], reason=""),
        )

    @pytest.mark.parametrize(
        ("should_end", "should_schedule", "expected"),
        [
            (False, False, CONTINUE),
            (False, True, SCHEDULE),
            (True, False, END),
            # The case that mattered: both fire when a candidate has just accepted
            # a time, and `end` must win.
            (True, True, END),
        ],
    )
    def test_precedence(self, should_end, should_schedule, expected):
        exit_verdict, scheduling_verdict = self._verdicts(should_end, should_schedule)
        assert resolve_action(exit_verdict, scheduling_verdict) == expected


class TestRelativeDates:
    """Date arithmetic anchored on the conversation, not on today."""

    #: A Monday, so every weekday offset is easy to reason about.
    ANCHOR = "2026-08-17T10:00:00Z"

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("today", "2026-08-17"),
            ("tomorrow", "2026-08-18"),
            ("next Tuesday", "2026-08-18"),
            ("Friday", "2026-08-21"),
            ("next Sunday", "2026-08-23"),
            ("next week", "2026-08-24"),
            ("", "2026-08-17"),
            ("whenever suits you", "2026-08-17"),
        ],
    )
    def test_resolution(self, expression, expected):
        assert resolve_relative_date(expression, self.ANCHOR) == expected

    def test_same_weekday_means_next_week(self):
        # "Monday" said on a Monday is the coming Monday, not today.
        assert resolve_relative_date("Monday", self.ANCHOR) == "2026-08-24"

    def test_anchors_on_the_conversation_not_today(self):
        # The whole point of the anchor: a 2026-07 conversation must not resolve
        # against the current date.
        assert resolve_relative_date("Friday", "2026-07-15T12:00:00Z") == "2026-07-17"


class TestSeededDatabase:
    """The generated data/tech.db, if it has been built."""

    @pytest.fixture(scope="class")
    def engine(self):
        db_file = Path(settings.db_url.removeprefix("sqlite:///"))
        if not db_file.is_file():
            pytest.skip("data/tech.db not built; run python scripts/seed_database.py")
        return get_engine()

    def test_only_available_weekdays_exist(self, engine):
        with engine.connect() as conn:
            dates = [d for (d,) in conn.execute(sa_text("SELECT DISTINCT date FROM Schedule"))]
        weekdays = {dt.date.fromisoformat(d).weekday() for d in dates}
        assert not weekdays & {0, 5}, "Monday or Saturday slots leaked into the schedule"

    def test_hours_are_within_business_range(self, engine):
        with engine.connect() as conn:
            times = [t for (t,) in conn.execute(sa_text("SELECT DISTINCT time FROM Schedule"))]
        hours = {int(t[:2]) for t in times}
        assert min(hours) == 9 and max(hours) == 17

    def test_positions_match_the_original_script(self, engine):
        with engine.connect() as conn:
            positions = {p for (p,) in conn.execute(sa_text("SELECT DISTINCT position FROM Schedule"))}
        assert positions == {"Python Dev", "Sql Dev", "Analyst", "ML"}

    def test_queries_return_only_available_slots(self, engine):
        frame = get_available_slots("2026-08-17", limit=5)
        assert len(frame) <= 5
        for _, row in frame.iterrows():
            assert row["date"] >= "2026-08-17"
            assert is_slot_available(row["date"], row["time"])


class TestJsonRecovery:
    """Tolerating the ways models wrap their JSON."""

    @pytest.mark.parametrize(
        "raw",
        [
            '{"should_end": true, "reason": "done"}',
            '```json\n{"should_end": true, "reason": "done"}\n```',
            '```\n{"should_end": true, "reason": "done"}\n```',
            'Here is my verdict: {"should_end": true, "reason": "done"}',
        ],
    )
    def test_recovers_the_object(self, raw):
        assert parse_json_output(raw)["should_end"] is True

    def test_raises_when_there_is_no_json(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            parse_json_output("I could not decide.")


class TestChunking:
    """PDF chunking for the vector store."""

    def test_respects_the_size_budget(self):
        text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        assert chunks
        # Allow a little slack: a paragraph is only split when it alone exceeds
        # the budget, so a chunk can end slightly over.
        assert all(len(c) <= 700 for c in chunks)

    def test_splits_a_single_oversized_paragraph(self):
        chunks = chunk_text("word " * 500, chunk_size=300, overlap=50)
        assert len(chunks) > 1

    def test_keeps_short_text_in_one_chunk(self):
        assert chunk_text("A short job description.", chunk_size=1000) == [
            "A short job description."
        ]

    def test_loses_no_content(self):
        text = "\n\n".join(f"Requirement {i} is important." for i in range(10))
        joined = " ".join(chunk_text(text, chunk_size=120, overlap=20))
        for i in range(10):
            assert f"Requirement {i}" in joined


class TestHistoryRendering:
    """The plain-text format shared by the advisors and the fine-tuning data."""

    def test_labels_speakers(self):
        rendered = render_history(
            [
                {"speaker": "recruiter", "text": "Hi there."},
                {"speaker": "candidate", "text": "Hello."},
            ]
        )
        assert rendered == "Recruiter: Hi there.\nCandidate: Hello."

    def test_empty_history_is_empty_string(self):
        assert render_history([]) == ""
