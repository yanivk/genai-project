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
import importlib
import json
import re
import sys
from pathlib import Path

import httpx
import pytest
from openai import PermissionDeniedError
from sqlalchemy import text as sa_text

from app.config import ROOT_DIR, settings
from app.modules.advisors.exit_advisor import (
    DIRECTIVE,
    PROMPT_FINETUNED,
    build_system_text,
)
from app.modules.advisors.scheduling_advisor import (
    TIME_OF_DAY,
    _time_of_day_key,
    describe_window,
    reconcile,
    resolve_relative_date,
    validate_slots,
)
from app.modules.advisors.scheduling_advisor import get_available_slots as slots_tool
from app.modules.advisors.schemas import ExitVerdict, InfoVerdict, SchedulingVerdict
from app.modules.common import parse_json_output, render_history
from app.modules.database.engine import (
    get_availability_calendar,
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
from app.modules.finetuning.dataset import (
    MIN_EXAMPLES,
    build_examples,
    describe,
    write_jsonl,
)
from app.modules.finetuning.job import _is_platform_closed
from app.modules.main_agent.actions import (
    ACTIONS,
    CONTINUE,
    END,
    SCHEDULE,
    is_valid_action,
    parse_action,
)
from app.modules.main_agent.orchestrator import _decision_from, resolve_action


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
        assert (ROOT_DIR / "ENGINEERING.md").is_file()

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

    @pytest.mark.parametrize("name", ["scheduling_advisor", "main_agent"])
    def test_availability_first_is_documented(self, name):
        # The two prompts that read `needs_availability` must both name it, or one
        # of them silently reverts to handing out dates the candidate never asked
        # for — a drift no schema check would catch.
        text = settings.prompt_path(name).read_text(encoding="utf-8")
        assert "needs_availability" in text

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
        # confirmed bookings and opt-outs. See ENGINEERING.md section 6.1.
        for conversation in conversations:
            labeled = [t for t in conversation["turns"] if t["label"]]
            assert labeled[-1]["label"] == END
            assert all(t["label"] != END for t in labeled[:-1])

    def test_conversations_fall_inside_the_seeded_range(self, conversations):
        # Invariant 1 of ENGINEERING.md 6.3: relative dates anchor on start_time_utc
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
        # Invariant 2 of ENGINEERING.md 6.3: the schedule has no Monday or Saturday
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
        # none — and then the `end` class is untestable (ENGINEERING.md 10.1).
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
            # "tomorrow" is a substring of this, so the order of the checks
            # decides the answer. Matched first, it silently returns the 18th.
            ("day after tomorrow", "2026-08-19"),
            # An explicit date passes through: this is how the advisor searches
            # past slots the candidate already refused.
            ("2026-09-03", "2026-09-03"),
        ],
    )
    def test_resolution(self, expression, expected):
        assert resolve_relative_date(expression, self.ANCHOR) == expected

    def test_explicit_date_beats_a_weekday_name_inside_it(self):
        # Guards the precedence, not the arithmetic: a date must never be
        # re-derived from a stray weekday word around it.
        assert resolve_relative_date("2026-09-03", self.ANCHOR) == "2026-09-03"

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


class TestAvailabilityCalendar:
    """The windowed query behind the sidebar calendar view."""

    FROM_DATE = "2026-08-17"
    DAYS = 28

    @pytest.fixture(scope="class")
    def calendar(self):
        db_file = Path(settings.db_url.removeprefix("sqlite:///"))
        if not db_file.is_file():
            pytest.skip("data/tech.db not built; run python scripts/seed_database.py")
        return get_availability_calendar(self.FROM_DATE, days=self.DAYS)

    def test_window_is_half_open(self, calendar):
        # Inclusive on the left, exclusive on the right, so consecutive windows
        # tile the schedule without overlapping.
        end = (dt.date.fromisoformat(self.FROM_DATE) + dt.timedelta(days=self.DAYS)).isoformat()
        assert not calendar.empty
        assert calendar["date"].min() >= self.FROM_DATE
        assert calendar["date"].max() < end

    def test_columns_and_chronological_order(self, calendar):
        assert list(calendar.columns) == ["date", "time", "available"]
        assert set(calendar["available"].unique()) <= {0, 1}
        pairs = list(zip(calendar["date"], calendar["time"]))
        assert pairs == sorted(pairs)

    def test_keeps_the_taken_slots_too(self, calendar):
        # The point of the calendar: a booked slot is a rendered cell, not a
        # missing row. Only get_available_slots filters on `available`.
        assert 0 in set(calendar["available"])

    def test_accepts_a_full_iso_timestamp(self, calendar):
        # Callers pass the conversation anchor (start_time_utc) straight through.
        stamped = get_availability_calendar(f"{self.FROM_DATE}T17:04:22Z", days=self.DAYS)
        assert stamped.equals(calendar)

    def test_no_unavailable_weekday_appears(self, calendar):
        # Invariant 2 of ENGINEERING.md 6.3, seen from the query side: a day with no
        # rows is normal, and the UI must render it as such rather than invent
        # times the schedule never had.
        weekdays = {dt.date.fromisoformat(d).weekday() for d in calendar["date"]}
        assert not weekdays & {0, 5}, "Monday or Saturday slots leaked into the calendar"

    def test_window_outside_the_seeded_range_is_empty(self, calendar):
        assert get_availability_calendar("2030-01-01", days=self.DAYS).empty


class TestAvailabilityFirstScheduling:
    """The candidate names the window; the database names the times.

    The bot never opens with dates of its own: it asks when the candidate is free,
    then queries the schedule for that window only. These tests cover the two
    halves of that — the query bounds, and the verdict invariant that keeps a
    scheduling turn from having nothing to say.
    """

    #: A Wednesday inside the seeded range. "next Tuesday" from here is 2026-09-01.
    ANCHOR = "2026-08-26"
    NEXT_TUESDAY = "2026-09-01"

    @pytest.fixture(scope="class")
    def db(self):
        db_file = Path(settings.db_url.removeprefix("sqlite:///"))
        if not db_file.is_file():
            pytest.skip("data/tech.db not built; run python scripts/seed_database.py")
        return get_engine()

    def test_the_tool_takes_the_candidates_window(self):
        # The model can only pass a window it can see in the signature.
        assert {"when", "until", "time_of_day"} <= set(slots_tool.args)

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("morning", "morning"),
            ("Afternoon", "afternoon"),
            ("", ""),
            ("sometime soon", ""),
            # The longest key wins: "early afternoon" contains both "early" and
            # "afternoon", and answering with 9-11 AM would contradict the candidate.
            ("early afternoon", "afternoon"),
        ],
    )
    def test_time_of_day_phrasing(self, expression, expected):
        assert _time_of_day_key(expression) == expected

    def test_time_of_day_windows_stay_inside_business_hours(self):
        # The schedule holds nothing outside 09:00-17:00, so an "afternoon" that
        # ran to 23:59 would only ever widen the query into rows that do not exist.
        for low, high in TIME_OF_DAY.values():
            assert "09:00:00" <= low <= high <= "17:00:00"

    def test_engine_confines_a_query_to_one_day(self, db):
        frame = get_available_slots(
            self.NEXT_TUESDAY, limit=9, to_date=self.NEXT_TUESDAY
        )
        assert not frame.empty
        assert set(frame["date"]) == {self.NEXT_TUESDAY}

    def test_engine_honours_the_time_bounds(self, db):
        frame = get_available_slots(
            self.NEXT_TUESDAY,
            limit=9,
            to_date=self.NEXT_TUESDAY,
            time_from="12:00:00",
            time_to="17:00:00",
        )
        assert all("12:00:00" <= t <= "17:00:00" for t in frame["time"])

    def test_a_single_day_answer_never_leaves_that_day(self, db):
        # The candidate said Tuesday. Answering with Wednesday morning would be
        # offering a time they had just told us they could not take.
        answer = slots_tool.invoke(
            {
                "conversation_date": self.ANCHOR,
                "when": "next Tuesday",
                "until": "next Tuesday",
                "limit": 9,
            }
        )
        dates = re.findall(r"\d{4}-\d{2}-\d{2}(?= \d{2}:)", answer)
        assert dates and set(dates) == {self.NEXT_TUESDAY}

    def test_a_time_of_day_answer_stays_in_that_half_of_the_day(self, db):
        answer = slots_tool.invoke(
            {
                "conversation_date": self.ANCHOR,
                "when": "next Tuesday",
                "until": "next Tuesday",
                "time_of_day": "afternoon",
                "limit": 9,
            }
        )
        hours = [int(h) for h in re.findall(r"\d{4}-\d{2}-\d{2} (\d{2}):", answer)]
        assert hours and all(hour >= 12 for hour in hours)

    def test_a_backwards_window_is_read_as_one_day(self, db):
        # `until` before `when` would return nothing at all, which reads to the
        # model as "their day is fully booked" rather than as a bad argument.
        answer = slots_tool.invoke(
            {
                "conversation_date": self.ANCHOR,
                "when": self.NEXT_TUESDAY,
                "until": "2026-08-01",
                "limit": 9,
            }
        )
        dates = re.findall(r"\d{4}-\d{2}-\d{2}(?= \d{2}:)", answer)
        assert dates and set(dates) == {self.NEXT_TUESDAY}

    def test_a_day_the_calendar_cannot_serve_returns_nothing_for_that_day(self, db):
        # Mondays are never seeded (ENGINEERING.md 6.3). The tool must say so rather
        # than quietly answering with another day the candidate did not ask for;
        # widening is the advisor's next call, and the message must explain it.
        answer = slots_tool.invoke(
            {
                "conversation_date": self.ANCHOR,
                "when": "Monday",
                "until": "Monday",
                "limit": 9,
            }
        )
        assert answer.startswith("No available")

    @pytest.mark.parametrize(
        ("from_date", "to_date", "expected"),
        [
            ("2026-09-01", "2026-09-01", "on 2026-09-01"),
            ("2026-09-01", "2026-09-04", "between 2026-09-01 and 2026-09-04"),
            ("2026-09-01", None, "from 2026-09-01"),
        ],
    )
    def test_window_is_phrased_for_the_model_to_quote(self, from_date, to_date, expected):
        assert describe_window(from_date, to_date) == expected

    def test_window_phrasing_carries_the_time_of_day(self):
        assert describe_window("2026-09-01", "2026-09-01", "afternoon").endswith("(afternoon)")

    @pytest.mark.parametrize(
        ("should_schedule", "needs_availability", "slots"),
        [
            # The model said "schedule" but returned nothing: that is an ask.
            (True, False, []),
            (True, True, []),
        ],
    )
    def test_a_scheduling_turn_with_no_slots_is_an_ask(
        self, should_schedule, needs_availability, slots
    ):
        # Otherwise the Main Agent gets a schedule turn with nothing to propose,
        # and the reliable way it fills that gap is by inventing a time.
        verdict = reconcile(
            SchedulingVerdict(
                should_schedule=should_schedule,
                needs_availability=needs_availability,
                slots=slots,
                reason="",
            )
        )
        assert verdict.needs_availability is True
        assert verdict.slots == []

    def test_real_slots_win_over_a_stale_ask(self):
        # The window is known, so asking again would ignore what they just said.
        verdict = reconcile(
            SchedulingVerdict(
                should_schedule=True,
                needs_availability=True,
                slots=[{"date": "2026-09-01", "time": "12:00:00"}],
                reason="",
            )
        )
        assert verdict.needs_availability is False
        assert len(verdict.slots) == 1

    def test_invented_slots_are_replaced_with_real_openings(self, db):
        # The failure this guards was live: asked for a Monday, the advisor
        # answered with three plausible Tuesday morning slots, none of them free.
        # A candidate offered one of those has an interview the calendar cannot
        # honour, and nothing upstream would have noticed.
        verdict = validate_slots(
            SchedulingVerdict(
                should_schedule=True,
                slots=[
                    {"date": self.NEXT_TUESDAY, "time": "09:00:00"},
                    {"date": self.NEXT_TUESDAY, "time": "10:00:00"},
                ],
                reason="Invented.",
            ),
            self.ANCHOR,
        )
        assert verdict.slots
        for slot in verdict.slots:
            assert is_slot_available(slot.date, slot.time)
        # Substituted slots stay on the day the advisor was aiming at.
        assert verdict.slots[0].date >= self.NEXT_TUESDAY

    def test_validation_keeps_the_slots_that_are_real(self, db):
        free = get_available_slots(self.NEXT_TUESDAY, limit=1, to_date=self.NEXT_TUESDAY)
        real = {"date": free.iloc[0]["date"], "time": free.iloc[0]["time"]}
        verdict = validate_slots(
            SchedulingVerdict(
                should_schedule=True,
                slots=[real, {"date": self.NEXT_TUESDAY, "time": "09:00:00"}],
                reason="Half real.",
            ),
            self.ANCHOR,
        )
        assert [(s.date, s.time) for s in verdict.slots] == [(real["date"], real["time"])]

    def test_validation_leaves_an_availability_ask_alone(self, db):
        # No slots to check, and no DB query to waste: this turn is a question.
        verdict = SchedulingVerdict(
            should_schedule=True, needs_availability=True, slots=[], reason=""
        )
        assert validate_slots(verdict, self.ANCHOR) == verdict

    def test_not_scheduling_carries_neither(self):
        verdict = reconcile(
            SchedulingVerdict(
                should_schedule=False,
                needs_availability=True,
                slots=[{"date": "2026-09-01", "time": "12:00:00"}],
                reason="",
            )
        )
        assert verdict.needs_availability is False
        assert verdict.slots == []


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


class TestDeployedDependencies:
    """requirements.txt is what Streamlit Cloud installs — keep it runtime-only."""

    #: Needed by the notebook, the tests or the offline scripts, never by the app.
    DEV_ONLY = ["PyPDF2", "sklearn", "matplotlib", "seaborn", "pytest", "jupyter", "ipykernel"]

    def test_app_imports_without_the_dev_dependencies(self):
        """The app must run on requirements.txt alone.

        Adding a top-level `import matplotlib` somewhere under app/ would work
        locally and break the deploy. This catches it offline.
        """
        blocked = {name.split(".")[0] for name in self.DEV_ONLY}

        class Block:
            def find_module(self, name, path=None):
                return self if name.split(".")[0] in blocked else None

            def load_module(self, name):
                raise ImportError(f"{name} is a dev-only dependency")

        guard = Block()
        sys.meta_path.insert(0, guard)
        try:
            for module in (
                "streamlit_app.utils",
                "app.modules.main_agent.orchestrator",
                "app.modules.embedding.retriever",
                "app.modules.database.engine",
            ):
                importlib.reload(importlib.import_module(module))
        finally:
            sys.meta_path.remove(guard)

    @pytest.mark.parametrize("package", DEV_ONLY)
    def test_dev_dependencies_are_not_in_the_deployed_file(self, package):
        runtime = (ROOT_DIR / "requirements.txt").read_text(encoding="utf-8")
        pinned = [
            line.split("==")[0].strip().lower()
            for line in runtime.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert package.lower() not in pinned

    def test_dev_file_pulls_in_the_runtime_file(self):
        dev = (ROOT_DIR / "requirements-dev.txt").read_text(encoding="utf-8")
        assert "-r requirements.txt" in dev


class TestMainAgentOutputRecovery:
    """The Main Agent's contract is JSON; off the happy path it writes prose."""

    VERDICT = InfoVerdict(info_needed=False, answer="")

    def test_parses_the_json_contract(self):
        raw = '{"action": "continue", "message": "Tell me more.", "reason": "probing"}'
        decision = _decision_from(raw, SCHEDULE, self.VERDICT)
        assert decision.message == "Tell me more."
        # The action is resolved in code and always overrides the model's.
        assert decision.action == SCHEDULE

    def test_plain_prose_is_used_as_the_message(self):
        """The observed failure: a usable reply was thrown away for a canned one."""
        raw = "I understand. Unfortunately, the next available slots are all on Tuesday the 18th."
        decision = _decision_from(raw, SCHEDULE, self.VERDICT)
        assert decision.message == raw
        assert decision.action == SCHEDULE

    def test_truncated_json_is_not_shown_to_the_candidate(self):
        """Half an object is not a message — fall back rather than leak braces."""
        decision = _decision_from('{"action": "continue", "mess', CONTINUE, self.VERDICT)
        assert "{" not in decision.message

    def test_empty_output_falls_back_to_the_info_answer(self):
        verdict = InfoVerdict(info_needed=True, answer="We use Django and Flask.")
        assert _decision_from("", CONTINUE, verdict).message == "We use Django and Flask."

    def test_nothing_at_all_still_produces_a_message(self):
        decision = _decision_from("   ", END, self.VERDICT)
        assert decision.message.strip()
        assert decision.action == END


class TestFineTuningPrompt:
    """The condensed prompt the fine-tuned Exit Advisor is trained and served on."""

    def test_exists(self):
        assert settings.prompt_path(PROMPT_FINETUNED).is_file()

    def test_carries_no_few_shot_examples(self):
        # Replacing the examples with learned behaviour is the point of
        # fine-tuning. If they creep back in, every training row pays for them.
        text = settings.prompt_path(PROMPT_FINETUNED).read_text(encoding="utf-8")
        assert "# Examples" not in text
        assert "<conversation id=" not in text

    def test_states_the_same_output_contract(self):
        text = settings.prompt_path(PROMPT_FINETUNED).read_text(encoding="utf-8")
        assert '"should_end"' in text and '"reason"' in text

    def test_context_is_last(self):
        text = settings.prompt_path(PROMPT_FINETUNED).read_text(encoding="utf-8")
        assert text.index("# Identity") < text.index("# Instructions") < text.index("# Context")

    def test_build_system_text_substitutes_the_conversation(self):
        rendered = build_system_text("Candidate: Hello.", finetuned=True)
        assert "Candidate: Hello." in rendered
        assert "{conversation}" not in rendered


class TestFineTuningDataset:
    """The JSONL that trains the Exit Advisor (ENGINEERING.md section 11)."""

    @pytest.fixture(scope="class")
    def split(self):
        return split_by_conversation(load_conversations(), test_size=5, seed=42)

    @pytest.fixture(scope="class")
    def examples(self, split):
        train_ids, _ = split
        return build_examples(train_ids)

    def test_held_out_conversations_never_enter_the_jsonl(self, split, examples):
        """The rule that, broken, silently invalidates the whole evaluation."""
        train_ids, test_ids = split
        points = build_decision_points(load_conversations())
        held_out_texts = {
            render_history(p.history) for p in points if p.conversation_id in test_ids
        }
        trained_texts = {e["messages"][0]["content"] for e in examples}
        assert not any(
            held and held in system for held in held_out_texts if held for system in trained_texts
        )
        # And the row count matches the training split exactly.
        assert len(examples) == sum(1 for p in points if p.conversation_id in train_ids)

    def test_row_shape_matches_an_inference_call(self, examples):
        for example in examples:
            roles = [m["role"] for m in example["messages"]]
            assert roles == ["system", "user", "assistant"]
            assert example["messages"][1]["content"] == DIRECTIVE

    def test_targets_validate_against_the_advisor_contract(self, examples):
        for example in examples:
            ExitVerdict.model_validate(json.loads(example["messages"][-1]["content"]))

    def test_should_end_follows_the_label(self, split, examples):
        train_ids, _ = split
        points = [
            p for p in build_decision_points(load_conversations()) if p.conversation_id in train_ids
        ]
        for point, example in zip(points, examples, strict=True):
            target = json.loads(example["messages"][-1]["content"])
            assert target["should_end"] is (point.label == END)

    def test_both_end_flavours_are_represented(self, examples):
        """`end` is terminal in both directions — training on one teaches the wrong rule."""
        reasons = [
            json.loads(e["messages"][-1]["content"])["reason"]
            for e in examples
            if json.loads(e["messages"][-1]["content"])["should_end"]
        ]
        assert any("opted out" in r for r in reasons), "no opt-out endings in the training data"
        assert any("agreed" in r for r in reasons), "no booked endings in the training data"

    def test_enough_rows_for_the_api(self, examples):
        assert len(examples) >= MIN_EXAMPLES

    def test_describe_flags_a_one_sided_dataset(self, examples):
        booked_only = [
            e
            for e in examples
            if "opted out" not in json.loads(e["messages"][-1]["content"])["reason"]
        ]
        assert "WARNING" in describe(booked_only)
        assert "WARNING" not in describe(examples)

    def test_platform_closure_is_told_apart_from_a_key_problem(self):
        """A 403 from the wind-down must not read as a data or credentials bug.

        OpenAI closed self-serve fine-tuning on 2026-05-07, so `create_job` now
        always fails here. Misreading that as a bad key would send the next
        person debugging the JSONL for nothing (ENGINEERING.md 11.9).
        """

        def error(code: str) -> PermissionDeniedError:
            body = {"error": {"message": "denied", "code": code}}
            response = httpx.Response(
                403,
                request=httpx.Request("POST", "https://api.openai.com/v1/fine_tuning/jobs"),
                json=body,
            )
            return PermissionDeniedError("Error code: 403", response=response, body=body)

        assert _is_platform_closed(error("training_not_available"))
        assert not _is_platform_closed(error("insufficient_permissions"))

    def test_write_jsonl_round_trips(self, examples, tmp_path):
        path = write_jsonl(examples, tmp_path / "train.jsonl")
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(examples)
        assert [json.loads(line) for line in lines] == examples
