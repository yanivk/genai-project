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
from app.modules.advisors.scheduling_advisor import resolve_relative_date
from app.modules.advisors.schemas import ExitVerdict, InfoVerdict, SchedulingVerdict
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
    """The JSONL that trains the Exit Advisor (CLAUDE.md section 11)."""

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
        person debugging the JSONL for nothing (CLAUDE.md 11.9).
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
