"""Tests for research_radar.memory.capture: the memory capture policy."""

from __future__ import annotations

import base64
from dataclasses import asdict

import pytest

from research_radar.memory.capture import CaptureDecision, MemoryCapturePolicy
from research_radar.memory.models import MemoryClass
from research_radar.memory.secrets import REDACTED_PLACEHOLDER, contains_secret, redact_secrets


def _fake_discord_token() -> str:
    """Build a structurally valid but entirely synthetic Discord-style token."""
    first = base64.urlsafe_b64encode(b"555666777888999000").decode().rstrip("=")
    middle = "QmFzZTY0"
    third = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4OpQr5S"
    return f"{first}.{middle}.{third}"


def _assert_no_echo(decision: CaptureDecision, *forbidden: str) -> None:
    """Assert no field of the decision contains any forbidden substring."""
    for field_name, value in asdict(decision).items():
        for needle in forbidden:
            assert needle not in str(value), f"field {field_name} echoes secret"


def _pdf_paragraph() -> str:
    """A >2000-char single paragraph with no first-person marker."""
    sentence = (
        "Sedimentary basins preserve a continuous record of tectonic subsidence "
        "and paleoclimate variation across geological timescales. "
    )
    return sentence.strip() * 30


class TestDisabledPolicy:
    """enabled=False rejects everything with reason=capture_disabled."""

    def test_durable_message_rejected(self) -> None:
        policy = MemoryCapturePolicy(enabled=False)
        decision = policy.evaluate_user_message("I prefer concise answers.")
        assert decision.should_store is False
        assert decision.memory_class is None
        assert decision.reason == "capture_disabled"
        assert decision.redacted_text == ""

    def test_secret_message_still_reports_disabled(self) -> None:
        policy = MemoryCapturePolicy(enabled=False)
        decision = policy.evaluate_user_message(f"token {_fake_discord_token()}")
        assert decision.should_store is False
        assert decision.reason == "capture_disabled"


class TestAssistantMessages:
    """Assistant output is never stored, unconditionally."""

    def test_assistant_message_never_stored(self) -> None:
        policy = MemoryCapturePolicy()
        decision = policy.evaluate_assistant_message("Here is my summary of the paper.")
        assert decision.should_store is False
        assert decision.memory_class is None
        assert decision.reason == "assistant_output_never_stored"
        assert decision.redacted_text == ""

    def test_assistant_message_rejected_even_when_disabled(self) -> None:
        policy = MemoryCapturePolicy(enabled=False)
        decision = policy.evaluate_assistant_message("anything")
        assert decision.should_store is False
        assert decision.reason == "assistant_output_never_stored"


class TestAdversarialLeak:
    """Explicit adversarial case from the task spec."""

    def test_discord_token_plus_openai_key_is_rejected_and_scrubbed(self) -> None:
        token = _fake_discord_token()
        key = "sk-fake000000000000000000key"
        message = f"My bot token {token} and my OpenAI key {key} leaked, rotating now."
        assert contains_secret(message)

        policy = MemoryCapturePolicy()
        decision = policy.evaluate_user_message(message)

        assert decision.should_store is False
        assert decision.memory_class is None
        assert decision.reason == "secret_detected"
        assert decision.redacted_text == ""
        _assert_no_echo(decision, token, key)

        scrubbed = redact_secrets(message)
        assert token not in scrubbed
        assert key not in scrubbed
        assert REDACTED_PLACEHOLDER in scrubbed


class TestSecretRejection:
    """Secrets reject the whole message before any other rule runs."""

    def test_reason_does_not_echo_content(self) -> None:
        policy = MemoryCapturePolicy()
        secret = "AKIAIOSFODNN7EXAMPLE"
        decision = policy.evaluate_user_message(f"I only have one key left: {secret}")
        assert decision.should_store is False
        assert decision.reason == "secret_detected"
        assert decision.redacted_text == ""
        _assert_no_echo(decision, secret)


class TestStructuralRejections:
    """Stack traces, shell/log dumps, and PDF-style dumps are rejected."""

    def test_traceback_with_header(self) -> None:
        text = (
            "Traceback (most recent call last):\n"
            '  File "/app/service.py", line 42, in run\n'
            "    result = pipeline.execute(data)\n"
            "ValueError: bad input shape"
        )
        decision = MemoryCapturePolicy().evaluate_user_message(text)
        assert decision.should_store is False
        assert decision.reason == "stack_trace"

    def test_three_exception_lines(self) -> None:
        text = "KeyError: missing field\nTypeError: wrong type\nIndexError: way out of range"
        decision = MemoryCapturePolicy().evaluate_user_message(text)
        assert decision.should_store is False
        assert decision.reason == "stack_trace"

    def test_shell_block(self) -> None:
        text = "$ pytest -q tests/\n$ ruff check .\n> all checks passed"
        decision = MemoryCapturePolicy().evaluate_user_message(text)
        assert decision.should_store is False
        assert decision.reason == "shell_or_log_dump"

    def test_timestamped_log_block(self) -> None:
        text = (
            "2026-08-26 09:14:01 INFO server started on port 8000\n"
            "2026-08-26 09:14:02 INFO request received GET /health\n"
            "[09:14:03] WARN slow query took 1200ms"
        )
        decision = MemoryCapturePolicy().evaluate_user_message(text)
        assert decision.should_store is False
        assert decision.reason == "shell_or_log_dump"

    def test_pdf_style_dump(self) -> None:
        decision = MemoryCapturePolicy().evaluate_user_message(_pdf_paragraph())
        assert decision.should_store is False
        assert decision.reason == "pdf_text_dump"

    def test_same_dump_with_first_person_is_not_a_dump(self) -> None:
        text = _pdf_paragraph() + " I usually skim the methods sections."
        decision = MemoryCapturePolicy().evaluate_user_message(text)
        assert decision.should_store is True
        assert decision.memory_class == MemoryClass.PREFERENCE


class TestNonDurableChatter:
    """Greetings, thanks, questions, and very short messages are rejected."""

    def test_greeting(self) -> None:
        decision = MemoryCapturePolicy().evaluate_user_message("hey there!")
        assert decision.should_store is False
        assert decision.reason == "non_durable_chatter"

    def test_thanks(self) -> None:
        decision = MemoryCapturePolicy().evaluate_user_message("got it, thanks")
        assert decision.should_store is False
        assert decision.reason == "non_durable_chatter"

    def test_acknowledgement(self) -> None:
        assert MemoryCapturePolicy().evaluate_user_message("sounds good").should_store is False

    def test_bare_question(self) -> None:
        text = "What's the state of the art on sparse attention?"
        decision = MemoryCapturePolicy().evaluate_user_message(text)
        assert decision.should_store is False
        assert decision.reason == "question_not_durable"

    def test_durable_statement_with_question_mark_accepted(self) -> None:
        text = "We decided to go with SQLite — any objections?"
        decision = MemoryCapturePolicy().evaluate_user_message(text)
        assert decision.should_store is True
        assert decision.memory_class == MemoryClass.PROJECT_DECISION

    def test_short_message(self) -> None:
        decision = MemoryCapturePolicy().evaluate_user_message("brb")
        assert decision.should_store is False
        assert decision.reason == "message_too_short"

    def test_empty_text(self) -> None:
        policy = MemoryCapturePolicy()
        assert policy.evaluate_user_message("").reason == "empty_text"
        assert policy.evaluate_user_message("   \n\t ").reason == "empty_text"


class TestClassification:
    """Durable statements land in the right MemoryClass bucket."""

    def case_params(self) -> list[tuple[str, MemoryClass]]:
        return [
            ("I prefer concise answers with citations.", MemoryClass.PREFERENCE),
            ("Let's stop working on the vector database rewrite.", MemoryClass.REJECTED_IDEA),
            ("My goal is to finish the retrieval eval harness.", MemoryClass.GOAL),
            ("I only have CPU access and a tight budget.", MemoryClass.CONSTRAINT),
            ("I'm researching protein folding dynamics.", MemoryClass.RESEARCH_INTEREST),
            ("We decided to go with SQLite for storage.", MemoryClass.PROJECT_DECISION),
            ("I prefer pytest over unittest these days.", MemoryClass.TOOL_PREFERENCE),
            (
                "Always run tests first before committing anything.",
                MemoryClass.WORKFLOW_PREFERENCE,
            ),
            (
                "Next I want to explore retrieval evaluation datasets.",
                MemoryClass.RESEARCH_DIRECTION,
            ),
            ("I'm giving the lab talk next Friday.", MemoryClass.TEMPORAL_PLAN),
        ]

    def test_each_class_is_recognized(self) -> None:
        policy = MemoryCapturePolicy()
        for text, expected in self.case_params():
            decision = policy.evaluate_user_message(text)
            assert decision.should_store is True, text
            assert decision.memory_class == expected, text

    def test_accept_reason_names_the_class(self) -> None:
        decision = MemoryCapturePolicy().evaluate_user_message(
            "My goal is to finish the retrieval eval harness."
        )
        assert decision.reason == "accepted_goal"

    def test_ambiguous_durable_statement_falls_back_to_preference(self) -> None:
        text = "I keep losing track of which papers I already read."
        decision = MemoryCapturePolicy().evaluate_user_message(text)
        assert decision.should_store is True
        assert decision.memory_class == MemoryClass.PREFERENCE

    def test_non_durable_statement_rejected(self) -> None:
        text = "The mitochondria is the powerhouse of the cell and similar textbook facts."
        decision = MemoryCapturePolicy().evaluate_user_message(text)
        assert decision.should_store is False
        assert decision.reason == "not_durable"

    def test_redacted_text_is_whitespace_normalized(self) -> None:
        text = "I   prefer\n  short\tanswers  with citations."
        decision = MemoryCapturePolicy().evaluate_user_message(text)
        assert decision.should_store is True
        assert decision.redacted_text == "I prefer short answers with citations."

    def test_redacted_text_defensively_redacts_on_accept_path(self) -> None:
        text = "I like turtles, especially hatchlings."
        decision = MemoryCapturePolicy().evaluate_user_message(text)
        assert decision.redacted_text == redact_secrets("I like turtles, especially hatchlings.")


class TestImperativeResearchCommands:
    """Imperative commands and research queries must not be captured as user memory."""

    @pytest.mark.parametrize(
        "command",
        [
            "find recent papers on low-field MRI reconstruction",
            "search for papers on diffusion policy in robotics",
            "compare Transformer and Mamba architectures for long contexts",
            "summarize the latest work on protein structure prediction",
            "give me an overview of neural radiance fields",
            "list the top benchmarks for retrieval augmented generation",
        ],
    )
    def test_imperative_research_commands_not_stored(self, command: str) -> None:
        decision = MemoryCapturePolicy().evaluate_user_message(command)
        assert decision.should_store is False
        assert decision.reason in {"not_durable", "question_not_durable"}
        assert decision.redacted_text == ""


class TestTemporalPlanVariations:
    """Explicit temporal plans with first-person markers are classified as TEMPORAL_PLAN."""

    @pytest.mark.parametrize(
        "statement",
        [
            "I'm giving the lab talk tomorrow.",
            "I will submit the revised manuscript by next week.",
            "We plan to wrap up the evaluation in 2 weeks.",
            "I have a meeting with my advisor on Friday.",
            "I'm presenting our findings at NeurIPS in December.",
            "We need to finalize the paper by EOD.",
            "I am scheduled to run experiments tonight.",
        ],
    )
    def test_temporal_plans_classified_correctly(self, statement: str) -> None:
        decision = MemoryCapturePolicy().evaluate_user_message(statement)
        assert decision.should_store is True
        assert decision.memory_class == MemoryClass.TEMPORAL_PLAN



class TestInterrogativeSentencesAreNeverStored:
    """A question that merely CONTAINS a durable phrase is still a question.

    Classifying the raw message let an interrogative be read as an assertion:
    "should I drop the GAN baseline?" was stored as a REJECTED_IDEA, recording
    the opposite of what the user said. Only non-interrogative sentences are
    classified, and only those are persisted.
    """

    @pytest.mark.parametrize(
        "question",
        [
            "what do I prefer for training frameworks?",
            "what do I like about diffusion models?",
            "do I usually use pytest or unittest?",
            "should I drop the GAN baseline?",
            "can you remember what I decided about kuzu?",
            "did we decide to go with SQLite?",
            "is my goal still low-field MRI reconstruction?",
            "what am I working on this week?",
        ],
    )
    def test_first_person_question_is_not_stored(self, question: str) -> None:
        decision = MemoryCapturePolicy().evaluate_user_message(question)
        assert decision.should_store is False
        assert decision.reason == "question_not_durable"
        assert decision.redacted_text == ""

    def test_tag_question_after_a_decision_is_still_stored(self) -> None:
        """A decision with a tag question attached is a decision, not a question."""

        text = "We decided to go with SQLite - any objections?"
        decision = MemoryCapturePolicy().evaluate_user_message(text)
        assert decision.should_store is True
        assert decision.memory_class == MemoryClass.PROJECT_DECISION

    def test_statement_beside_a_question_stores_only_the_statement(self) -> None:
        decision = MemoryCapturePolicy().evaluate_user_message(
            "I prefer polars over pandas. What papers cover it?"
        )
        assert decision.should_store is True
        assert decision.memory_class == MemoryClass.TOOL_PREFERENCE
        assert decision.redacted_text == "I prefer polars over pandas."
        assert "What papers cover it?" not in decision.redacted_text
