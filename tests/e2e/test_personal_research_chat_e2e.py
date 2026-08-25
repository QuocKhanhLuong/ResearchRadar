"""Offline end-to-end scenarios for the personal research chat feature.

These tests exercise the full chat pipeline (mention admission -> routing ->
evidence assembly -> prompt construction -> structured LLM call -> response
validation -> memory capture) against deterministic fakes and a real temporary
SQLite database. No network, no credentials, no external services.

The production modules are written concurrently by other workers; the
importorskip calls below keep this suite green in a worktree that does not yet
contain them, and the scenarios go live as soon as integration lands.
"""

from __future__ import annotations

import asyncio
import logging
import re
import traceback
from datetime import UTC, datetime

import pytest

pytest.importorskip("research_radar.chat.service")
pytest.importorskip("research_radar.memory")
pytest.importorskip("research_radar.bot.mention")

from e2e.fakes import (  # noqa: E402
    BOT_USER_ID,
    FAKE_LLM_ANSWER,
    OTHER_BOT_USER_ID,
    PROMPT_DISCOVERY_HEADER,
    PROMPT_PROJECT_HEADER,
    PROMPT_QUESTION_HEADER,
    PROMPT_STORED_EVIDENCE_HEADER,
    PROMPT_SYSTEM_HEADER,
    PROMPT_USER_MEMORY_HEADER,
    SCIENTIFIC_EVIDENCE_HEADERS,
    SECRET_PAYLOAD,
    FailingLLMProvider,
    build_fake_user_memory,
    captured_episodes,
    joined_prompt_text,
    make_discord_message,
    prompt_section,
    seed_user_memory,
    total_scout_calls,
)
from research_radar.chat.models import ChatMode, ChatRequest  # noqa: E402
from research_radar.models.paper import Paper  # noqa: E402

MEMORY_PREFERENCE_QUERY = "what research topics do I prefer?"
DISCOVERY_QUERY = "find recent papers on quantum error correction"


async def _resolve_paper(repository, paper_id: str):
    """Resolve one canonical id through the repository off the event loop."""

    return await asyncio.to_thread(repository.get_paper, paper_id)


def _header_positions(prompt_text: str) -> dict[str, int]:
    """Return the character position of every contractual header that is present."""

    positions: dict[str, int] = {}
    for header in (
        PROMPT_SYSTEM_HEADER,
        PROMPT_USER_MEMORY_HEADER,
        PROMPT_PROJECT_HEADER,
        PROMPT_STORED_EVIDENCE_HEADER,
        PROMPT_DISCOVERY_HEADER,
        PROMPT_QUESTION_HEADER,
    ):
        found = re.search(rf"(?m)^{re.escape(header)}\s*$", prompt_text)
        if found is not None:
            positions[header] = found.start()
    return positions


async def test_hello_mention_gets_conversational_reply_with_zero_backends(
    chat_stack, count_rows, ingestion_repository
) -> None:
    """Scenario 1: a bare greeting is answered conversationally with no side effects."""

    stack = chat_stack()

    response = await stack.service.chat(ChatRequest(text="hello"))

    assert response.mode is ChatMode.CONVERSATIONAL
    assert FAKE_LLM_ANSWER in response.text
    assert stack.llm.call_count == 1
    assert total_scout_calls(stack.providers) == 0
    assert ingestion_repository.count_ingestion_runs() == 0
    assert count_rows("papers") == 0
    assert captured_episodes(stack.user_memory) == []


async def test_memory_question_uses_seeded_preferences_without_citations(
    chat_stack, count_rows, ingestion_repository
) -> None:
    """Scenario 2: personal-memory questions consult user memory only."""

    store = build_fake_user_memory("I prefer medical AI applications")
    seed_user_memory(store, "I track world models research")
    stack = chat_stack(user_memory=store)

    response = await stack.service.chat(ChatRequest(text=MEMORY_PREFERENCE_QUERY))

    # Both routes satisfy section 7 here: a first-person memory question may
    # land in PERSONAL_MEMORY, while the "I prefer" phrasing may legitimately
    # take the durable-statement CONVERSATIONAL route with memory lookup.
    assert response.mode in (ChatMode.PERSONAL_MEMORY, ChatMode.CONVERSATIONAL)
    assert response.used_user_memory is True
    assert FAKE_LLM_ANSWER in response.text
    assert total_scout_calls(stack.providers) == 0
    assert ingestion_repository.count_ingestion_runs() == 0
    assert count_rows("papers") == 0
    assert response.paper_ids == ()
    assert response.gap_ids == ()


async def test_new_topic_on_empty_corpus_discovers_and_persists_canonical_papers(
    chat_stack, repository, count_rows
) -> None:
    """Scenario 3: live discovery fills an empty corpus without any project setup."""

    stack = chat_stack()
    assert count_rows("papers") == 0

    response = await stack.service.chat(ChatRequest(text=DISCOVERY_QUERY))

    assert response.mode is ChatMode.RESEARCH_LIVE
    assert response.live_discovery_used is True
    assert total_scout_calls(stack.providers) == 3
    assert count_rows("papers") == 3
    paper_ids = list(response.paper_ids)
    assert len(paper_ids) == len(set(paper_ids))
    for paper_id in paper_ids:
        assert await _resolve_paper(repository, paper_id) is not None
    assert count_rows("projects") == 0
    assert count_rows("watch_topics") == 0


async def test_repeating_the_same_query_reuses_stored_results_without_duplicates(
    chat_stack, repository, count_rows
) -> None:
    """Scenario 4: the second identical turn is served from SQLite alone."""

    stack = chat_stack()
    first = await stack.service.chat(ChatRequest(text=DISCOVERY_QUERY))
    calls_after_first_turn = total_scout_calls(stack.providers)
    rows_after_first_turn = count_rows("papers")
    assert first.live_discovery_used is True
    assert rows_after_first_turn > 0

    second = await stack.service.chat(ChatRequest(text=DISCOVERY_QUERY))

    assert second.mode is ChatMode.RESEARCH_STORED
    assert second.live_discovery_used is False
    assert total_scout_calls(stack.providers) == calls_after_first_turn
    assert count_rows("papers") == rows_after_first_turn
    for paper_id in second.paper_ids:
        assert await _resolve_paper(repository, paper_id) is not None


async def test_unavailable_user_memory_backend_degrades_without_failing_chat(
    chat_stack, count_rows
) -> None:
    """Scenario 5: an outage of the advisory memory backend never blocks chat."""

    stack = chat_stack(user_memory=build_fake_user_memory(fail=True))

    response = await stack.service.chat(ChatRequest(text=MEMORY_PREFERENCE_QUERY))

    assert response.used_user_memory is False
    assert response.text
    assert FAKE_LLM_ANSWER in response.text
    assert total_scout_calls(stack.providers) == 0
    assert count_rows("papers") == 0


async def test_unavailable_semantic_index_preserves_lexical_and_live_paths(
    chat_stack, direct_ingestion, semantic_index, repository, count_rows
) -> None:
    """Scenario 6: Pinecone-style outages degrade retrieval to lexical only."""

    _, mri_ingestion = direct_ingestion("brain mri segmentation", works=3)
    await mri_ingestion.ingest_research_topic("brain mri segmentation")
    semantic_index.set_available(False)
    assert semantic_index.available is False
    stack = chat_stack(semantic_index=semantic_index)

    live = await stack.service.chat(ChatRequest(text=DISCOVERY_QUERY))
    assert live.live_discovery_used is True
    live_rows = count_rows("papers")
    assert live_rows >= 3
    for paper_id in live.paper_ids:
        assert await _resolve_paper(repository, paper_id) is not None

    calls_before_stored_turn = total_scout_calls(stack.providers)
    stored = await stack.service.chat(ChatRequest(text="find work on brain mri segmentation"))
    assert stored.mode is ChatMode.RESEARCH_STORED
    assert stored.live_discovery_used is False
    assert total_scout_calls(stack.providers) == calls_before_stored_turn
    lexical_hits = await asyncio.to_thread(
        repository.get_papers_for_local_lexical_search, "brain mri segmentation", 8
    )
    assert len(lexical_hits) >= 3


async def test_llm_failure_returns_degraded_response_and_writes_no_episodes(
    chat_stack, ingestion_repository, count_rows
) -> None:
    """Scenario 7: an LLM outage yields a safe degraded answer and zero captures."""

    failing = FailingLLMProvider()
    stack = chat_stack(llm=failing)

    response = await stack.service.chat(
        ChatRequest(text="I prefer world models over reinforcement learning")
    )

    assert response.degraded is True
    assert response.text
    assert FAKE_LLM_ANSWER not in response.text
    assert failing.attempts == 1
    assert captured_episodes(stack.user_memory) == []
    assert total_scout_calls(stack.providers) == 0
    assert ingestion_repository.count_ingestion_runs() == 0
    assert count_rows("papers") == 0


def test_guild_message_without_bot_mention_triggers_no_backends(
    mention_policy, chat_stack
) -> None:
    """Scenario 8: unmentioned guild chatter is admitted nowhere at all."""

    message = make_discord_message("completely ordinary guild chatter")
    stack = chat_stack()

    admission = mention_policy.admit(message, bot_user_id=BOT_USER_ID)

    assert admission.accepted is False
    assert admission.reason == "no_mention"
    assert stack.llm.call_count == 0
    assert total_scout_calls(stack.providers) == 0
    assert captured_episodes(stack.user_memory) == []


def test_message_from_another_bot_is_ignored(mention_policy, chat_stack) -> None:
    """Scenario 9: messages authored by other bots never reach the pipeline."""

    message = make_discord_message(
        f"<@{BOT_USER_ID}> hello are you there",
        author_id=OTHER_BOT_USER_ID,
        author_is_bot=True,
    )
    stack = chat_stack()

    admission = mention_policy.admit(message, bot_user_id=BOT_USER_ID)

    assert admission.accepted is False
    assert admission.reason == "bot_author"
    assert stack.llm.call_count == 0
    assert total_scout_calls(stack.providers) == 0
    assert captured_episodes(stack.user_memory) == []


async def test_message_with_secret_is_never_captured_logged_or_raised(
    chat_stack, caplog
) -> None:
    """Scenario 10: secret-bearing messages are rejected everywhere they flow."""

    stack = chat_stack()
    request_text = f"please remember my {SECRET_PAYLOAD} for later"

    with caplog.at_level(logging.DEBUG):
        response = await stack.service.chat(ChatRequest(text=request_text))
        log_text = "\n".join(record.getMessage() for record in caplog.records)
        for record in caplog.records:
            if record.exc_text:
                log_text += f"\n{record.exc_text}"
            if record.exc_info:
                log_text += "\n" + "".join(traceback.format_exception(*record.exc_info))

    assert captured_episodes(stack.user_memory) == []
    assert SECRET_PAYLOAD not in log_text
    assert SECRET_PAYLOAD not in response.text


async def test_user_belief_is_confined_to_the_advisory_prompt_section(
    chat_stack, repository
) -> None:
    """Scenario 11: unsupported personal beliefs stay out of scientific evidence."""

    claim = "I think contrastive world models are novel"
    repository.upsert_merged_paper(
        Paper(
            id="openalex:W501",
            title="Brain MRI Segmentation Benchmarks",
            doi="10.7777/mri-bench.1",
            source="openalex",
            external_ids={"openalex": "W501", "doi": "10.7777/mri-bench.1"},
        )
    )
    stack = chat_stack(user_memory=build_fake_user_memory(claim))

    response = await stack.service.chat(
        ChatRequest(text="what do I think about contrastive world models?")
    )

    assert response.used_user_memory is True
    prompt_text = joined_prompt_text(stack.llm.last_messages)
    advisory_section = prompt_section(prompt_text, PROMPT_USER_MEMORY_HEADER)
    assert claim in advisory_section
    for scientific_header in SCIENTIFIC_EVIDENCE_HEADERS:
        if re.search(rf"(?m)^{re.escape(scientific_header)}\s*$", prompt_text):
            assert claim not in prompt_section(prompt_text, scientific_header)
    question_section = prompt_section(prompt_text, PROMPT_QUESTION_HEADER)
    assert "contrastive world models" in question_section
    system_section = prompt_section(prompt_text, PROMPT_SYSTEM_HEADER).casefold()
    assert "user memory" in system_section
    assert claim not in response.text


async def test_project_rejected_idea_outranks_older_graphiti_fact(
    chat_stack, repository
) -> None:
    """Scenario 12: explicit SQLite project state beats advisory Graphiti memory."""

    from research_radar.memory.models import MemoryFact

    repository.create_project(
        "GAN Robustness",
        rejected_ideas=["Pure GAN pipelines for MRI reconstruction"],
    )
    older_belief = MemoryFact(
        fact="User believes pure GAN pipelines always win for MRI reconstruction",
        valid_at=datetime(2020, 1, 1, tzinfo=UTC),
        source="graphiti",
    )
    stack = chat_stack(user_memory=build_fake_user_memory(older_belief))

    response = await stack.service.chat(
        ChatRequest(text="what should we try next on this project?", project_hint="GAN Robustness")
    )

    assert response.mode is ChatMode.PROJECT_RESEARCH
    prompt_text = joined_prompt_text(stack.llm.last_messages)
    project_section = prompt_section(prompt_text, PROMPT_PROJECT_HEADER)
    advisory_section = prompt_section(prompt_text, PROMPT_USER_MEMORY_HEADER)
    assert "Pure GAN pipelines for MRI reconstruction" in project_section
    assert "pure GAN pipelines always win" in advisory_section
    system_section = prompt_section(prompt_text, PROMPT_SYSTEM_HEADER).casefold()
    assert "conflict" in system_section
    assert re.search(
        r"explicit\s+project\s+memory[^.!]{0,240}\b(?:wins|outranks?|overrides?|precedence)\b",
        system_section,
    )
    positions = _header_positions(prompt_text)
    assert positions[PROMPT_USER_MEMORY_HEADER] < positions[PROMPT_PROJECT_HEADER]
    assert positions[PROMPT_PROJECT_HEADER] < positions[PROMPT_QUESTION_HEADER]
