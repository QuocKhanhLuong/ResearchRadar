"""Deterministic offline smoke test for the personal research chat feature.

Mirrors ``scripts/smoke_test_research_radar.py`` and exercises the same twelve
scenarios as ``tests/e2e/test_personal_research_chat_e2e.py``, but standalone:
no network, no credentials, no Discord connection, no external LLM APIs.

Requires no command-line setup. Prints one readable PASS/FAIL line per scenario
and exits non-zero when any scenario fails. When the personal-research-chat
modules have not been integrated into this checkout yet, the script prints a
SKIP notice and exits successfully, exactly like the importorskip-gated e2e
suite.

Usage:
    python scripts/smoke_test_personal_research_chat.py [--db-url sqlite:///...]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(REPO_ROOT / "src"), str(REPO_ROOT / "tests")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from e2e.fakes import (  # noqa: E402
    BOT_USER_ID,
    FAKE_LLM_ANSWER,
    OTHER_BOT_USER_ID,
    PROMPT_PROJECT_HEADER,
    PROMPT_QUESTION_HEADER,
    PROMPT_SYSTEM_HEADER,
    PROMPT_USER_MEMORY_HEADER,
    SCIENTIFIC_EVIDENCE_HEADERS,
    SECRET_PAYLOAD,
    FailingLLMProvider,
    RecordingLLMProvider,
    build_fake_user_memory,
    captured_episodes,
    joined_prompt_text,
    make_discord_message,
    prompt_section,
    provider_trio_for_topic,
    seed_user_memory,
    total_scout_calls,
)

MEMORY_PREFERENCE_QUERY = "what research topics do I prefer?"
DISCOVERY_QUERY = "find recent papers on quantum error correction"


class _ListLogHandler(logging.Handler):
    """Capture every log record so secrets can be scanned for afterwards."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Store the record verbatim for offline inspection."""

        self.records.append(record)


def _count_rows(database: Any, table_name: str) -> int:
    """Return the raw row count of one table for persistence assertions."""

    from sqlalchemy import text

    with database.engine.connect() as connection:
        return int(connection.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())


def _build_stack(ctx: SimpleNamespace, **overrides: Any) -> SimpleNamespace:
    """Assemble one fully wired offline chat stack over the shared database."""

    from research_radar.chat.router import ChatRouter
    from research_radar.chat.service import ChatService

    from research_radar.research.ingestion import IngestionService
    from research_radar.research.scout import ScoutService

    llm = overrides.get("llm") or RecordingLLMProvider()
    user_memory = overrides.get("user_memory") or build_fake_user_memory()
    providers = provider_trio_for_topic()
    ingestion_service = IngestionService(
        scout=ScoutService(list(providers)),
        repository=ctx.repository,
        ingestion_repository=ctx.ingestion_repository,
        reader_service=None,
        metadata_limit=50,
    )
    service = ChatService(
        repository=ctx.repository,
        router=ChatRouter(llm_provider=llm),
        user_memory=user_memory,
        capture_policy=ctx.capture_policy,
        llm_provider=llm,
        ingestion_service=ingestion_service,
        embedding_provider=None,
        semantic_index=overrides.get("semantic_index"),
        budget=None,
    )
    return SimpleNamespace(service=service, llm=llm, user_memory=user_memory, providers=providers)


async def _resolve_paper(repository: Any, paper_id: str) -> Any:
    """Resolve one canonical id through the repository off the event loop."""

    return await asyncio.to_thread(repository.get_paper, paper_id)


async def _scenario_01_hello(ctx: SimpleNamespace) -> tuple[bool, str]:
    """Greeting -> conversational LLM answer, zero providers, zero ingestion."""

    stack = _build_stack(ctx)
    response = await stack.service.chat(ctx.ChatRequest(text="hello"))
    ok = (
        response.mode is ctx.ChatMode.CONVERSATIONAL
        and FAKE_LLM_ANSWER in response.text
        and stack.llm.call_count == 1
        and total_scout_calls(stack.providers) == 0
        and ctx.ingestion_repository.count_ingestion_runs() == 0
        and _count_rows(ctx.database, "papers") == 0
        and not captured_episodes(stack.user_memory)
    )
    return ok, f"mode={response.mode.value}"


async def _scenario_02_preferences(ctx: SimpleNamespace) -> tuple[bool, str]:
    """Seeded preferences answer a memory question without citations."""

    store = build_fake_user_memory("I prefer medical AI applications")
    seed_user_memory(store, "I track world models research")
    stack = _build_stack(ctx, user_memory=store)
    response = await stack.service.chat(ctx.ChatRequest(text=MEMORY_PREFERENCE_QUERY))
    ok = (
        response.mode in (ctx.ChatMode.PERSONAL_MEMORY, ctx.ChatMode.CONVERSATIONAL)
        and response.used_user_memory is True
        and FAKE_LLM_ANSWER in response.text
        and total_scout_calls(stack.providers) == 0
        and response.paper_ids == ()
        and response.gap_ids == ()
    )
    return ok, f"used_user_memory={response.used_user_memory}"


async def _scenario_03_live_discovery(ctx: SimpleNamespace) -> tuple[bool, str]:
    """Empty corpus plus fresh topic -> discovery, dedup, persistence, no setup."""

    stack = _build_stack(ctx)
    ctx.stack = stack
    response = await stack.service.chat(ctx.ChatRequest(text=DISCOVERY_QUERY))
    rows = _count_rows(ctx.database, "papers")
    resolved = [await _resolve_paper(ctx.repository, paper_id) for paper_id in response.paper_ids]
    ok = (
        response.mode is ctx.ChatMode.RESEARCH_LIVE
        and response.live_discovery_used is True
        and total_scout_calls(stack.providers) == 3
        and rows == 3
        and len(set(response.paper_ids)) == len(response.paper_ids)
        and all(resolved)
        and _count_rows(ctx.database, "projects") == 0
        and _count_rows(ctx.database, "watch_topics") == 0
    )
    return ok, f"papers={rows} live={response.live_discovery_used}"


async def _scenario_04_repeat_query(ctx: SimpleNamespace) -> tuple[bool, str]:
    """Repeating the query reuses stored evidence with zero duplicate rows."""

    assert ctx.stack is not None, "Scenario 3 must run before scenario 4."
    stack = ctx.stack
    calls_before = total_scout_calls(stack.providers)
    rows_before = _count_rows(ctx.database, "papers")
    response = await stack.service.chat(ctx.ChatRequest(text=DISCOVERY_QUERY))
    rows_after = _count_rows(ctx.database, "papers")
    calls_after = total_scout_calls(stack.providers)
    ok = (
        response.mode is ctx.ChatMode.RESEARCH_STORED
        and response.live_discovery_used is False
        and calls_after == calls_before
        and rows_after == rows_before
    )
    detail = f"papers {rows_before}->{rows_after} scout {calls_before}->{calls_after}"
    return ok, detail


async def _scenario_05_memory_outage(ctx: SimpleNamespace) -> tuple[bool, str]:
    """User-memory backend outage -> chat still succeeds without memory."""

    stack = _build_stack(ctx, user_memory=build_fake_user_memory(fail=True))
    response = await stack.service.chat(ctx.ChatRequest(text=MEMORY_PREFERENCE_QUERY))
    ok = (
        response.used_user_memory is False
        and bool(response.text)
        and FAKE_LLM_ANSWER in response.text
        and total_scout_calls(stack.providers) == 0
    )
    return ok, f"used_user_memory={response.used_user_memory}"


async def _scenario_06_semantic_outage(ctx: SimpleNamespace) -> tuple[bool, str]:
    """Semantic index down -> lexical retrieval and live discovery still work."""

    from e2e.fakes import FakeSemanticIndex

    from research_radar.research.ingestion import IngestionService
    from research_radar.research.scout import ScoutService

    seed_trio = provider_trio_for_topic("brain mri segmentation", works=3)
    seed_ingestion = IngestionService(
        scout=ScoutService(list(seed_trio)),
        repository=ctx.repository,
        ingestion_repository=ctx.ingestion_repository,
        reader_service=None,
        metadata_limit=50,
    )
    await seed_ingestion.ingest_research_topic("brain mri segmentation")
    index = FakeSemanticIndex()
    index.set_available(False)
    stack = _build_stack(ctx, semantic_index=index)
    # A topic no earlier scenario has ingested keeps the live-discovery path live.
    live = await stack.service.chat(
        ctx.ChatRequest(text="find recent papers on topological photonics computing")
    )
    calls_before = total_scout_calls(stack.providers)
    stored = await stack.service.chat(ctx.ChatRequest(text="find work on brain mri segmentation"))
    lexical_hits = await asyncio.to_thread(
        ctx.repository.get_papers_for_local_lexical_search, "brain mri segmentation", 8
    )
    ok = (
        index.available is False
        and live.live_discovery_used is True
        and _count_rows(ctx.database, "papers") >= 6
        and total_scout_calls(stack.providers) == calls_before
        and stored.mode is ctx.ChatMode.RESEARCH_STORED
        and stored.live_discovery_used is False
        and len(lexical_hits) >= 3
    )
    detail = (
        f"live={live.live_discovery_used} lexical_hits={len(lexical_hits)} "
        f"scout_delta={total_scout_calls(stack.providers) - calls_before}"
    )
    return ok, detail


async def _scenario_07_llm_outage(ctx: SimpleNamespace) -> tuple[bool, str]:
    """LLM outage -> degraded safe answer and zero memory captures."""

    failing = FailingLLMProvider()
    stack = _build_stack(ctx, llm=failing)
    response = await stack.service.chat(
        ctx.ChatRequest(text="I prefer world models over reinforcement learning")
    )
    episodes = captured_episodes(stack.user_memory)
    ok = (
        response.degraded is True
        and bool(response.text)
        and FAKE_LLM_ANSWER not in response.text
        and failing.attempts == 1
        and not episodes
        and total_scout_calls(stack.providers) == 0
    )
    return ok, f"degraded={response.degraded} episodes={len(episodes)}"


def _scenario_08_no_mention(ctx: SimpleNamespace) -> tuple[bool, str]:
    """Guild message without a mention is admitted nowhere."""

    message = make_discord_message("completely ordinary guild chatter")
    stack = _build_stack(ctx)
    admission = ctx.mention_policy.admit(message, bot_user_id=BOT_USER_ID)
    ok = (
        admission.accepted is False
        and admission.reason == "no_mention"
        and stack.llm.call_count == 0
        and total_scout_calls(stack.providers) == 0
        and not captured_episodes(stack.user_memory)
    )
    return ok, f"reason={admission.reason}"


def _scenario_09_other_bot(ctx: SimpleNamespace) -> tuple[bool, str]:
    """Messages authored by another bot are ignored entirely."""

    message = make_discord_message(
        f"<@{BOT_USER_ID}> hello are you there",
        author_id=OTHER_BOT_USER_ID,
        author_is_bot=True,
    )
    stack = _build_stack(ctx)
    admission = ctx.mention_policy.admit(message, bot_user_id=BOT_USER_ID)
    ok = (
        admission.accepted is False
        and admission.reason == "bot_author"
        and stack.llm.call_count == 0
        and not captured_episodes(stack.user_memory)
    )
    return ok, f"reason={admission.reason}"


async def _scenario_10_secret(ctx: SimpleNamespace) -> tuple[bool, str]:
    """Secret-bearing message leaves no trace in episodes or responses."""

    stack = _build_stack(ctx)
    handler = _ListLogHandler()
    logging.getLogger().addHandler(handler)
    try:
        response = await stack.service.chat(
            ctx.ChatRequest(text=f"please remember my {SECRET_PAYLOAD} for later")
        )
    finally:
        logging.getLogger().removeHandler(handler)
    log_text = "\n".join(record.getMessage() for record in handler.records)
    for record in handler.records:
        if record.exc_text:
            log_text += f"\n{record.exc_text}"
        if record.exc_info:
            log_text += "\n" + "".join(traceback.format_exception(*record.exc_info))
    episodes = captured_episodes(stack.user_memory)
    ok = (
        not episodes
        and SECRET_PAYLOAD not in log_text
        and SECRET_PAYLOAD not in response.text
    )
    return ok, "secret contained"


async def _scenario_11_belief_placement(ctx: SimpleNamespace) -> tuple[bool, str]:
    """Personal belief lands under USER MEMORY, never under evidence sections."""

    claim = "I think contrastive world models are novel"
    ctx.repository.upsert_merged_paper(
        ctx.Paper(
            id="openalex:W501",
            title="Brain MRI Segmentation Benchmarks",
            doi="10.7777/mri-bench.1",
            source="openalex",
            external_ids={"openalex": "W501", "doi": "10.7777/mri-bench.1"},
        )
    )
    stack = _build_stack(ctx, user_memory=build_fake_user_memory(claim))
    response = await stack.service.chat(
        ctx.ChatRequest(text="what do I think about contrastive world models?")
    )
    prompt_text = joined_prompt_text(stack.llm.last_messages)
    advisory_section = prompt_section(prompt_text, PROMPT_USER_MEMORY_HEADER)
    clean_evidence = all(
        claim not in prompt_section(prompt_text, header)
        for header in SCIENTIFIC_EVIDENCE_HEADERS
        if header in prompt_text
    )
    system_rules = prompt_section(prompt_text, PROMPT_SYSTEM_HEADER).casefold()
    ok = (
        response.used_user_memory is True
        and claim in advisory_section
        and clean_evidence
        and claim not in response.text
        and "user memory" in system_rules
    )
    return ok, "belief confined to advisory section"


async def _scenario_12_project_over_memory(ctx: SimpleNamespace) -> tuple[bool, str]:
    """Explicit project rejected idea outranks the older Graphiti belief."""

    from research_radar.memory.models import MemoryFact

    ctx.repository.create_project(
        "GAN Robustness",
        rejected_ideas=["Pure GAN pipelines for MRI reconstruction"],
    )
    older_belief = MemoryFact(
        fact="User believes pure GAN pipelines always win for MRI reconstruction",
        valid_at=datetime(2020, 1, 1, tzinfo=UTC),
        source="graphiti",
    )
    stack = _build_stack(ctx, user_memory=build_fake_user_memory(older_belief))
    request = ctx.ChatRequest(
        text="what should we try next on this project?",
        project_hint="GAN Robustness",
    )
    response = await stack.service.chat(request)
    prompt_text = joined_prompt_text(stack.llm.last_messages)
    project_section = prompt_section(prompt_text, PROMPT_PROJECT_HEADER)
    advisory_section = prompt_section(prompt_text, PROMPT_USER_MEMORY_HEADER)
    position_memory = prompt_text.find(PROMPT_USER_MEMORY_HEADER)
    position_project = prompt_text.find(PROMPT_PROJECT_HEADER)
    position_question = prompt_text.find(PROMPT_QUESTION_HEADER)
    ordered = -1 < position_memory < position_project < position_question
    system_rules = prompt_section(prompt_text, PROMPT_SYSTEM_HEADER).casefold()
    ok = (
        response.mode is ctx.ChatMode.PROJECT_RESEARCH
        and "Pure GAN pipelines for MRI reconstruction" in project_section
        and "pure GAN pipelines always win" in advisory_section
        and "conflict" in system_rules
        and ordered
    )
    return ok, "project state marked canonical and outranking"


SCENARIOS = (
    ("hello gets conversational reply with zero backends", _scenario_01_hello),
    ("memory question uses seeded preferences without citations", _scenario_02_preferences),
    ("new topic discovers and persists canonical papers", _scenario_03_live_discovery),
    ("repeat query reuses stored results without duplicates", _scenario_04_repeat_query),
    ("user-memory outage still answers without memory", _scenario_05_memory_outage),
    ("semantic outage preserves lexical retrieval and discovery", _scenario_06_semantic_outage),
    ("LLM outage degrades safely with zero episodes", _scenario_07_llm_outage),
    ("guild message without mention triggers nothing", _scenario_08_no_mention),
    ("message from another bot is ignored", _scenario_09_other_bot),
    ("secret-bearing message never captured or logged", _scenario_10_secret),
    ("personal belief confined to advisory prompt section", _scenario_11_belief_placement),
    ("explicit project state outranks Graphiti fact", _scenario_12_project_over_memory),
)


class ProductionModules(SimpleNamespace):
    """Namespace bundling the lazily imported production classes."""


def load_production_modules() -> ProductionModules:
    """Import every production module this smoke script depends on.

    Any missing module (feature branch not integrated yet) raises ImportError,
    which the caller turns into a SKIP.
    """

    from research_radar.bot.mention import MentionPolicy
    from research_radar.chat.models import ChatMode, ChatRequest
    from research_radar.chat.router import ChatRouter
    from research_radar.chat.service import ChatService
    from research_radar.memory.capture import MemoryCapturePolicy
    from research_radar.memory.fakes import FakeUserMemoryStore
    from research_radar.memory.models import MemoryFact

    from research_radar.config import Settings
    from research_radar.models.paper import Paper
    from research_radar.research.ingestion import IngestionService
    from research_radar.research.scout import ScoutService
    from research_radar.storage.database import Database, initialize_schema
    from research_radar.storage.ingestion_repository import IngestionRepository
    from research_radar.storage.repositories import ResearchRepository

    return ProductionModules(
        Database=Database,
        initialize_schema=initialize_schema,
        ResearchRepository=ResearchRepository,
        IngestionRepository=IngestionRepository,
        ChatRequest=ChatRequest,
        ChatMode=ChatMode,
        Paper=Paper,
        MentionPolicy=MentionPolicy,
        Settings=Settings,
        MemoryCapturePolicy=MemoryCapturePolicy,
        MemoryFact=MemoryFact,
        FakeUserMemoryStore=FakeUserMemoryStore,
        ChatRouter=ChatRouter,
        ChatService=ChatService,
        IngestionService=IngestionService,
        ScoutService=ScoutService,
    )


async def execute_all_scenarios(database: Any, production: ProductionModules) -> list:
    """Execute every scenario against one shared fresh database in order."""

    repository = production.ResearchRepository(database)
    ingestion_repository = production.IngestionRepository(database)
    ctx = SimpleNamespace(
        database=database,
        repository=repository,
        ingestion_repository=ingestion_repository,
        capture_policy=production.MemoryCapturePolicy(),
        mention_policy=production.MentionPolicy(production.Settings()),
        ChatRequest=production.ChatRequest,
        ChatMode=production.ChatMode,
        Paper=production.Paper,
        stack=None,
    )

    results = []
    for name, scenario in SCENARIOS:
        try:
            outcome = scenario(ctx)
            if asyncio.iscoroutine(outcome):
                outcome = await outcome
            ok, detail = outcome
        except Exception as error:  # smoke harness reports any failure as FAIL
            ok, detail = False, f"{type(error).__name__}: {error}"
        results.append((name, ok, detail))
    return results


def run_smoke_tests(db_url: str | None = None) -> bool:
    """Run every offline scenario and return True only when all pass."""

    try:
        production = load_production_modules()
    except ImportError as error:
        print(f"[SKIP] Personal research chat modules not integrated yet ({error}).")
        print("[SKIP] Nothing to smoke test in this worktree; exiting successfully.")
        return True

    temporary_dir: tempfile.TemporaryDirectory | None = None
    if db_url is None:
        temporary_dir = tempfile.TemporaryDirectory(prefix="rr-chat-smoke-")
        db_url = f"sqlite:///{Path(temporary_dir.name) / 'chat_smoke.db'}"

    print("=" * 60)
    print(" ResearchRadar Personal Research Chat Smoke Test (offline)")
    print(f" Target Database: {db_url}")
    print("=" * 60)

    database = production.Database.create(db_url)
    production.initialize_schema(database)
    try:
        results = asyncio.run(execute_all_scenarios(database, production))
    finally:
        database.dispose()
        if temporary_dir is not None:
            temporary_dir.cleanup()

    passed = sum(1 for _name, ok, _detail in results if ok)
    for index, (name, ok, detail) in enumerate(results, start=1):
        marker = "PASS" if ok else "FAIL"
        print(f" [{index:2d}/{len(SCENARIOS)}] {name}: {marker} ({detail})")
    print("=" * 60)
    print(f" Results: {passed}/{len(SCENARIOS)} scenarios passed.")
    print("=" * 60)
    return passed == len(SCENARIOS)


def main() -> None:
    """Parse arguments, run the offline suite, and exit non-zero on failure."""

    parser = argparse.ArgumentParser(
        description="Run the offline personal research chat smoke test."
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="SQLite database URL (default: a throwaway file in a temporary directory). "
        "A fresh empty database is expected.",
    )
    args = parser.parse_args()

    success = run_smoke_tests(args.db_url)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
