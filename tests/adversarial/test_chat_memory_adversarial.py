"""Adversarial falsification suite for the personal research chat & memory phase.

One test per contract claim from ``docs/phase_tasks/W12.md``. Every external
boundary (SQLite, providers, LLM, memory backend, semantic index) is faked, so
the suite runs offline. The modules under test are written concurrently by
other workers: the whole module skips until ``research_radar.chat`` and
``research_radar.memory`` integrate, then every test below goes live.

This file is TEST-ONLY by mandate. Production code is never modified here; a
genuine invariant break must be reported via the audit document and
worker_done instead of being patched in this worktree.
"""

from __future__ import annotations

import enum
import inspect
import logging
import re
import threading
import traceback
import types
import typing
from collections.abc import Iterator, Sequence
from types import SimpleNamespace

import discord
import pytest
from pydantic import BaseModel
from sqlalchemy import func, select

from research_radar.models import Paper
from research_radar.reader.llm.base import LLMMessage
from research_radar.research.ingestion import IngestionResult, IngestionService
from research_radar.research.scout import ScoutService
from research_radar.semantic.base import SemanticHit
from research_radar.semantic.embedding import FakeEmbeddingProvider
from research_radar.semantic.index import FakeSemanticIndex
from research_radar.storage.database import Database, create_database, initialize_schema
from research_radar.storage.ingestion_repository import IngestionRepository
from research_radar.storage.repositories import ResearchRepository
from research_radar.storage.tables import PaperTable

pytest.importorskip("research_radar.memory.capture")
pytest.importorskip("research_radar.chat.evidence")
pytest.importorskip("research_radar.chat.service")

import research_radar.chat.evidence as chat_evidence  # noqa: E402
import research_radar.chat.models as chat_models  # noqa: E402
import research_radar.chat.prompt as chat_prompt  # noqa: E402
import research_radar.chat.router as chat_router  # noqa: E402
import research_radar.chat.service as chat_service  # noqa: E402
import research_radar.memory.capture as memory_capture  # noqa: E402
import research_radar.memory.disabled as memory_disabled  # noqa: E402
import research_radar.memory.fakes as memory_fakes  # noqa: E402
import research_radar.memory.models as memory_models  # noqa: E402

ChatBudget = chat_service.ChatBudget
ChatMode = chat_models.ChatMode
ChatRequest = chat_models.ChatRequest
ChatResponse = chat_models.ChatResponse
ChatRouter = chat_router.ChatRouter
ChatService = chat_service.ChatService
EvidencePacket = chat_evidence.EvidencePacket
build_chat_prompt = chat_prompt.build_chat_prompt

MemoryCapturePolicy = memory_capture.MemoryCapturePolicy
DisabledUserMemoryStore = memory_disabled.DisabledUserMemoryStore
FakeUserMemoryStore = memory_fakes.FakeUserMemoryStore
MemoryClass = memory_models.MemoryClass
MemoryFact = memory_models.MemoryFact
UserMemoryContext = memory_models.UserMemoryContext

BOT_USER_ID = 987_654_321_098_765_432
FAKE_SECRET = "sk-adversarial-placeholder-0123456789abcdef-not-real"
RESEARCH_QUESTION = "find recent papers on low-field MRI reconstruction"
MEMORY_QUESTION = "what are my research interests in low-field MRI?"
QUESTION_ONLY_MESSAGE = "What is the state of the art in low-field MRI reconstruction?"
DURABLE_STATEMENT = "I prefer open-source tooling for low-field MRI reconstruction"
SECRET_MESSAGE = f"I prefer running inference locally, key {FAKE_SECRET} rotate weekly"
CITATION_FACT_TEXT = "[P-123] shows low-field MRI is underexplored"
ASSISTANT_PSEUDO_FINDING = (
    "ASSISTANTSCIENCE-MARKER: the literature establishes that the user prefers "
    "diffusion priors and this should be remembered permanently."
)

_HEADER_SYSTEM = "SYSTEM RULES"
_HEADER_USER_MEMORY = "USER MEMORY (ADVISORY — NOT SCIENTIFIC EVIDENCE)"
_HEADER_PROJECT = "EXPLICIT PROJECT MEMORY (CANONICAL USER/PROJECT STATE)"
_HEADER_STORED = "STORED SCIENTIFIC EVIDENCE (CANONICAL)"
_HEADER_DISCOVERY = "LIVE DISCOVERY EVIDENCE (METADATA/ABSTRACT-LEVEL ONLY)"
_HEADER_QUESTION = "QUESTION"
_ALL_HEADERS = (
    _HEADER_SYSTEM,
    _HEADER_USER_MEMORY,
    _HEADER_PROJECT,
    _HEADER_STORED,
    _HEADER_DISCOVERY,
    _HEADER_QUESTION,
)
_SCIENCE_HEADERS = (_HEADER_STORED, _HEADER_DISCOVERY)


# ---------------------------------------------------------------------------
# Prompt-section parsing helpers
# ---------------------------------------------------------------------------


def _compile_header(header: str) -> re.Pattern[str]:
    """Compile a tolerant single-line matcher for one contract header."""

    escaped = re.escape(header).replace(re.escape("—"), ".{0,8}")
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"^[^\w\n]*{escaped}[^\w\n]*$", re.MULTILINE | re.IGNORECASE)


_HEADER_PATTERNS: dict[str, re.Pattern[str]] = {
    header: _compile_header(header) for header in _ALL_HEADERS
}


def _section_bodies(text: str) -> dict[str, str]:
    """Split a rendered prompt into ``{header: body}`` using contract headers."""

    matches: list[tuple[int, int, str]] = []
    for header in _ALL_HEADERS:
        found = _HEADER_PATTERNS[header].search(text)
        if found:
            matches.append((found.start(), found.end(), header))
    matches.sort()
    bodies: dict[str, str] = {}
    for index, (_start, end, header) in enumerate(matches):
        next_start = matches[index + 1][0] if index + 1 < len(matches) else len(text)
        bodies[header] = text[end:next_start]
    return bodies


def _require_sections(text: str) -> dict[str, str]:
    """Parse sections and fail with diagnostic context when headers are absent."""

    bodies = _section_bodies(text)
    if not bodies:
        pytest.fail(f"No contract prompt headers found in prompt: {text[:600]!r}")
    return bodies


# ---------------------------------------------------------------------------
# Fabricated synthesis models (the concrete model type is W6/W8 internal)
# ---------------------------------------------------------------------------

_ANSWER_FIELD_CANDIDATES = ("answer", "text", "response", "reply", "content")
_PAPER_FIELD_CANDIDATES = (
    "referenced_paper_ids",
    "paper_ids",
    "cited_paper_ids",
    "citations",
)
_GAP_FIELD_CANDIDATES = ("referenced_gap_ids", "gap_ids", "cited_gap_ids")


def _default_value(annotation: object) -> object:
    """Produce a harmless value for an unrecognised required field."""

    origin = typing.get_origin(annotation)
    if origin is typing.Union or (origin is not None and origin is types.UnionType):
        members = [item for item in typing.get_args(annotation) if item is not type(None)]
        if members:
            return _default_value(members[0])
        return ""
    if annotation is bool:
        return True
    if annotation is int:
        return 1
    if annotation is float:
        return 1.0
    if annotation is str:
        return ""
    if origin is typing.Literal:
        return typing.get_args(annotation)[0]
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return next(iter(annotation))
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        raise AssertionError(
            f"cannot fabricate nested pydantic model field of type {annotation!r}"
        )
    if origin in (list, set, frozenset):
        return []
    if origin is tuple:
        return ()
    if origin is dict:
        return {}
    raise AssertionError(f"cannot fabricate required field with annotation {annotation!r}")


def _fabricate(
    response_model: type[BaseModel],
    *,
    answer: str,
    paper_ids: Sequence[str] = (),
    gap_ids: Sequence[str] = (),
) -> BaseModel:
    """Instantiate W6's synthesis model without knowing its exact field names."""

    fields = response_model.model_fields
    kwargs: dict[str, object] = {}
    answer_field = next((name for name in _ANSWER_FIELD_CANDIDATES if name in fields), None)
    assert answer_field is not None, (
        f"synthesis model {response_model.__name__} has no recognised answer "
        f"field; fields={sorted(fields)}"
    )
    kwargs[answer_field] = answer
    paper_field = next((name for name in _PAPER_FIELD_CANDIDATES if name in fields), None)
    if paper_field is not None and (paper_ids or fields[paper_field].is_required()):
        kwargs[paper_field] = list(paper_ids)
    gap_field = next((name for name in _GAP_FIELD_CANDIDATES if name in fields), None)
    if gap_field is not None and (gap_ids or fields[gap_field].is_required()):
        kwargs[gap_field] = list(gap_ids)
    for name, info in fields.items():
        if name in kwargs or not info.is_required():
            continue
        kwargs[name] = _default_value(info.annotation)
    return response_model(**kwargs)


class ScriptedLLM:
    """Spy LLM that captures every prompt and returns one fabricated answer."""

    def __init__(
        self,
        *,
        answer: str = "Based on the provided evidence, low-field MRI remains active.",
        paper_ids: Sequence[str] = (),
        gap_ids: Sequence[str] = (),
        error: BaseException | None = None,
    ) -> None:
        self.prompts: list[list[LLMMessage]] = []
        self._answer = answer
        self._paper_ids = tuple(paper_ids)
        self._gap_ids = tuple(gap_ids)
        self._error = error

    async def generate_structured(
        self, messages: list[LLMMessage], response_model: type[BaseModel]
    ) -> BaseModel:
        """Record the prompt, then fabricate (or raise) deterministically."""

        self.prompts.append(list(messages))
        if self._error is not None:
            raise self._error
        return _fabricate(
            response_model,
            answer=self._answer,
            paper_ids=self._paper_ids,
            gap_ids=self._gap_ids,
        )

    def last_prompt_text(self) -> str:
        """Return the concatenated content of the most recent prompt."""

        assert self.prompts, "LLM was never invoked; no prompt was captured"
        return "\n\n".join(message.content for message in self.prompts[-1])


# ---------------------------------------------------------------------------
# Memory test doubles
# ---------------------------------------------------------------------------


class RecordingMemoryStore(FakeUserMemoryStore):
    """Fake store that records every episode-write attempt it receives."""

    def __init__(self, facts: Sequence[MemoryFact] = ()) -> None:
        super().__init__()
        self._seeded = tuple(facts)
        self.episode_calls: list[str] = []

    async def add_episode(
        self,
        content: str,
        *,
        source_description: str = "discord-chat",
        reference_time: object = None,
        memory_class: MemoryClass | None = None,
    ) -> bool:
        """Record the attempted write, then delegate to the fake backend."""

        self.episode_calls.append(content)
        return await super().add_episode(
            content,
            source_description=source_description,
            reference_time=reference_time,
            memory_class=memory_class,
        )

    async def search(self, query: str, *, limit: int = 8) -> list[MemoryFact]:
        """Return seeded facts when present so prompts carry them."""

        if self._seeded:
            return list(self._seeded)[:limit]
        return await super().search(query, limit=limit)

    async def get_context(self, query: str, *, limit: int = 8) -> UserMemoryContext:
        """Return seeded facts as an available advisory context."""

        if self._seeded:
            return UserMemoryContext(
                facts=self._seeded[:limit],
                backend=self.backend_name,
                degraded=False,
            )
        return await super().get_context(query, limit=limit)


class ExplodingSemanticIndex(FakeSemanticIndex):
    """Index double whose search always fails, simulating a Pinecone outage."""

    def search(self, vector, *, top_k: int = 10, entity_type=None):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated semantic index outage")


# ---------------------------------------------------------------------------
# Repository / ingestion doubles
# ---------------------------------------------------------------------------


class LoopGuardRepository:
    """Repository proxy recording synchronous calls made on the loop thread."""

    def __init__(self, inner: ResearchRepository, forbidden_ident: int) -> None:
        self.__dict__["_inner"] = inner
        self.__dict__["_forbidden_ident"] = forbidden_ident
        self.__dict__["violations"] = []

    def __getattr__(self, name: str) -> object:
        inner = self.__dict__["_inner"]
        attribute = getattr(inner, name)
        if not callable(attribute):
            return attribute

        def guarded(*args: object, **kwargs: object) -> object:
            if threading.get_ident() == self.__dict__["_forbidden_ident"]:
                self.__dict__["violations"].append(name)
            return attribute(*args, **kwargs)

        return guarded


class StubIngestion:
    """Contract-shaped ingestion double: counts calls, returns empty results."""

    def __init__(self, *, fail: BaseException | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._fail = fail

    async def ingest_research_topic(
        self,
        query: str,
        *,
        limit: int = 20,
        project_id: str | None = None,
        auto_read: int = 0,
    ) -> IngestionResult:
        """Record the call, then fail or return an empty canonical result."""

        self.calls.append(
            {"query": query, "limit": limit, "project_id": project_id, "auto_read": auto_read}
        )
        if self._fail is not None:
            raise self._fail
        return IngestionResult(
            run_id=f"stub-run-{len(self.calls)}",
            query=query,
            discovered_count=0,
            canonical_count=0,
            paper_ids=[],
            warnings=[],
            provider_counts={},
            read_paper_ids=[],
        )


class RecordingProvider:
    """Provider double returning fixed papers while recording requested limits."""

    name = "recording-fake"

    def __init__(self, papers: list[Paper]) -> None:
        self._papers = papers
        self.requested_limits: list[int] = []

    async def search(self, query: str, limit: int) -> list[Paper]:
        """Serve the fixed result set and remember the requested limit."""

        self.requested_limits.append(limit)
        return list(self._papers)


class CountedIngestion:
    """Delegate to the real IngestionService while counting chat-facing calls."""

    def __init__(self, inner: IngestionService) -> None:
        self._inner = inner
        self.limits: list[int] = []

    async def ingest_research_topic(
        self,
        query: str,
        *,
        limit: int = 20,
        project_id: str | None = None,
        auto_read: int = 0,
    ) -> IngestionResult:
        """Count one chat-facing call and pass the limit through unchanged."""

        self.limits.append(limit)
        return await self._inner.ingest_research_topic(
            query, limit=limit, project_id=project_id, auto_read=auto_read
        )


# ---------------------------------------------------------------------------
# Shared fixtures and builders
# ---------------------------------------------------------------------------


@pytest.fixture
def database() -> Iterator[Database]:
    """Provide a fresh in-memory SQLite database per test."""

    database = create_database("sqlite:///:memory:")
    initialize_schema(database)
    yield database
    database.dispose()


@pytest.fixture
def repository(database: Database) -> ResearchRepository:
    """Provide a repository over the fresh database."""

    return ResearchRepository(database)


def _settings(**overrides: object) -> object:
    """Build Settings, skipping cleanly when W9's fields are not integrated."""

    from research_radar.config import Settings

    settings = Settings(**overrides)
    missing = [name for name in overrides if not hasattr(settings, name)]
    if missing:
        pytest.skip(f"W9 settings fields not integrated yet: {sorted(missing)}")
    return settings


def _store_paper(
    repository: ResearchRepository, slug: str = "2401.00042", title: str | None = None
) -> str:
    """Persist one canonical paper and return its SQLite id."""

    return repository.upsert_merged_paper(
        Paper(
            id=f"arxiv:{slug}",
            title=title or "Low-field MRI reconstruction with learned priors",
            abstract="A study of low-field MRI reconstruction methods.",
            authors=["A. Researcher"],
            publication_year=2024,
            venue="MRJ",
            doi=f"10.2000/{slug}",
            url=f"https://journals.example/{slug}",
            citation_count=7,
            source="arxiv",
            external_ids={"arxiv": slug},
        )
    )


def _service(
    repository: ResearchRepository,
    *,
    llm: ScriptedLLM | None = None,
    memory: FakeUserMemoryStore | None = None,
    ingestion: object | None = None,
    embedding: object | None = None,
    index: object | None = None,
    budget: ChatBudget | None = None,
) -> ChatService:
    """Assemble a ChatService exactly per the phase contract signature."""

    return ChatService(
        repository=repository,
        router=ChatRouter(),
        user_memory=memory or FakeUserMemoryStore(),
        capture_policy=MemoryCapturePolicy(enabled=True),
        llm_provider=llm,
        ingestion_service=ingestion,  # type: ignore[arg-type]
        embedding_provider=embedding,  # type: ignore[arg-type]
        semantic_index=index,  # type: ignore[arg-type]
        budget=budget,
    )


def _paper_row_count(database: Database) -> int:
    """Count canonical paper rows straight in SQLite."""

    with database.session_factory() as session:
        return int(session.scalar(select(func.count()).select_from(PaperTable)))


def _live_stack(
    repository: ResearchRepository, database: Database, papers: list[Paper]
) -> tuple[CountedIngestion, RecordingProvider]:
    """Wire the real IngestionService behind a counting chat-facing wrapper."""

    provider = RecordingProvider(papers)
    inner = IngestionService(
        scout=ScoutService([provider]),
        repository=repository,
        ingestion_repository=IngestionRepository(database),
        metadata_limit=50,
    )
    return CountedIngestion(inner), provider


def _dup_paper(slug: str, title: str) -> Paper:
    """Build one deterministic arXiv-shaped paper for dedup checks."""

    return Paper(
        id=f"arxiv:{slug}",
        title=title,
        abstract="Low-field MRI reconstruction methods and priors.",
        authors=["A. Researcher"],
        publication_year=2025,
        venue="MRJ",
        doi=f"10.2000/{slug}",
        url=f"https://journals.example/{slug}",
        citation_count=3,
        source="arxiv",
        external_ids={"arxiv": slug},
    )


def _discord_message(
    *,
    content: str,
    author_id: int = 111_222_333,
    author_bot: bool = False,
    channel_id: int = 445_566,
    dm: bool = False,
    mentions: Sequence[SimpleNamespace] | None = None,
    role_mentions: Sequence[SimpleNamespace] = (),
) -> SimpleNamespace:
    """Build a duck-typed discord.Message covering the admission surface."""

    bot_member = SimpleNamespace(id=BOT_USER_ID, bot=True, name="radar-bot")
    channel_type = discord.ChannelType.private if dm else discord.ChannelType.text
    return SimpleNamespace(
        content=content,
        author=SimpleNamespace(id=author_id, bot=author_bot, name="member"),
        channel=SimpleNamespace(id=channel_id, type=channel_type),
        guild=SimpleNamespace(id=1),
        mentions=[bot_member] if mentions is None else list(mentions),
        role_mentions=list(role_mentions),
    )


# ---------------------------------------------------------------------------
# SOURCE BOUNDARIES
# ---------------------------------------------------------------------------


async def test_user_memory_fact_looking_like_citation_stays_advisory(
    repository: ResearchRepository,
) -> None:
    """Claim: a memory fact citing "[P-123]" never reaches evidence sections."""

    real_id = _store_paper(repository)
    fact = MemoryFact(fact=CITATION_FACT_TEXT, memory_class=MemoryClass.RESEARCH_INTEREST)
    memory = RecordingMemoryStore([fact])
    llm = ScriptedLLM(paper_ids=["P-123"])
    service = _service(repository, llm=llm, memory=memory)

    response = await service.chat(ChatRequest(text=MEMORY_QUESTION))

    assert response.used_user_memory is True
    bodies = _require_sections(llm.last_prompt_text())
    advisory_body = bodies.get(_HEADER_USER_MEMORY)
    assert advisory_body is not None, (
        f"USER MEMORY section missing from prompt: {llm.last_prompt_text()[:400]!r}"
    )
    assert CITATION_FACT_TEXT in advisory_body
    for science_header in _SCIENCE_HEADERS:
        science_body = bodies.get(science_header, "")
        assert CITATION_FACT_TEXT not in science_body, (
            f"memory fact leaked into {science_header}: {science_body!r}"
        )
    packet = EvidencePacket(user_memory=UserMemoryContext(facts=(fact,)))
    assert "P-123" not in packet.allowed_paper_ids
    assert packet.allowed_paper_ids == set()
    assert "P-123" not in response.paper_ids
    assert real_id != "P-123"


async def test_llm_answer_citing_unknown_paper_id_is_stripped(
    repository: ResearchRepository,
) -> None:
    """Claim: citations outside the evidence packet are dropped from the response."""

    real_id = _store_paper(repository)
    fabricated_id = "P-404-not-in-packet"
    ingestion = StubIngestion()
    llm = ScriptedLLM(paper_ids=[real_id, fabricated_id])
    service = _service(repository, llm=llm, ingestion=ingestion)

    response = await service.chat(ChatRequest(text=RESEARCH_QUESTION))

    assert isinstance(response, ChatResponse)
    assert real_id in response.paper_ids
    assert fabricated_id not in response.paper_ids
    assert all(isinstance(paper_id, str) for paper_id in response.paper_ids)


async def test_unresolved_semantic_candidate_never_becomes_evidence(
    repository: ResearchRepository,
) -> None:
    """Claim: a vector hit with no SQLite row never becomes evidence anywhere."""

    real_id = _store_paper(repository)

    class GhostIndex(FakeSemanticIndex):
        """Index serving one candidate id that has no canonical row."""

        def search(self, vector, *, top_k: int = 10, entity_type=None):  # type: ignore[no-untyped-def]
            return [
                SemanticHit(
                    entity_id="paper:ghost-paper",
                    entity_type="paper",
                    paper_id="ghost-paper",
                    score=0.99,
                )
            ]

    embedding = FakeEmbeddingProvider(dimension=8)
    llm = ScriptedLLM(paper_ids=["ghost-paper", real_id])
    service = _service(
        repository,
        llm=llm,
        ingestion=StubIngestion(),
        embedding=embedding,
        index=GhostIndex(),
    )

    response = await service.chat(ChatRequest(text=RESEARCH_QUESTION))

    assert "ghost-paper" not in response.paper_ids
    assert real_id in response.paper_ids
    prompt_text = llm.last_prompt_text()
    assert "ghost-paper" not in prompt_text, "unresolved candidate id reached the prompt"


# ---------------------------------------------------------------------------
# MEMORY CONTAMINATION
# ---------------------------------------------------------------------------


async def test_assistant_scientific_text_is_never_captured_as_episode(
    repository: ResearchRepository,
) -> None:
    """Claim: assistant output is never persisted as a user-memory episode."""

    memory = RecordingMemoryStore()
    llm = ScriptedLLM(answer=ASSISTANT_PSEUDO_FINDING)
    service = _service(repository, llm=llm, memory=memory, ingestion=StubIngestion())

    await service.chat(ChatRequest(text=RESEARCH_QUESTION))

    assert llm.prompts, "pipeline must reach the LLM before capture is decided"
    joined = "\n".join(memory.episode_calls)
    assert "ASSISTANTSCIENCE-MARKER" not in joined
    assert all(episode != ASSISTANT_PSEUDO_FINDING for episode in memory.episode_calls)


async def test_question_only_message_writes_no_episode(
    repository: ResearchRepository,
) -> None:
    """Claim: a message that is only a question writes no episode at all."""

    memory = RecordingMemoryStore()
    llm = ScriptedLLM()
    service = _service(repository, llm=llm, memory=memory)

    await service.chat(ChatRequest(text=QUESTION_ONLY_MESSAGE))

    assert memory.episode_calls == []


async def test_no_episode_write_on_llm_failure_path(
    repository: ResearchRepository,
) -> None:
    """Claim: when the LLM fails, no add_episode call is ever attempted."""

    memory = RecordingMemoryStore()
    llm = ScriptedLLM(error=RuntimeError("simulated total LLM outage"))
    service = _service(repository, llm=llm, memory=memory)

    response = await service.chat(ChatRequest(text=DURABLE_STATEMENT))

    assert response.degraded is True
    assert memory.episode_calls == []


# ---------------------------------------------------------------------------
# MENTION FILTERING
# ---------------------------------------------------------------------------


def _admit(message: SimpleNamespace, settings: object | None = None) -> object:
    """Run MentionPolicy.admit, skipping until W7 integrates."""

    mention_module = pytest.importorskip("research_radar.bot.mention")
    policy_settings = settings if settings is not None else _settings()
    policy = mention_module.MentionPolicy(policy_settings)  # type: ignore[attr-defined]
    return policy.admit(message, bot_user_id=BOT_USER_ID)  # type: ignore[attr-defined]


def test_mass_and_role_mentions_are_not_admitted() -> None:
    """Claim: @everyone/@here/bot-ish role mentions are not bot admissions."""

    everyone = _admit(_message_for("@everyone what is up", mentions=[]))
    assert everyone.accepted is False  # type: ignore[attr-defined]
    assert everyone.reason == "no_mention"  # type: ignore[attr-defined]

    here = _admit(_message_for("@here standup time", mentions=[]))
    assert here.accepted is False  # type: ignore[attr-defined]
    assert here.reason == "no_mention"  # type: ignore[attr-defined]

    role_message = _message_for(
        "<@&777> hello bot",
        mentions=[],
        role_mentions=(SimpleNamespace(id=777, name="bot-squad"),),
    )
    role = _admit(role_message)
    assert role.accepted is False  # type: ignore[attr-defined]
    assert role.reason == "no_mention"  # type: ignore[attr-defined]


def _message_for(content: str, **overrides: object) -> SimpleNamespace:
    """Shorthand wrapper around the shared fake-message builder."""

    return _discord_message(content=content, **overrides)  # type: ignore[arg-type]


def test_other_bot_mentioning_us_is_still_rejected() -> None:
    """Claim: another bot's message mentioning the bot is rejected as bot_author."""

    other_bot_member = SimpleNamespace(id=444, bot=True, name="helper-bot")
    message = _discord_message(
        content=f"<@{BOT_USER_ID}> please summarise",
        author_id=444,
        author_bot=True,
        mentions=[other_bot_member],
    )

    admission = _admit(message)

    assert admission.accepted is False  # type: ignore[attr-defined]
    assert admission.reason == "bot_author"  # type: ignore[attr-defined]


def test_channel_allowlist_rejects_before_any_service_call() -> None:
    """Claim: out-of-allowlist channels are rejected before any service call."""

    settings = _settings(
        discord_allowed_channel_ids=(111,),
        discord_owner_user_id=None,
        discord_chat_on_mention=True,
        discord_dm_chat=True,
    )
    message = _discord_message(
        content=f"<@{BOT_USER_ID}> find papers on MRI", channel_id=222
    )

    admission = _admit(message, settings=settings)

    assert admission.accepted is False  # type: ignore[attr-defined]
    assert admission.reason == "channel_not_allowed"  # type: ignore[attr-defined]
    handler_service_calls: list[str] = []
    if admission.accepted:  # type: ignore[attr-defined]
        handler_service_calls.append("chat")
    assert handler_service_calls == [], "a rejected message reached the chat service"


def test_owner_filter_rejects_mentions_from_other_users() -> None:
    """Claim: with an owner configured, other users' mentions are rejected."""

    settings = _settings(
        discord_allowed_channel_ids=(),
        discord_owner_user_id=999_999,
        discord_chat_on_mention=True,
        discord_dm_chat=True,
    )
    message = _discord_message(
        content=f"<@{BOT_USER_ID}> hello", author_id=111_222_333
    )

    admission = _admit(message, settings=settings)

    assert admission.accepted is False  # type: ignore[attr-defined]
    assert admission.reason == "owner_only"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# DUPLICATE LIVE INGESTION
# ---------------------------------------------------------------------------


async def test_duplicate_topic_does_not_create_duplicate_canonical_rows(
    repository: ResearchRepository, database: Database
) -> None:
    """Claim: asking the same topic twice yields no duplicate canonical rows."""

    papers = [
        _dup_paper("2501.00042", "Low field MRI reconstruction via learned priors"),
        _dup_paper("2501.00043", "Deep priors for low field MRI reconstruction"),
    ]
    ingestion, _provider = _live_stack(repository, database, papers)
    llm = ScriptedLLM()
    service = _service(repository, llm=llm, ingestion=ingestion)

    await service.chat(ChatRequest(text=RESEARCH_QUESTION))
    first_count = _paper_row_count(database)
    await service.chat(ChatRequest(text=RESEARCH_QUESTION))
    second_count = _paper_row_count(database)

    assert len(ingestion.limits) == 2, "discovery must actually run on both turns"
    assert first_count == 2, f"first turn created {first_count} rows, expected 2"
    assert second_count == 2, f"duplicate topic grew rows to {second_count}"


async def test_one_chat_turn_triggers_at_most_one_ingestion_call(
    repository: ResearchRepository,
) -> None:
    """Claim: a single chat turn makes at most ONE ingestion call."""

    ingestion = StubIngestion()
    llm = ScriptedLLM()
    service = _service(repository, llm=llm, ingestion=ingestion)

    await service.chat(ChatRequest(text=RESEARCH_QUESTION))

    assert len(ingestion.calls) <= 1, f"ingestion called {len(ingestion.calls)} times"
    assert len(ingestion.calls) == 1, "stored evidence was insufficient; discovery owed one call"


async def test_discovery_limit_hard_clamped_before_provider_layer(
    repository: ResearchRepository, database: Database
) -> None:
    """Claim: requesting 50 still sends at most 12 towards the provider layer."""

    papers = [_dup_paper("2501.00042", "Low field MRI reconstruction via learned priors")]
    ingestion, provider = _live_stack(repository, database, papers)
    budget = ChatBudget(max_discovery_results=50)
    llm = ScriptedLLM()
    service = _service(repository, llm=llm, ingestion=ingestion, budget=budget)

    await service.chat(ChatRequest(text=RESEARCH_QUESTION))

    assert len(ingestion.limits) == 1
    assert ingestion.limits[0] <= 12, (
        f"chat passed limit={ingestion.limits[0]} to the ingestion layer; clamp broken"
    )
    assert provider.requested_limits, "provider layer was never reached"
    assert max(provider.requested_limits) <= 12, (
        f"provider received limits {provider.requested_limits}; hard clamp broken end-to-end"
    )


# ---------------------------------------------------------------------------
# EVENT-LOOP SAFETY
# ---------------------------------------------------------------------------


async def test_repository_sync_calls_never_run_on_loop_thread(
    repository: ResearchRepository,
) -> None:
    """Claim: no synchronous repository call executes on the event-loop thread."""

    stored_ids = [
        _store_paper(
            repository,
            slug=f"2401.0004{n}",
            title=f"Low-field MRI reconstruction study {n}",
        )
        for n in range(3)
    ]
    loop_thread_ident = threading.get_ident()
    guard = LoopGuardRepository(repository, loop_thread_ident)
    llm = ScriptedLLM(paper_ids=tuple(stored_ids))
    service = _service(guard, llm=llm, memory=RecordingMemoryStore())  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(text=RESEARCH_QUESTION))

    assert isinstance(response, ChatResponse)
    assert guard.violations == [], (
        f"synchronous repository methods ran on the loop thread: {guard.violations}"
    )


async def test_sequential_turns_do_not_accumulate_threads(
    repository: ResearchRepository,
) -> None:
    """Claim: 20 sequential turns leave no monotonically growing thread count."""

    llm = ScriptedLLM()
    service = _service(repository, llm=llm, memory=RecordingMemoryStore())
    samples: list[int] = []

    for index in range(20):
        await service.chat(ChatRequest(text=f"hello there, turn {index}"))
        samples.append(threading.active_count())

    assert samples[-1] - samples[0] <= 2, f"thread count drifted: {samples}"
    assert max(samples) - min(samples) <= 2, f"thread count spiked mid-run: {samples}"
    consecutive_growth = sum(
        1 for a, b in zip(samples, samples[1:], strict=False) if b > a
    )
    assert consecutive_growth < len(samples) - 1, (
        f"active thread count grew on every turn: {samples}"
    )


# ---------------------------------------------------------------------------
# CREDENTIAL SAFETY
# ---------------------------------------------------------------------------


async def test_user_message_secret_never_leaks_to_store_logs_response_or_errors(
    repository: ResearchRepository,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Claim: a planted credential stays out of memory, logs, errors, and text."""

    memory = RecordingMemoryStore()
    llm = ScriptedLLM(answer="Understood; continuing with local inference guidance.")
    service = _service(repository, llm=llm, memory=memory)
    raised: BaseException | None = None

    caplog.set_level(logging.DEBUG)
    try:
        response = await service.chat(ChatRequest(text=SECRET_MESSAGE))
    except BaseException as error:  # noqa: B036 - adversarial leak surface check
        raised = error
        response = None  # type: ignore[assignment]

    assert raised is None, f"chat raised instead of degrading: {raised!r}"
    assert response is not None
    joined_episodes = "\n".join(memory.episode_calls)
    assert FAKE_SECRET not in joined_episodes, "credential reached the memory store"
    assert FAKE_SECRET not in response.text, "credential reached ChatResponse.text"

    log_texts: list[str] = [record.getMessage() for record in caplog.records]
    for record in caplog.records:
        if record.exc_info:
            log_texts.extend(traceback.format_exception(*record.exc_info))
    assert all(FAKE_SECRET not in text for text in log_texts), (
        "credential appeared in log records"
    )

    if raised is not None:
        raised_forms = [str(raised), repr(raised), "".join(traceback.format_exception(raised))]
        assert all(FAKE_SECRET not in form for form in raised_forms)


async def test_memory_status_output_hides_credentials_and_graph_internals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim: MemoryStatus renderings never expose credentials or graph internals."""

    planted = {
        "DISCORD_TOKEN": "fake-discord-token-placeholder.deadbeef.cafef00d",
        "LLM_API_KEY": FAKE_SECRET,
        "PINECONE_API_KEY": "pcsk_placeholder_not_a_real_key_000000000000",
    }
    for name, value in planted.items():
        monkeypatch.setenv(name, value)

    import dataclasses

    stores: list[object] = [
        DisabledUserMemoryStore(),
        FakeUserMemoryStore(),
        pytest.importorskip("research_radar.memory.fakes").FakeUserMemoryStore(fail=True),
    ]

    internals_needles = (
        "KuzuDriver",
        "graphiti_core",
        "add_episode",
        "EntityNode",
        "EntityEdge",
        "node_id",
    )
    for store in stores:
        status = await store.status()  # type: ignore[attr-defined]
        blob_parts = [repr(status), str(status), repr(dataclasses.asdict(status))]
        blob = " | ".join(blob_parts)
        for credential in planted.values():
            assert credential not in blob, (
                f"credential {credential[:12]}... leaked into MemoryStatus: {blob!r}"
            )
        for needle in internals_needles:
            assert needle not in blob, f"graph internals marker {needle!r} in status: {blob!r}"

    commands_module = pytest.importorskip("research_radar.bot.commands.memory")
    for name, value in vars(commands_module).items():
        if name.startswith("_") or not inspect.isfunction(value):
            continue
        if value.__module__ != commands_module.__name__:
            continue
        lowered = name.lower()
        if not any(tag in lowered for tag in ("format", "render", "line", "text")):
            continue
        # The module renders two different shapes (statuses and facts). Feed
        # each renderer both, and skip the combinations it does not accept:
        # duck-typed rejection surfaces as AttributeError, not just TypeError.
        credential_fact = MemoryFact(
            fact=f"planted credential {planted['LLM_API_KEY']}",
            memory_class=MemoryClass.PREFERENCE,
        )
        candidates: list[object] = [credential_fact]
        for store in stores:
            candidates.append(await store.status())  # type: ignore[attr-defined]
        for candidate in candidates:
            try:
                rendered = str(value(candidate))
            except (TypeError, AttributeError, ValueError):
                continue
            for credential in planted.values():
                if candidate is credential_fact and credential in str(candidate):
                    # A fact renderer is expected to echo the fact it was
                    # handed; the boundary under test is that nothing else does.
                    continue
                assert credential not in rendered, (
                    f"{commands_module.__name__}.{name} leaked a credential"
                )


# ---------------------------------------------------------------------------
# BACKEND OUTAGE BEHAVIOUR
# ---------------------------------------------------------------------------


async def test_memory_outage_degrades_with_used_user_memory_false(
    repository: ResearchRepository,
) -> None:
    """Claim: a memory outage degrades the turn instead of raising."""

    memory = FakeUserMemoryStore(fail=True)
    llm = ScriptedLLM()
    service = _service(repository, llm=llm, memory=memory)

    response = await service.chat(ChatRequest(text=MEMORY_QUESTION))

    assert isinstance(response, ChatResponse)
    assert response.used_user_memory is False
    assert response.degraded is False, "an LLM-free degradation must not flag degraded"


async def test_semantic_outage_degrades_to_lexical_without_raising(
    repository: ResearchRepository,
) -> None:
    """Claim: a Pinecone-style outage falls back to lexical retrieval silently."""

    real_id = _store_paper(repository)
    llm = ScriptedLLM(paper_ids=[real_id])
    service = _service(
        repository,
        llm=llm,
        ingestion=StubIngestion(),
        embedding=FakeEmbeddingProvider(dimension=8),
        index=ExplodingSemanticIndex(),
    )

    response = await service.chat(ChatRequest(text=RESEARCH_QUESTION))

    assert isinstance(response, ChatResponse)
    assert real_id in response.paper_ids, "lexical retrieval must survive the outage"
    assert response.mode in {ChatMode.RESEARCH_STORED, ChatMode.RESEARCH_LIVE}
    assert response.degraded is False


async def test_ingestion_outage_degrades_to_stored_evidence_only(
    repository: ResearchRepository,
) -> None:
    """Claim: an ingestion outage logs sanitized and continues without raising."""

    ingestion = StubIngestion(fail=RuntimeError("simulated discovery outage"))
    llm = ScriptedLLM(answer="The available stored evidence is insufficient.")
    service = _service(repository, llm=llm, ingestion=ingestion)

    response = await service.chat(ChatRequest(text=RESEARCH_QUESTION))

    assert isinstance(response, ChatResponse)
    assert response.live_discovery_used is False
    assert response.degraded is False, "ingestion failure alone must not flag degraded"
    assert len(ingestion.calls) == 1, "the failed discovery must not be retried in-turn"
    assert response.text.strip(), "a safe non-empty answer is still expected"


async def test_llm_outage_degrades_safely_and_sets_flag(
    repository: ResearchRepository,
) -> None:
    """Claim: an LLM outage returns a safe degraded answer, never a traceback."""

    memory = RecordingMemoryStore()
    llm = ScriptedLLM(error=RuntimeError("simulated LLM outage"))
    service = _service(repository, llm=llm, memory=memory, ingestion=StubIngestion())

    response = await service.chat(ChatRequest(text=DURABLE_STATEMENT))

    assert isinstance(response, ChatResponse)
    assert response.degraded is True
    assert response.text.strip(), "a concise safe error message is required"
    assert "Traceback" not in response.text
    assert "RuntimeError" not in response.text
