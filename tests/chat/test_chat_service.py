"""Tests for ChatService: pipeline order, budgets, outages, and capture rules."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest

from research_radar.chat.models import ChatMode, ChatRequest, EvidenceScope
from research_radar.chat.router import ChatRouter
from research_radar.chat.service import ChatBudget, ChatService
from research_radar.memory.capture import MemoryCapturePolicy
from research_radar.memory.fakes import FakeUserMemoryStore
from research_radar.memory.models import MemoryClass, MemoryFact
from research_radar.models import Paper
from research_radar.models.gap import CandidateGap, GapProvenance
from research_radar.models.paper_card import EvidenceClaim, PaperCard
from research_radar.reader.llm.base import LLMMessage
from research_radar.research.ingestion import IngestionResult
from research_radar.semantic.base import SemanticRecord
from research_radar.semantic.embedding import FakeEmbeddingProvider
from research_radar.semantic.index import FakeSemanticIndex
from research_radar.storage.database import Database, create_database, initialize_schema
from research_radar.storage.repositories import ResearchRepository

DIMENSION = 8
RESEARCH_QUERY = "find recent papers on diffusion policy"
MEMORY_QUESTION = "what are my interests?"
DURABLE_STATEMENT = "I prefer smaller open-source models."
SMALL_TALK = "thanks bot!"


# ---------------------------------------------------------------------------
# Fixtures and doubles.
# ---------------------------------------------------------------------------


@pytest.fixture
def database() -> Iterator[Database]:
    database = create_database("sqlite:///:memory:")
    initialize_schema(database)
    yield database
    database.dispose()


@pytest.fixture
def repository(database: Database) -> ResearchRepository:
    return ResearchRepository(database)


class RecordingIngestionService:
    """Duck-typed IngestionService double that records calls like the real one."""

    def __init__(
        self,
        repository: ResearchRepository | None = None,
        pending_papers: Iterator[Paper] | list[Paper] = (),
        *,
        exc: Exception | None = None,
    ) -> None:
        self._repository = repository
        self._pending = list(pending_papers)
        self._exc = exc
        self.calls: list[dict[str, Any]] = []
        self.persisted_ids: list[str] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def ingest_research_topic(
        self,
        query: str,
        *,
        limit: int = 20,
        project_id: str | None = None,
        auto_read: int = 0,
    ) -> IngestionResult:
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "project_id": project_id,
                "auto_read": auto_read,
            }
        )
        if self._exc is not None:
            raise self._exc
        paper_ids: list[str] = []
        if self._repository is not None:
            for paper in self._pending:
                paper_ids.append(
                    await asyncio.to_thread(self._repository.upsert_merged_paper, paper)
                )
        self.persisted_ids.extend(paper_ids)
        return IngestionResult(
            run_id="run-chat-test",
            query=query,
            discovered_count=len(paper_ids),
            canonical_count=len(paper_ids),
            paper_ids=paper_ids,
            warnings=[],
            provider_counts={},
            read_paper_ids=[],
        )


class ScriptedChatLLM:
    """Deterministic LLM double returning a fixed payload or raising.

    With ``cite_ids_from_prompt=True`` the double extracts every bracketed
    ``[id]`` evidence marker from the rendered prompt and cites it, mimicking a
    well-behaved model that only cites listed evidence.
    """

    # The integrated prompt builder (research_radar.chat.prompt) labels
    # evidence blocks with the established /ask convention.
    _EVIDENCE_ID_RE = re.compile(r"(?m)^--- Paper (\S+) ---$")

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        exc: Exception | None = None,
        cite_ids_from_prompt: bool = False,
    ) -> None:
        self._payload = payload
        self._exc = exc
        self._cite_from_prompt = cite_ids_from_prompt
        self.calls = 0
        self.last_messages: list[LLMMessage] = []

    async def generate_structured(
        self, messages: list[LLMMessage], response_model: type[Any]
    ) -> Any:
        self.calls += 1
        self.last_messages = list(messages)
        if self._exc is not None:
            raise self._exc
        if self._cite_from_prompt:
            cited = self._EVIDENCE_ID_RE.findall(self.last_messages[-1].content)
            return response_model.model_validate(
                {
                    "answer": "Cited from the supplied evidence.",
                    "referenced_paper_ids": cited,
                    "referenced_gap_ids": [],
                }
            )
        return response_model.model_validate(
            self._payload
            or {
                "answer": "Synthesized from the supplied evidence.",
                "referenced_paper_ids": [],
                "referenced_gap_ids": [],
            }
        )


class ExplodingRouter:
    """Route double that fails the test if routing ever runs."""

    def __init__(self) -> None:
        self.calls = 0

    async def route(self, request: ChatRequest) -> Any:
        self.calls += 1
        raise AssertionError("router.route must not be called")


class ExplodingLLM:
    """LLM double that fails the test if generate_structured ever runs."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(self, messages: list[LLMMessage], model: type) -> Any:
        self.calls += 1
        raise AssertionError("generate_structured must not be called")


class ExplodingCapturePolicy:
    """Capture-policy double that fails the test if evaluation ever runs."""

    def evaluate_user_message(self, text: str) -> Any:
        raise AssertionError("capture policy must not be evaluated")


def _service(
    repository: ResearchRepository,
    *,
    router: Any | None = None,
    memory: FakeUserMemoryStore | None = None,
    capture: Any | None = None,
    llm: Any | None = None,
    ingestion: RecordingIngestionService | None = None,
    embedding: Any | None = None,
    index: Any | None = None,
    budget: ChatBudget | None = None,
) -> ChatService:
    return ChatService(
        repository=repository,
        router=router or ChatRouter(),
        user_memory=memory or FakeUserMemoryStore(),
        capture_policy=capture or MemoryCapturePolicy(),
        llm_provider=llm,
        ingestion_service=ingestion,
        embedding_provider=embedding,
        semantic_index=index,
        budget=budget,
    )


def _paper(slug: str, title: str, abstract: str) -> Paper:
    return Paper(
        id=f"openalex:{slug}",
        title=title,
        abstract=abstract,
        authors=[],
        publication_year=2024,
        venue="Venue",
        doi=f"10.1000/{slug}",
        url=f"https://example.test/{slug}",
        citation_count=1,
        source="openalex",
        external_ids={"openalex": slug},
    )


def _seed_paper_with_card(
    repository: ResearchRepository,
    slug: str,
    claim: str = "Diffusion policies beat baselines.",
) -> str:
    paper_id = repository.upsert_merged_paper(
        _paper(slug, f"Diffusion policy study {slug}", "Advances in diffusion policy learning.")
    )
    repository.upsert_paper_card(
        PaperCard(
            paper_id=paper_id,
            problem="Robotic manipulation lacks robust policies.",
            main_claims=[EvidenceClaim(claim=claim)],
        )
    )
    return paper_id


def _discovery_batch(count: int, prefix: str = "fresh") -> list[Paper]:
    return [
        _paper(
            f"{prefix}{index}",
            f"Fresh diffusion policy report {index}",
            "Newly discovered diffusion policy findings.",
        )
        for index in range(count)
    ]


# ---------------------------------------------------------------------------
# Step 1: empty input short-circuits everything.
# ---------------------------------------------------------------------------


async def test_empty_text_returns_usage_hint_without_any_backend_call(
    repository: ResearchRepository,
) -> None:
    router = ExplodingRouter()
    llm = ExplodingLLM()
    service = _service(repository, router=router, llm=llm, capture=ExplodingCapturePolicy())

    response = await service.chat(ChatRequest(text="   "))

    assert response.mode is ChatMode.CONVERSATIONAL
    assert response.degraded is False
    assert response.paper_ids == ()
    assert "find recent papers" in response.text
    assert router.calls == 0
    assert llm.calls == 0


# ---------------------------------------------------------------------------
# Stored-sufficient path: no discovery at all.
# ---------------------------------------------------------------------------


async def test_stored_sufficient_path_skips_live_discovery(
    repository: ResearchRepository,
) -> None:
    stored_ids = [_seed_paper_with_card(repository, f"seed{i}") for i in range(3)]
    ingestion = RecordingIngestionService(repository)
    llm = ScriptedChatLLM()

    response = await _service(repository, llm=llm, ingestion=ingestion).chat(
        ChatRequest(text=RESEARCH_QUERY)
    )

    assert ingestion.call_count == 0
    assert response.mode is ChatMode.RESEARCH_STORED
    assert response.live_discovery_used is False
    assert response.degraded is False
    assert set(response.paper_ids) <= set(stored_ids)
    assert response.evidence_scope is EvidenceScope.STORED_CARDS


# ---------------------------------------------------------------------------
# Stored-insufficient path: exactly one discovery, ids re-resolved from SQLite.
# ---------------------------------------------------------------------------


async def test_stored_insufficient_path_runs_one_discovery_and_resolves_sqlite_ids(
    repository: ResearchRepository,
) -> None:
    ingestion = RecordingIngestionService(repository, _discovery_batch(3))
    llm = ScriptedChatLLM(cite_ids_from_prompt=True)

    response = await _service(repository, llm=llm, ingestion=ingestion).chat(
        ChatRequest(text=RESEARCH_QUERY)
    )

    assert ingestion.call_count == 1
    recorded = ingestion.calls[0]
    assert recorded["auto_read"] == 0
    assert isinstance(recorded["limit"], int) and 1 <= recorded["limit"] <= 12
    assert response.mode is ChatMode.RESEARCH_LIVE
    assert response.live_discovery_used is True
    assert response.evidence_scope is EvidenceScope.DISCOVERY_METADATA
    for paper_id in response.paper_ids:
        resolved = repository.get_paper(paper_id)
        assert resolved is not None
        assert ":" not in paper_id  # never a raw provider id
    assert set(response.paper_ids) == set(ingestion.persisted_ids)


async def test_one_ingestion_per_turn_even_when_storage_stays_thin(
    repository: ResearchRepository,
) -> None:
    """A single discovery result below the threshold triggers no second call."""

    ingestion = RecordingIngestionService(repository, _discovery_batch(1))

    response = await _service(repository, llm=ScriptedChatLLM(), ingestion=ingestion).chat(
        ChatRequest(text=RESEARCH_QUERY)
    )

    assert ingestion.call_count == 1
    assert response.live_discovery_used is True


async def test_repeat_query_reuses_stored_evidence_without_second_ingestion(
    repository: ResearchRepository,
) -> None:
    ingestion = RecordingIngestionService(repository, _discovery_batch(3))
    service = _service(repository, llm=ScriptedChatLLM(), ingestion=ingestion)

    first = await service.chat(ChatRequest(text=RESEARCH_QUERY))
    second = await service.chat(ChatRequest(text=RESEARCH_QUERY))

    assert ingestion.call_count == 1
    assert first.mode is ChatMode.RESEARCH_LIVE
    assert second.mode is ChatMode.RESEARCH_STORED
    assert second.live_discovery_used is False


# ---------------------------------------------------------------------------
# Budgets: auto_read is pinned to 0 and the discovery limit is clamped.
# ---------------------------------------------------------------------------


def test_budget_rejects_nonzero_auto_read() -> None:
    with pytest.raises(ValueError):
        ChatBudget(auto_read_pdfs=1)


async def test_discovery_limit_is_hard_clamped_to_twelve(
    repository: ResearchRepository,
) -> None:
    ingestion = RecordingIngestionService(repository)
    service = _service(
        repository,
        llm=ScriptedChatLLM(),
        ingestion=ingestion,
        budget=ChatBudget(max_discovery_results=999),
    )

    await service.chat(ChatRequest(text=RESEARCH_QUERY))

    assert ingestion.calls[0]["limit"] == 12


async def test_discovery_limit_floors_at_one_for_zero_or_negative_budgets(
    repository: ResearchRepository,
) -> None:
    ingestion = RecordingIngestionService(repository)

    for invalid in (0, -5):
        service = _service(
            repository,
            llm=ScriptedChatLLM(),
            ingestion=ingestion,
            budget=ChatBudget(max_discovery_results=invalid),
        )
        await service.chat(ChatRequest(text=RESEARCH_QUERY))

    assert [call["limit"] for call in ingestion.calls] == [1, 1]


# ---------------------------------------------------------------------------
# Citation validation inside the pipeline.
# ---------------------------------------------------------------------------


async def test_unknown_cited_id_is_stripped_from_the_response(
    repository: ResearchRepository,
) -> None:
    real_id = _seed_paper_with_card(repository, "seed0")
    _seed_paper_with_card(repository, "seed1")
    _seed_paper_with_card(repository, "seed2")
    llm = ScriptedChatLLM(
        payload={
            "answer": "Grounded answer.",
            "referenced_paper_ids": [real_id, "totally-made-up-id"],
            "referenced_gap_ids": [],
        }
    )

    response = await _service(repository, llm=llm).chat(ChatRequest(text=RESEARCH_QUERY))

    assert response.paper_ids == (real_id,)


# ---------------------------------------------------------------------------
# Outages degrade gracefully; the chat still returns.
# ---------------------------------------------------------------------------


async def test_memory_outage_degrades_without_failing_the_turn(
    repository: ResearchRepository,
) -> None:
    memory = FakeUserMemoryStore(fail=True)
    llm = ScriptedChatLLM()

    response = await _service(repository, memory=memory, llm=llm).chat(
        ChatRequest(text=MEMORY_QUESTION)
    )

    assert response.degraded is False
    assert response.used_user_memory is False
    assert response.mode is ChatMode.PERSONAL_MEMORY


async def test_semantic_outage_degrades_to_lexical_retrieval(
    repository: ResearchRepository,
) -> None:
    stored_ids = [_seed_paper_with_card(repository, f"seed{i}") for i in range(3)]

    class FailingIndex(FakeSemanticIndex):
        def search(self, vector, *, top_k=10, entity_type=None):  # type: ignore[override]
            raise RuntimeError("index unavailable")

    embedding = FakeEmbeddingProvider(dimension=DIMENSION)
    ingestion = RecordingIngestionService(repository)

    response = await _service(
        repository,
        llm=ScriptedChatLLM(),
        ingestion=ingestion,
        embedding=embedding,
        index=FailingIndex(),
    ).chat(ChatRequest(text=RESEARCH_QUERY))

    assert response.mode is ChatMode.RESEARCH_STORED
    assert set(response.paper_ids) <= set(stored_ids)
    assert ingestion.call_count == 0


async def test_ingestion_failure_degrades_to_stored_only_with_sanitized_log(
    repository: ResearchRepository,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ingestion = RecordingIngestionService(repository, exc=RuntimeError("provider exploded"))
    llm = ScriptedChatLLM()

    with caplog.at_level(logging.WARNING, logger="research_radar.chat.service"):
        response = await _service(repository, llm=llm, ingestion=ingestion).chat(
            ChatRequest(text=RESEARCH_QUERY)
        )

    assert ingestion.call_count == 1
    assert response.degraded is False
    assert response.live_discovery_used is False
    assert response.mode is ChatMode.RESEARCH_STORED
    joined = caplog.text
    assert "RuntimeError" in joined
    assert "provider exploded" not in joined
    assert "diffusion" not in joined.lower()


# ---------------------------------------------------------------------------
# LLM failure paths: safe error, discovered ids reported, NO memory write.
# ---------------------------------------------------------------------------


async def test_llm_failure_returns_safe_error_with_stored_discovery_ids_and_no_capture(
    repository: ResearchRepository,
) -> None:
    memory = FakeUserMemoryStore()
    ingestion = RecordingIngestionService(repository, _discovery_batch(2))
    llm = ScriptedChatLLM(exc=RuntimeError("llm down"))

    response = await _service(
        repository, memory=memory, llm=llm, ingestion=ingestion
    ).chat(ChatRequest(text=RESEARCH_QUERY))

    assert response.degraded is True
    assert response.mode is ChatMode.RESEARCH_LIVE
    assert len(response.paper_ids) == 2
    for paper_id in response.paper_ids:
        assert repository.get_paper(paper_id) is not None
    assert "stored" in response.text.lower()
    assert memory.episodes == ()  # capture never runs on the LLM failure path


async def test_missing_llm_returns_safe_error_without_invented_content(
    repository: ResearchRepository,
) -> None:
    response = await _service(repository, llm=None).chat(
        ChatRequest(text=RESEARCH_QUERY)
    )

    assert response.degraded is True
    assert response.paper_ids == ()
    assert response.text.strip()


# ---------------------------------------------------------------------------
# Casual conversation: zero retrieval, zero ingestion, no durable capture.
# ---------------------------------------------------------------------------


async def test_casual_request_does_no_retrieval_and_no_ingestion(
    repository: ResearchRepository,
) -> None:
    memory = FakeUserMemoryStore()
    ingestion = RecordingIngestionService(repository)
    llm = ScriptedChatLLM()

    response = await _service(
        repository, memory=memory, llm=llm, ingestion=ingestion
    ).chat(ChatRequest(text=SMALL_TALK))

    assert ingestion.call_count == 0
    assert response.mode is ChatMode.CONVERSATIONAL
    assert response.used_user_memory is False
    assert memory.episodes == ()
    prompt_text = llm.last_messages[-1].content
    assert "STORED SCIENTIFIC EVIDENCE" not in prompt_text
    assert "LIVE DISCOVERY EVIDENCE" not in prompt_text


# ---------------------------------------------------------------------------
# Personal memory questions use advisory context only.
# ---------------------------------------------------------------------------


async def test_memory_question_uses_advisory_context_without_research_backends(
    repository: ResearchRepository,
) -> None:
    fact_text = "My research interests include surgical robotics"
    memory = FakeUserMemoryStore(
        facts=[MemoryFact(fact=fact_text, memory_class=MemoryClass.RESEARCH_INTEREST)]
    )
    ingestion = RecordingIngestionService(repository)
    llm = ScriptedChatLLM()

    response = await _service(
        repository, memory=memory, llm=llm, ingestion=ingestion
    ).chat(ChatRequest(text=MEMORY_QUESTION))

    assert ingestion.call_count == 0
    assert response.mode is ChatMode.PERSONAL_MEMORY
    assert response.used_user_memory is True
    assert response.evidence_scope is EvidenceScope.USER_MEMORY


# ---------------------------------------------------------------------------
# Project turns load explicit canonical state.
# ---------------------------------------------------------------------------


async def test_project_hint_loads_explicit_project_memory_into_the_prompt(
    repository: ResearchRepository,
) -> None:
    repository.create_project(
        "medvla",
        constraints=["No cloud-only pipelines"],
        hypotheses=["On-device inference suffices"],
        rejected_ideas=["GAN-based augmentation"],
    )
    llm = ScriptedChatLLM()

    response = await _service(repository, llm=llm).chat(
        ChatRequest(text="what is the current status?", project_hint="medvla")
    )

    assert response.mode is ChatMode.PROJECT_RESEARCH
    prompt_text = "\n".join(message.content for message in llm.last_messages)
    assert "No cloud-only pipelines" in prompt_text
    assert "GAN-based augmentation" in prompt_text


async def test_project_hint_without_a_matching_project_still_answers(
    repository: ResearchRepository,
) -> None:
    llm = ScriptedChatLLM()

    response = await _service(repository, llm=llm).chat(
        ChatRequest(text="what is the current status?", project_hint="ghost")
    )

    assert response.mode is ChatMode.PROJECT_RESEARCH
    assert response.degraded is False


# ---------------------------------------------------------------------------
# Capture runs LAST, after a successful response, through the real policy.
# ---------------------------------------------------------------------------


async def test_durable_statement_is_captured_after_success(
    repository: ResearchRepository,
) -> None:
    memory = FakeUserMemoryStore()
    llm = ScriptedChatLLM()

    response = await _service(repository, memory=memory, llm=llm).chat(
        ChatRequest(text=DURABLE_STATEMENT)
    )

    assert response.degraded is False
    assert len(memory.episodes) == 1
    episode = memory.episodes[0]
    assert episode.content == DURABLE_STATEMENT
    assert episode.memory_class is MemoryClass.PREFERENCE


async def test_mixed_stored_and_discovered_scope_is_reported(
    repository: ResearchRepository,
) -> None:
    _seed_paper_with_card(repository, "seed0")
    ingestion = RecordingIngestionService(repository, _discovery_batch(2))

    response = await _service(
        repository, llm=ScriptedChatLLM(), ingestion=ingestion
    ).chat(ChatRequest(text=RESEARCH_QUERY))

    assert ingestion.call_count == 1
    assert response.evidence_scope is EvidenceScope.MIXED
    assert response.live_discovery_used is True


# ---------------------------------------------------------------------------
# Semantic channel widens stored recall without enabling discovery.
# ---------------------------------------------------------------------------


async def test_semantic_hit_can_satisfy_sufficiency_without_discovery(
    repository: ResearchRepository,
) -> None:
    stored_ids = [_seed_paper_with_card(repository, f"seed{i}") for i in range(2)]
    neighbour_id = repository.upsert_merged_paper(
        _paper("neighbour", "Conditional computation gating", "Gating networks.")
    )
    embedding = FakeEmbeddingProvider(dimension=DIMENSION)
    index = FakeSemanticIndex()
    index.upsert(
        [
            SemanticRecord(
                entity_id=f"paper:{neighbour_id}",
                entity_type="paper",
                paper_id=neighbour_id,
                vector=embedding.embed_texts([RESEARCH_QUERY])[0],
                embedding_schema_version="paper-v1",
                embedding_model="fake-embedding-v1",
            )
        ]
    )
    ingestion = RecordingIngestionService(repository)

    response = await _service(
        repository,
        llm=ScriptedChatLLM(),
        ingestion=ingestion,
        embedding=embedding,
        index=index,
    ).chat(ChatRequest(text=RESEARCH_QUERY))

    assert ingestion.call_count == 0
    assert response.mode is ChatMode.RESEARCH_STORED
    assert set(response.paper_ids) <= {*stored_ids, neighbour_id}


# ---------------------------------------------------------------------------
# Focused audit verification tests.
# ---------------------------------------------------------------------------


async def test_new_topic_without_project_default_discovery_limit_ten(
    repository: ResearchRepository,
) -> None:
    """A new topic without project runs stored-first (0 hits), then discovery with default 10."""

    discovered_papers = _discovery_batch(10, prefix="newtopic")
    ingestion = RecordingIngestionService(repository, discovered_papers)
    llm = ScriptedChatLLM(cite_ids_from_prompt=True)

    response = await _service(repository, llm=llm, ingestion=ingestion).chat(
        ChatRequest(text="what is the latest research on world models for robotics")
    )

    assert ingestion.call_count == 1
    call = ingestion.calls[0]
    assert call["query"] == "world models for robotics"
    assert call["limit"] == 10
    assert call["auto_read"] == 0
    assert call["project_id"] is None
    assert response.mode is ChatMode.RESEARCH_LIVE
    assert response.live_discovery_used is True
    assert response.evidence_scope is EvidenceScope.DISCOVERY_METADATA
    assert response.degraded is False
    assert len(response.paper_ids) == 10
    for paper_id in response.paper_ids:
        assert repository.get_paper(paper_id) is not None
    prompt_text = llm.last_messages[-1].content
    assert "LIVE DISCOVERY EVIDENCE" in prompt_text
    assert "EXPLICIT PROJECT MEMORY" not in prompt_text


async def test_discovery_deduplication_drops_duplicate_paper_ids(
    repository: ResearchRepository,
) -> None:
    """Duplicate paper IDs returned from discovery are resolved and deduped."""

    paper1 = _paper("dup1", "Duplicate Paper 1", "Abstract 1")
    paper2 = _paper("dup2", "Duplicate Paper 2", "Abstract 2")
    p1_id = repository.upsert_merged_paper(paper1)
    p2_id = repository.upsert_merged_paper(paper2)

    class DuplicateIngestionService:
        async def ingest_research_topic(self, query: str, **kwargs: Any) -> IngestionResult:
            return IngestionResult(
                run_id="run-dup",
                query=query,
                discovered_count=4,
                canonical_count=2,
                paper_ids=[p1_id, p1_id, p2_id, p1_id],
                warnings=[],
                provider_counts={},
                read_paper_ids=[],
            )

    llm = ScriptedChatLLM(cite_ids_from_prompt=True)
    service = _service(
        repository,
        llm=llm,
        ingestion=DuplicateIngestionService(),  # type: ignore[arg-type]
    )

    response = await service.chat(ChatRequest(text=RESEARCH_QUERY))

    assert response.mode is ChatMode.RESEARCH_LIVE
    assert response.paper_ids == (p1_id, p2_id)


async def test_unresolved_semantic_candidate_skipped_and_backfilled(
    repository: ResearchRepository,
) -> None:
    """Unresolvable candidate IDs in SQLite are skipped, and valid subsequent candidates
    fill budget.
    """

    valid_ids = [_seed_paper_with_card(repository, f"valid{i}") for i in range(3)]
    ghost_id = "ghost_nonexistent_id"

    class GhostRetrieverIndex(FakeSemanticIndex):
        pass

    embedding = FakeEmbeddingProvider(dimension=DIMENSION)
    index = GhostRetrieverIndex()
    # Upsert ghost first, then valid IDs
    index.upsert(
        [
            SemanticRecord(
                entity_id=f"paper:{ghost_id}",
                entity_type="paper",
                paper_id=ghost_id,
                vector=embedding.embed_texts([RESEARCH_QUERY])[0],
                embedding_schema_version="paper-v1",
                embedding_model="fake-embedding-v1",
            )
        ]
    )

    service = _service(
        repository,
        llm=ScriptedChatLLM(cite_ids_from_prompt=True),
        embedding=embedding,
        index=index,
        budget=ChatBudget(max_stored_evidence=3),
    )

    response = await service.chat(ChatRequest(text=RESEARCH_QUERY))

    assert ghost_id not in response.paper_ids
    assert set(response.paper_ids) == set(valid_ids)


async def test_candidate_gap_citation_validation(
    repository: ResearchRepository,
) -> None:
    """Candidate gap IDs referenced by the LLM are validated against the evidence packet."""

    paper_id = _seed_paper_with_card(repository, "gap_seed")
    gap = CandidateGap(
        id="GAP-001",
        title="Diffusion policy latency gap",
        description="Inference speed bottleneck in real-time control.",
        gap_type="contradiction",
        research_question="Can diffusion policy achieve 100Hz on edge GPUs?",
        supporting_papers=[paper_id],
        evidence_count=1,
        search_scope="1 paper",
        provenance=GapProvenance(corpus_description="Diffusion test"),
        created_at=datetime.now(UTC).replace(tzinfo=None),
    )
    repository.save_candidate(gap)

    llm = ScriptedChatLLM(
        payload={
            "answer": "There is a gap in latency.",
            "referenced_paper_ids": [paper_id],
            "referenced_gap_ids": ["GAP-001", "GAP-FABRICATED-999"],
        }
    )

    response = await _service(repository, llm=llm).chat(
        ChatRequest(text="find recent papers on diffusion policy latency")
    )

    assert response.gap_ids == ("GAP-001",)


async def test_llm_failure_on_stored_evidence_returns_no_paper_ids(
    repository: ResearchRepository,
) -> None:
    """When stored evidence is sufficient and the LLM fails, degraded response
    carries empty paper_ids.
    """

    for i in range(3):
        _seed_paper_with_card(repository, f"stored_seed{i}")

    llm = ScriptedChatLLM(exc=RuntimeError("LLM synthesis error"))
    response = await _service(repository, llm=llm).chat(
        ChatRequest(text=RESEARCH_QUERY)
    )

    assert response.degraded is True
    assert response.live_discovery_used is False
    assert response.paper_ids == ()
    assert "couldn't synthesize an answer" in response.text
