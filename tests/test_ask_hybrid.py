"""Tests that semantic retrieval widens /ask recall without weakening its rules."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from research_radar.models import Paper
from research_radar.research.ask import AskService
from research_radar.semantic.base import SemanticRecord
from research_radar.semantic.embedding import FakeEmbeddingProvider
from research_radar.semantic.index import DisabledSemanticIndex, FakeSemanticIndex
from research_radar.storage.database import Database, create_database, initialize_schema
from research_radar.storage.repositories import ResearchRepository

DIMENSION = 8
QUESTION = "sparse mixture of experts routing"


@pytest.fixture
def database() -> Iterator[Database]:
    database = create_database("sqlite:///:memory:")
    initialize_schema(database)
    yield database
    database.dispose()


@pytest.fixture
def repository(database: Database) -> ResearchRepository:
    return ResearchRepository(database)


def _store(repository: ResearchRepository, slug: str, title: str, abstract: str) -> str:
    return repository.upsert_merged_paper(
        Paper(
            id=f"openalex:{slug}",
            title=title,
            abstract=abstract,
            authors=[],
            publication_year=2023,
            venue="Venue",
            doi=f"10.1000/{slug}",
            url=f"https://example.test/{slug}",
            citation_count=1,
            source="openalex",
            external_ids={"openalex": slug},
        )
    )


def _index(index: FakeSemanticIndex, paper_id: str, vector: list[float]) -> None:
    index.upsert(
        [
            SemanticRecord(
                entity_id=f"paper:{paper_id}",
                entity_type="paper",
                paper_id=paper_id,
                vector=vector,
                embedding_schema_version="paper-v1",
                embedding_model="fake-embedding-v1",
            )
        ]
    )


def test_disabled_semantic_index_preserves_lexical_ask_context(
    repository: ResearchRepository,
) -> None:
    """With semantics off, the retrieved set is exactly the lexical set."""

    lexical = _store(repository, "a", "Sparse mixture of experts routing", "Routing tokens.")
    _store(repository, "b", "Medieval pottery kiln typology", "Kilns and glazes.")

    plain = AskService(repository)
    disabled = AskService(
        repository,
        embedding_provider=FakeEmbeddingProvider(dimension=DIMENSION),
        semantic_index=DisabledSemanticIndex(),
    )

    plain_ids = [p.id for p in plain.build_ask_context(QUESTION).retrieved_papers]
    disabled_ids = [p.id for p in disabled.build_ask_context(QUESTION).retrieved_papers]

    assert plain_ids == disabled_ids == [lexical]


def test_lexical_gate_still_excludes_papers_no_channel_retrieved(
    repository: ResearchRepository,
) -> None:
    """A paper with neither lexical overlap nor a semantic hit stays out."""

    _store(repository, "a", "Sparse mixture of experts routing", "Routing tokens.")
    unrelated = _store(repository, "b", "Medieval pottery kiln typology", "Kilns and glazes.")

    embedding = FakeEmbeddingProvider(dimension=DIMENSION)
    index = FakeSemanticIndex()
    service = AskService(repository, embedding_provider=embedding, semantic_index=index)

    context = service.build_ask_context(QUESTION)

    assert unrelated not in {p.id for p in context.retrieved_papers}
    assert unrelated not in context.allowed_paper_ids


def test_semantic_hit_can_fill_unused_evidence_budget(
    repository: ResearchRepository,
) -> None:
    """A semantically related paper enters evidence when budget remains."""

    lexical = _store(repository, "a", "Sparse mixture of experts routing", "Routing tokens.")
    neighbour = _store(repository, "b", "Conditional computation gating", "Gating networks.")

    embedding = FakeEmbeddingProvider(dimension=DIMENSION)
    index = FakeSemanticIndex()
    _index(index, neighbour, embedding.embed_texts([QUESTION])[0])
    service = AskService(repository, embedding_provider=embedding, semantic_index=index)

    context = service.build_ask_context(QUESTION)
    retrieved = [p.id for p in context.retrieved_papers]

    assert lexical in retrieved
    assert neighbour in retrieved
    # Lexical evidence always outranks a semantic-only candidate.
    assert retrieved.index(lexical) < retrieved.index(neighbour)


def test_semantic_candidates_never_displace_lexical_evidence(
    repository: ResearchRepository,
) -> None:
    """When budget is full, semantic-only candidates take no slot."""

    lexical_ids = [
        _store(repository, f"lex{n}", f"Sparse mixture of experts routing {n}", "Routing.")
        for n in range(6)
    ]
    neighbour = _store(repository, "sem", "Conditional computation gating", "Gating networks.")

    embedding = FakeEmbeddingProvider(dimension=DIMENSION)
    index = FakeSemanticIndex()
    _index(index, neighbour, embedding.embed_texts([QUESTION])[0])
    service = AskService(repository, embedding_provider=embedding, semantic_index=index)

    retrieved = [p.id for p in service.build_ask_context(QUESTION).retrieved_papers]

    assert len(retrieved) == 6
    assert neighbour not in retrieved
    assert set(retrieved) <= set(lexical_ids)


def test_semantic_id_without_a_canonical_row_is_discarded(
    repository: ResearchRepository,
) -> None:
    """A stale vector id must never reach AskContext or the allowed id set."""

    _store(repository, "a", "Sparse mixture of experts routing", "Routing tokens.")

    embedding = FakeEmbeddingProvider(dimension=DIMENSION)
    index = FakeSemanticIndex()
    _index(index, "vanished-paper", embedding.embed_texts([QUESTION])[0])
    service = AskService(repository, embedding_provider=embedding, semantic_index=index)

    context = service.build_ask_context(QUESTION)

    assert "vanished-paper" not in {p.id for p in context.retrieved_papers}
    assert "vanished-paper" not in context.allowed_paper_ids


def test_semantic_outage_leaves_ask_context_on_lexical_results(
    repository: ResearchRepository,
) -> None:
    """An index failure degrades /ask to lexical retrieval without raising."""

    lexical = _store(repository, "a", "Sparse mixture of experts routing", "Routing tokens.")

    class _FailingIndex(FakeSemanticIndex):
        def search(self, vector, *, top_k=10, entity_type=None):  # type: ignore[override]
            raise RuntimeError("index unavailable")

    service = AskService(
        repository,
        embedding_provider=FakeEmbeddingProvider(dimension=DIMENSION),
        semantic_index=_FailingIndex(),
    )

    context = service.build_ask_context(QUESTION)

    assert [p.id for p in context.retrieved_papers] == [lexical]


def test_ask_context_is_deterministic_with_semantics_enabled(
    repository: ResearchRepository,
) -> None:
    """Repeated calls must produce an identical retrieved ordering."""

    for slug in ("a", "b", "c"):
        _store(repository, slug, f"Sparse mixture of experts routing {slug}", "Routing.")
    neighbour = _store(repository, "sem", "Conditional computation gating", "Gating.")

    embedding = FakeEmbeddingProvider(dimension=DIMENSION)
    index = FakeSemanticIndex()
    _index(index, neighbour, embedding.embed_texts([QUESTION])[0])
    service = AskService(repository, embedding_provider=embedding, semantic_index=index)

    first = [p.id for p in service.build_ask_context(QUESTION).retrieved_papers]
    second = [p.id for p in service.build_ask_context(QUESTION).retrieved_papers]

    assert first == second


async def test_source_id_validation_still_rejects_unretrieved_ids(
    repository: ResearchRepository,
) -> None:
    """A model citing a paper outside AskContext must still be filtered."""

    _store(repository, "a", "Sparse mixture of experts routing", "Routing tokens.")

    class _FabricatingLLM:
        async def generate_structured(self, messages, response_model):  # type: ignore[no-untyped-def]
            return response_model(
                answer="Synthesized from stored evidence.",
                referenced_paper_ids=["totally-made-up-id"],
                referenced_gap_ids=["invented-gap"],
                is_sufficient_evidence=True,
            )

    service = AskService(
        repository,
        llm_provider=_FabricatingLLM(),
        embedding_provider=FakeEmbeddingProvider(dimension=DIMENSION),
        semantic_index=FakeSemanticIndex(),
    )

    response = await service.ask(QUESTION)

    assert response.referenced_paper_ids == []
    assert response.referenced_gap_ids == []
