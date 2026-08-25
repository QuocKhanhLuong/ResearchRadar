"""Tests for deterministic lexical/semantic candidate fusion."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest

from research_radar.models import Paper
from research_radar.research.hybrid import (
    HybridConfig,
    HybridRetriever,
    project_prior,
    reciprocal_rank,
    semantic_only_ids,
)
from research_radar.semantic.base import SemanticHit, SemanticIndex, SemanticRecord
from research_radar.semantic.embedding import FakeEmbeddingProvider
from research_radar.semantic.index import DisabledSemanticIndex, FakeSemanticIndex
from research_radar.storage.database import Database, create_database, initialize_schema
from research_radar.storage.repositories import ResearchRepository

DIMENSION = 8


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


def _index_paper(index: FakeSemanticIndex, paper_id: str, vector: list[float]) -> None:
    index.upsert(
        [
            SemanticRecord(
                entity_id=f"paper:{paper_id}",
                entity_type="paper",
                paper_id=paper_id,
                vector=vector,
                publication_year=2023,
                embedding_schema_version="paper-v1",
                embedding_model="fake-embedding-v1",
            )
        ]
    )


class _CountingIndex:
    """Fails on search and records how many times it was asked."""

    backend = "counting"

    def __init__(self) -> None:
        self.search_calls = 0

    @property
    def available(self) -> bool:
        return True

    def upsert(self, records: Sequence[SemanticRecord]) -> int:
        return 0

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 10,
        entity_type: str | None = None,
    ) -> list[SemanticHit]:
        self.search_calls += 1
        raise RuntimeError("index unavailable")

    def delete(self, entity_ids: Sequence[str]) -> int:
        return 0

    def status(self) -> object:
        return None


def test_config_rejects_a_project_prior_that_could_replace_retrieval() -> None:
    """A prior large enough to outweigh a whole channel must be refused."""

    with pytest.raises(ValueError, match="max_project_prior"):
        HybridConfig(project_prior_weight=0.25, max_project_prior=0.25)


def test_default_prior_stays_below_the_weakest_semantic_contribution() -> None:
    """The shipped defaults must satisfy the invariant they claim."""

    config = HybridConfig()
    weakest_semantic = config.semantic_weight * reciprocal_rank(
        config.semantic_limit, k=config.rrf_k
    )
    assert config.max_project_prior < weakest_semantic


def test_lexical_only_retrieval_preserves_existing_behaviour(
    repository: ResearchRepository,
) -> None:
    """With no semantic channel the fused order is the lexical order."""

    first = _store(repository, "a", "Sparse mixture of experts routing", "Routing tokens.")
    _store(repository, "b", "Unrelated crystallography study", "Diffraction patterns.")

    retriever = HybridRetriever(repository=repository)

    candidates = retriever.retrieve("sparse mixture of experts")

    assert retriever.semantic_available is False
    assert [candidate.paper_id for candidate in candidates][:1] == [first]
    assert all(candidate.semantic_rank is None for candidate in candidates)


def test_semantic_channel_contributes_candidates_lexical_search_misses(
    repository: ResearchRepository,
) -> None:
    """A paper only the vector index surfaces still becomes a candidate."""

    _store(repository, "a", "Sparse mixture of experts routing", "Routing tokens.")
    semantic_paper = _store(
        repository, "b", "Conditional computation in transformers", "Gating networks."
    )

    embedding = FakeEmbeddingProvider(dimension=DIMENSION)
    index = FakeSemanticIndex()
    _index_paper(index, semantic_paper, embedding.embed_texts(["sparse experts"])[0])

    retriever = HybridRetriever(
        repository=repository, embedding_provider=embedding, semantic_index=index
    )

    candidates = retriever.retrieve("sparse experts")

    assert retriever.semantic_available is True
    assert semantic_paper in {candidate.paper_id for candidate in candidates}
    assert semantic_only_ids(candidates) == [semantic_paper]


def test_a_paper_in_both_channels_outranks_a_paper_in_one(
    repository: ResearchRepository,
) -> None:
    """Appearing in both channels is worth more than appearing in one."""

    both = _store(repository, "a", "Sparse mixture of experts routing", "Sparse experts.")
    lexical_only = _store(repository, "b", "Sparse experts survey", "Sparse experts.")

    embedding = FakeEmbeddingProvider(dimension=DIMENSION)
    index = FakeSemanticIndex()
    _index_paper(index, both, embedding.embed_texts(["sparse experts"])[0])

    retriever = HybridRetriever(
        repository=repository, embedding_provider=embedding, semantic_index=index
    )

    ordering = [candidate.paper_id for candidate in retriever.retrieve("sparse experts")]

    assert ordering.index(both) < ordering.index(lexical_only)


def test_project_prior_cannot_outrank_an_extra_retrieval_channel(
    repository: ResearchRepository,
) -> None:
    """A seed relation must not compensate for missing the semantic channel."""

    both = _store(repository, "a", "Sparse mixture of experts routing", "Sparse experts.")
    lexical_only = _store(repository, "b", "Sparse experts survey", "Sparse experts.")

    embedding = FakeEmbeddingProvider(dimension=DIMENSION)
    index = FakeSemanticIndex()
    _index_paper(index, both, embedding.embed_texts(["sparse experts"])[0])

    retriever = HybridRetriever(
        repository=repository, embedding_provider=embedding, semantic_index=index
    )

    ordering = [
        candidate.paper_id
        for candidate in retriever.retrieve(
            "sparse experts", project_paper_relations={lexical_only: "seed"}
        )
    ]

    assert ordering.index(both) < ordering.index(lexical_only)


def test_project_prior_never_exceeds_its_cap(repository: ResearchRepository) -> None:
    """Every relation, known or unknown, stays within the configured cap."""

    config = HybridConfig()
    for relation in ("seed", "supporting", "conflicting", "relevant", "background", "mystery"):
        assert (
            project_prior(
                relation,
                weight=config.project_prior_weight,
                maximum=config.max_project_prior,
            )
            <= config.max_project_prior + 1e-12
        )
    assert project_prior(None) == 0.0


def test_a_paper_in_neither_channel_is_never_a_candidate(
    repository: ResearchRepository,
) -> None:
    """Project membership alone cannot introduce a paper retrieval did not find."""

    _store(repository, "a", "Sparse mixture of experts routing", "Sparse experts.")
    unrelated = _store(repository, "b", "Medieval pottery kiln typology", "Kilns.")

    retriever = HybridRetriever(repository=repository)

    candidates = retriever.retrieve(
        "sparse experts", project_paper_relations={unrelated: "seed"}
    )

    assert unrelated not in {candidate.paper_id for candidate in candidates}


def test_results_are_deterministic_across_runs(repository: ResearchRepository) -> None:
    """Identical inputs must produce an identical ordering, ties included."""

    for slug in ("a", "b", "c", "d"):
        _store(repository, slug, f"Sparse experts study {slug}", "Sparse experts.")

    retriever = HybridRetriever(repository=repository)

    first = [candidate.paper_id for candidate in retriever.retrieve("sparse experts")]
    second = [candidate.paper_id for candidate in retriever.retrieve("sparse experts")]

    assert first == second


def test_candidates_that_no_longer_resolve_in_sqlite_are_discarded(
    repository: ResearchRepository,
) -> None:
    """A vector id without a canonical row must never reach the caller."""

    _store(repository, "a", "Sparse mixture of experts routing", "Sparse experts.")

    embedding = FakeEmbeddingProvider(dimension=DIMENSION)
    index = FakeSemanticIndex()
    _index_paper(index, "deleted-paper-id", embedding.embed_texts(["sparse experts"])[0])

    retriever = HybridRetriever(
        repository=repository, embedding_provider=embedding, semantic_index=index
    )

    candidates = retriever.retrieve("sparse experts")

    assert "deleted-paper-id" not in {candidate.paper_id for candidate in candidates}


def test_semantic_outage_degrades_to_lexical_without_raising(
    repository: ResearchRepository,
) -> None:
    """An index failure must leave lexical retrieval fully intact."""

    _store(repository, "a", "Sparse mixture of experts routing", "Sparse experts.")
    _store(repository, "b", "Sparse experts survey", "Sparse experts.")

    embedding = FakeEmbeddingProvider(dimension=DIMENSION)
    failing = _CountingIndex()
    hybrid = HybridRetriever(
        repository=repository, embedding_provider=embedding, semantic_index=failing
    )
    lexical = HybridRetriever(repository=repository)

    degraded = [candidate.paper_id for candidate in hybrid.retrieve("sparse experts")]
    baseline = [candidate.paper_id for candidate in lexical.retrieve("sparse experts")]

    assert degraded == baseline
    assert failing.search_calls == 1


def test_unavailable_index_matches_lexical_results(repository: ResearchRepository) -> None:
    """An index reporting unavailable contributes nothing and raises nothing."""

    _store(repository, "a", "Sparse mixture of experts routing", "Sparse experts.")

    embedding = FakeEmbeddingProvider(dimension=DIMENSION)
    index = FakeSemanticIndex(available=False)
    retriever = HybridRetriever(
        repository=repository, embedding_provider=embedding, semantic_index=index
    )

    assert retriever.semantic_available is False
    assert [candidate.paper_id for candidate in retriever.retrieve("sparse experts")] == [
        candidate.paper_id
        for candidate in HybridRetriever(repository=repository).retrieve("sparse experts")
    ]


def test_semantic_requires_both_an_embedder_and_an_index(
    repository: ResearchRepository,
) -> None:
    """Neither half alone enables the semantic channel."""

    index_only = HybridRetriever(repository=repository, semantic_index=FakeSemanticIndex())
    embedder_only = HybridRetriever(
        repository=repository, embedding_provider=FakeEmbeddingProvider(dimension=DIMENSION)
    )
    disabled = HybridRetriever(
        repository=repository,
        embedding_provider=FakeEmbeddingProvider(dimension=DIMENSION),
        semantic_index=DisabledSemanticIndex(),
    )

    assert index_only.semantic_available is False
    assert embedder_only.semantic_available is False
    assert disabled.semantic_available is False


def test_top_k_bounds_results_and_the_argument_wins(repository: ResearchRepository) -> None:
    """An explicit top_k overrides the configured default."""

    for slug in ("a", "b", "c", "d", "e"):
        _store(repository, slug, f"Sparse experts study {slug}", "Sparse experts.")

    retriever = HybridRetriever(repository=repository, config=HybridConfig(top_k=4))

    assert len(retriever.retrieve("sparse experts")) <= 4
    assert len(retriever.retrieve("sparse experts", top_k=2)) == 2


def test_blank_query_returns_no_candidates(repository: ResearchRepository) -> None:
    """A query with no content retrieves nothing rather than everything."""

    _store(repository, "a", "Sparse mixture of experts routing", "Sparse experts.")

    retriever = HybridRetriever(repository=repository)

    assert retriever.retrieve("   ") == []


def test_paper_and_card_hits_collapse_to_the_best_rank(
    repository: ResearchRepository,
) -> None:
    """One paper hitting twice stays one candidate holding its best rank."""

    target = _store(repository, "a", "Sparse mixture of experts routing", "Sparse experts.")
    other = _store(repository, "b", "Conditional computation", "Gating.")

    embedding = FakeEmbeddingProvider(dimension=DIMENSION)
    query_vector = embedding.embed_texts(["sparse experts"])[0]
    index = FakeSemanticIndex()
    _index_paper(index, other, embedding.embed_texts(["totally different"])[0])
    index.upsert(
        [
            SemanticRecord(
                entity_id=f"paper:{target}",
                entity_type="paper",
                paper_id=target,
                vector=query_vector,
                embedding_schema_version="paper-v1",
                embedding_model="fake-embedding-v1",
            ),
            SemanticRecord(
                entity_id=f"papercard:{target}",
                entity_type="papercard",
                paper_id=target,
                vector=query_vector,
                embedding_schema_version="papercard-v1",
                embedding_model="fake-embedding-v1",
            ),
        ]
    )

    retriever = HybridRetriever(
        repository=repository, embedding_provider=embedding, semantic_index=index
    )

    candidates = retriever.retrieve("sparse experts")
    matching = [candidate for candidate in candidates if candidate.paper_id == target]

    assert len(matching) == 1
    assert matching[0].semantic_rank == 1


def test_fake_index_and_retriever_satisfy_the_semantic_protocol() -> None:
    """The fusion module depends only on the shared SemanticIndex contract."""

    assert isinstance(FakeSemanticIndex(), SemanticIndex)
    assert isinstance(DisabledSemanticIndex(), SemanticIndex)
