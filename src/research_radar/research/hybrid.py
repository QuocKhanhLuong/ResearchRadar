"""Deterministic fusion of lexical and semantic paper candidates.

This module produces bounded CANDIDATE IDs and nothing else. It never loads
evidence, never talks to a language model, and never lets a vector-store result
stand on its own: every candidate is resolved against SQLite, which remains the
canonical source of truth, and an id that no longer resolves is discarded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from research_radar.semantic.base import EmbeddingProvider, SemanticIndex
from research_radar.storage.repositories import ResearchRepository

logger = logging.getLogger(__name__)

_RELATION_FACTORS: dict[str, float] = {
    "seed": 1.0,
    "supporting": 1.0,
    "conflicting": 1.0,
    "relevant": 0.6,
    "background": 0.25,
}
_DEFAULT_RELATION_FACTOR = 0.25


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """Fixed weights bounding how the two retrieval channels are combined.

    ``max_project_prior`` is deliberately tiny. With these defaults a single
    reciprocal-rank contribution spans 0.01639 (lexical rank 1) down to 0.00889
    (semantic rank 30), so capping the prior below 0.00889 guarantees that a
    project relationship can never compensate for missing an entire retrieval
    channel. The prior reorders papers retrieval already surfaced; it cannot
    promote one it did not. ``assert_prior_cannot_dominate`` proves this holds
    for whatever values a caller supplies.
    """

    top_k: int = 20
    lexical_limit: int = 30
    semantic_limit: int = 30
    rrf_k: int = 60
    lexical_weight: float = 1.0
    semantic_weight: float = 0.8
    project_prior_weight: float = 0.008
    max_project_prior: float = 0.008

    def __post_init__(self) -> None:
        assert_prior_cannot_dominate(self)


def assert_prior_cannot_dominate(config: HybridConfig) -> None:
    """Reject a configuration whose project prior could outweigh a channel.

    Raising here rather than silently ranking badly keeps the epistemic rule -
    evidence retrieval decides relevance, project membership only nudges it -
    enforceable rather than aspirational.
    """

    weakest_semantic = config.semantic_weight * reciprocal_rank(
        config.semantic_limit, k=config.rrf_k
    )
    effective_prior = min(config.max_project_prior, config.project_prior_weight)
    if effective_prior >= weakest_semantic:
        raise ValueError(
            "max_project_prior must stay below the weakest semantic contribution "
            f"({weakest_semantic:.5f}) so a project prior cannot replace retrieval evidence."
        )


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """One candidate paper and the channels that surfaced it."""

    paper_id: str
    lexical_rank: int | None
    semantic_rank: int | None
    semantic_score: float | None
    project_relation: str | None
    fused_score: float

    @property
    def retrieval_score(self) -> float:
        """Return the fused score without the project prior contribution."""

        return self.fused_score - project_prior(self.project_relation)


def reciprocal_rank(rank: int, *, k: int = 60) -> float:
    """Return the reciprocal-rank contribution of a 1-based rank."""

    if rank < 1:
        raise ValueError("rank must be 1-based")
    return 1.0 / (k + rank)


def project_prior(
    relation: str | None,
    *,
    weight: float = 0.008,
    maximum: float = 0.008,
) -> float:
    """Return the bounded relevance bonus a project relationship may contribute.

    The cap is the whole point. A project prior nudges ordering among papers
    the retrieval channels already surfaced; it must never be large enough to
    lift a paper that neither channel returned above one that both did.
    """

    if relation is None:
        return 0.0
    factor = _RELATION_FACTORS.get(relation, _DEFAULT_RELATION_FACTOR)
    return min(maximum, weight * factor)


def semantic_only_ids(candidates: list[FusedCandidate]) -> list[str]:
    """Return ids that only the semantic channel surfaced, in fused order."""

    return [candidate.paper_id for candidate in candidates if candidate.lexical_rank is None]


class HybridRetriever:
    """Fuse lexical and semantic candidates into a bounded, resolved id list."""

    def __init__(
        self,
        *,
        repository: ResearchRepository,
        embedding_provider: EmbeddingProvider | None = None,
        semantic_index: SemanticIndex | None = None,
        config: HybridConfig | None = None,
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._semantic_index = semantic_index
        self._config = config or HybridConfig()

    @property
    def config(self) -> HybridConfig:
        """Return the fusion weights in force."""

        return self._config

    @property
    def semantic_available(self) -> bool:
        """Return whether a semantic channel can currently contribute."""

        return (
            self._embedding_provider is not None
            and self._semantic_index is not None
            and self._semantic_index.available
        )

    def retrieve(
        self,
        query: str,
        *,
        project_paper_relations: dict[str, str] | None = None,
        top_k: int | None = None,
    ) -> list[FusedCandidate]:
        """Return bounded candidates ordered by deterministic reciprocal-rank fusion."""

        normalized_query = " ".join(query.split())
        if not normalized_query:
            return []

        relations = project_paper_relations or {}
        limit = top_k if top_k is not None else self._config.top_k

        lexical_ranks = self._lexical_ranks(normalized_query)
        semantic_ranks, semantic_scores = self._semantic_ranks(normalized_query)

        candidates: list[FusedCandidate] = []
        for paper_id in {**lexical_ranks, **semantic_ranks}:
            # Pinecone only proposes candidates. SQLite decides what exists, so
            # an id that no longer resolves is dropped rather than surfaced.
            if self._repository.get_paper(paper_id) is None:
                logger.debug("Discarding a candidate that no longer resolves in storage.")
                continue

            lexical_rank = lexical_ranks.get(paper_id)
            semantic_rank = semantic_ranks.get(paper_id)
            relation = relations.get(paper_id)
            score = project_prior(
                relation,
                weight=self._config.project_prior_weight,
                maximum=self._config.max_project_prior,
            )
            if lexical_rank is not None:
                score += self._config.lexical_weight * reciprocal_rank(
                    lexical_rank, k=self._config.rrf_k
                )
            if semantic_rank is not None:
                score += self._config.semantic_weight * reciprocal_rank(
                    semantic_rank, k=self._config.rrf_k
                )
            candidates.append(
                FusedCandidate(
                    paper_id=paper_id,
                    lexical_rank=lexical_rank,
                    semantic_rank=semantic_rank,
                    semantic_score=semantic_scores.get(paper_id),
                    project_relation=relation,
                    fused_score=score,
                )
            )

        candidates.sort(key=lambda candidate: (-candidate.fused_score, candidate.paper_id))
        return candidates[: max(0, limit)]

    def _lexical_ranks(self, query: str) -> dict[str, int]:
        """Return 1-based lexical ranks from the existing SQLite search."""

        rows = self._repository.search_papers(query, limit=self._config.lexical_limit)
        return {row.id: index for index, row in enumerate(rows, start=1)}

    def _semantic_ranks(self, query: str) -> tuple[dict[str, int], dict[str, float]]:
        """Return 1-based semantic ranks per paper, best rank wins.

        A semantic failure is absorbed here: the caller degrades to lexical
        retrieval rather than failing. The index is called once, never retried.
        """

        if not self.semantic_available:
            return {}, {}
        assert self._embedding_provider is not None  # narrowed by semantic_available
        assert self._semantic_index is not None
        try:
            vectors = self._embedding_provider.embed_texts([query])
            if not vectors:
                return {}, {}
            hits = self._semantic_index.search(
                vectors[0], top_k=self._config.semantic_limit, entity_type=None
            )
        except Exception:
            logger.warning("Semantic retrieval unavailable; continuing lexically.")
            return {}, {}

        ranks: dict[str, int] = {}
        scores: dict[str, float] = {}
        for index, hit in enumerate(hits, start=1):
            # A paper and its card can both hit; collapse to the best rank.
            if hit.paper_id not in ranks:
                ranks[hit.paper_id] = index
                scores[hit.paper_id] = hit.score
        return ranks, scores
