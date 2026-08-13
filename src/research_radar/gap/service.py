"""Domain facade for research gap scoping, mining, critic verification, and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from research_radar.gap.corpus import ScopedCorpusService
from research_radar.gap.coverage import CoverageGapMiner
from research_radar.gap.critic import CriticService, ScoutSearchProtocol
from research_radar.gap.evaluation import EvaluationGapMiner
from research_radar.gap.miner import ExplicitGapMiner
from research_radar.models.gap import CandidateGap, CriticReview
from research_radar.storage.repositories import ScopedCorpusResult


class GapStorageProtocol(Protocol):
    """Minimum repository interface required for GapService persistence."""

    def get_scoped_corpus(self, topic: str, limit: int = 50) -> ScopedCorpusResult: ...

    def save_candidate(self, candidate: CandidateGap) -> CandidateGap: ...

    def get_candidate(self, candidate_id: str) -> CandidateGap | None: ...

    def list_candidates(
        self, gap_type: str | None = None, limit: int = 50
    ) -> list[CandidateGap]: ...

    def save_critic_review(self, review: CriticReview) -> CriticReview: ...

    def list_critic_reviews(self, candidate_id: str) -> list[CriticReview]: ...


@dataclass(frozen=True, slots=True)
class GapAnalysisResult:
    """The complete result of a gap analysis workflow pass."""

    candidates: list[CandidateGap] = field(default_factory=list)
    reviews: list[CriticReview] = field(default_factory=list)
    is_insufficient_evidence: bool = False
    message: str | None = None
    matching_papers_count: int = 0
    analyzed_cards_count: int = 0
    unanalyzed_papers_count: int = 0


class GapService:
    """Orchestrate scoped corpus selection, gap mining, and Critic review."""

    def __init__(
        self,
        repository: GapStorageProtocol,
        scout: ScoutSearchProtocol,
        *,
        corpus_service: ScopedCorpusService | None = None,
        explicit_miner: ExplicitGapMiner | None = None,
        miner: ExplicitGapMiner | None = None,
        coverage_miner: CoverageGapMiner | None = None,
        evaluation_miner: EvaluationGapMiner | None = None,
        critic: CriticService | None = None,
    ) -> None:
        self._repository = repository
        self._scout = scout
        self._corpus_service = corpus_service or ScopedCorpusService(repository)
        self._explicit_miner = explicit_miner or miner or ExplicitGapMiner()
        self._coverage_miner = coverage_miner or CoverageGapMiner()
        self._evaluation_miner = evaluation_miner or EvaluationGapMiner()
        self._critic = critic or CriticService(scout)

    async def analyze_gaps(
        self,
        topic: str,
        count: int = 1,
        gap_type: Literal["explicit", "coverage", "evaluation"] = "explicit",
    ) -> GapAnalysisResult:
        """Run the end-to-end gap pipeline for the requested gap type."""

        clean_topic = " ".join(topic.split())
        if not clean_topic:
            raise ValueError("Topic query cannot be empty.")
        if count < 1:
            raise ValueError("Count must be at least 1.")

        corpus = self._corpus_service.select_corpus(clean_topic, limit=50)

        analyzed_cards_count = len(corpus.cards)
        matching_papers_count = corpus.total_matching_papers
        unanalyzed_papers_count = len(corpus.missing_cards_paper_ids)

        if analyzed_cards_count < 2:
            msg = (
                f"Insufficient structured evidence.\n\n"
                f"I found only {analyzed_cards_count} analyzed paper(s) (with PaperCards) "
                f"in the scoped corpus for '{clean_topic}'.\n"
                f"Add/read more papers or expand topic before generating candidate."
            )
            if unanalyzed_papers_count > 0:
                msg += (
                    f"\nNote: There are {unanalyzed_papers_count} stored matching paper(s) "
                    f"that have not been read/analyzed yet."
                )
            return GapAnalysisResult(
                candidates=[],
                reviews=[],
                is_insufficient_evidence=True,
                message=msg,
                matching_papers_count=matching_papers_count,
                analyzed_cards_count=analyzed_cards_count,
                unanalyzed_papers_count=unanalyzed_papers_count,
            )

        if gap_type == "coverage":
            mined_candidates = self._coverage_miner.mine_coverage_gaps(clean_topic, corpus)
        elif gap_type == "evaluation":
            mined_candidates = self._evaluation_miner.mine_evaluation_gaps(clean_topic, corpus)
        else:
            mined_candidates = self._explicit_miner.mine_gaps(clean_topic, corpus)

        if not mined_candidates:
            msg = (
                f"Insufficient repeated gap signals.\n\n"
                f"No {gap_type} gap candidate was found across the scoped corpus "
                f"for '{clean_topic}'."
            )
            return GapAnalysisResult(
                candidates=[],
                reviews=[],
                is_insufficient_evidence=True,
                message=msg,
                matching_papers_count=matching_papers_count,
                analyzed_cards_count=analyzed_cards_count,
                unanalyzed_papers_count=unanalyzed_papers_count,
            )

        selected_candidates = mined_candidates[:count]
        final_candidates: list[CandidateGap] = []
        final_reviews: list[CriticReview] = []

        for mined_candidate in selected_candidates:
            existing_cand = self._repository.get_candidate(mined_candidate.id)
            target_candidate = existing_cand if existing_cand is not None else mined_candidate

            existing_reviews = self._repository.list_critic_reviews(target_candidate.id)
            next_version = len(existing_reviews) + 1

            review, updated_candidate = await self._critic.review_candidate(
                target_candidate, review_version=next_version, memory_cards=corpus.cards
            )

            self._repository.save_candidate(updated_candidate)
            self._repository.save_critic_review(review)

            final_candidates.append(updated_candidate)
            final_reviews.append(review)

        return GapAnalysisResult(
            candidates=final_candidates,
            reviews=final_reviews,
            is_insufficient_evidence=False,
            matching_papers_count=matching_papers_count,
            analyzed_cards_count=analyzed_cards_count,
            unanalyzed_papers_count=unanalyzed_papers_count,
        )

    def get_candidate_detail(
        self, candidate_id: str
    ) -> tuple[CandidateGap | None, list[CriticReview]]:
        """Retrieve a candidate gap and all its append-only Critic reviews."""

        candidate = self._repository.get_candidate(candidate_id)
        if candidate is None:
            return None, []
        reviews = self._repository.list_critic_reviews(candidate_id)
        return candidate, reviews
