"""Tests for bounded Critic verification pass with fake/mocked providers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from research_radar.errors import ProviderUnavailableError
from research_radar.gap.critic import CriticService
from research_radar.models.gap import CandidateGap, GapProvenance
from research_radar.models.paper import Paper
from research_radar.models.paper_card import PaperCard
from research_radar.research.scout import ScoutResult
from research_radar.storage.repositories import StoredPaperCard


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class FakeScout:
    def __init__(
        self,
        return_papers: list[Paper] | None = None,
        fail: bool = False,
        warnings: list[str] | None = None,
    ) -> None:
        self.return_papers = return_papers or []
        self.fail = fail
        self.warnings = warnings or []
        self.queries_received: list[str] = []

    async def search(self, query: str, limit: int) -> ScoutResult:
        self.queries_received.append(query)
        if self.fail:
            raise ProviderUnavailableError("All scholarly providers failed")
        counts = {"fake_provider": len(self.return_papers)} if self.return_papers else {}
        return ScoutResult(
            papers=self.return_papers, warnings=self.warnings, provider_counts=counts
        )


def _make_candidate(now: datetime) -> CandidateGap:
    prov = GapProvenance(
        retrievals=[],
        corpus_paper_ids=["p1", "p2"],
        corpus_description="scope",
    )
    return CandidateGap(
        id="gap-1",
        title="Scanner domain shift in MRI",
        description="Description",
        research_question="How to maintain quality under scanner domain shift?",
        supporting_papers=["p1", "p2"],
        evidence_count=2,
        search_scope="2 cards",
        provenance=prov,
        review_status="candidate",
        confidence=0.5,
        created_at=now,
    )


@pytest.mark.asyncio
async def test_critic_preserves_candidate_when_no_overlap_found() -> None:
    now = _utc_now()
    cand = _make_candidate(now)

    fake_scout = FakeScout(return_papers=[])
    critic = CriticService(fake_scout)

    review, updated = await critic.review_candidate(cand)

    assert review.decision == "preserved"
    assert updated.review_status == "preserved"
    assert updated.confidence is not None and updated.confidence > 0.5
    assert len(fake_scout.queries_received) <= 4


@pytest.mark.asyncio
async def test_critic_downgrades_candidate_when_overlap_found() -> None:
    now = _utc_now()
    cand = _make_candidate(now)

    fresh_paper = Paper(
        id="fresh-1",
        title="Robust Scanner Domain Shift Adaptation in MRI",
        abstract="We solve scanner domain shift for MRI reconstruction.",
        source="arxiv",
    )
    fake_scout = FakeScout(return_papers=[fresh_paper])
    critic = CriticService(fake_scout)

    review, updated = await critic.review_candidate(cand)

    assert review.decision == "downgraded"
    assert updated.review_status == "downgraded"
    assert "fresh-1" in review.overlapping_paper_ids


@pytest.mark.asyncio
async def test_critic_handles_partial_provider_failure() -> None:
    now = _utc_now()
    cand = _make_candidate(now)

    fake_scout = FakeScout(return_papers=[], warnings=["Semantic Scholar was unavailable"])
    critic = CriticService(fake_scout)

    review, updated = await critic.review_candidate(cand)

    assert review.decision == "downgraded"
    assert "Semantic Scholar was unavailable" in updated.caveats


@pytest.mark.asyncio
async def test_critic_rejects_when_memory_paper_card_resolves_gap() -> None:
    now = _utc_now()
    cand = _make_candidate(now)

    # Paper p3 is in memory and explicitly resolves scanner domain shift in MRI
    resolving_card = StoredPaperCard(
        card=PaperCard(
            paper_id="p3",
            contributions=["We propose domain adaptation for scanner domain shift in MRI."],
        ),
        source_url=None,
        document_sha256=None,
        selected_sections=(),
        llm_provider=None,
        llm_model=None,
        created_at=now,
        updated_at=now,
    )

    fake_scout = FakeScout(return_papers=[])
    critic = CriticService(fake_scout)

    review, updated = await critic.review_candidate(
        cand, memory_cards=(resolving_card,)
    )

    assert review.decision == "rejected"
    assert updated.review_status == "rejected"
    assert "p3" in review.rationale


@pytest.mark.asyncio
async def test_critic_downgrades_when_all_providers_fail() -> None:
    now = _utc_now()
    cand = _make_candidate(now)

    fake_scout = FakeScout(fail=True)
    critic = CriticService(fake_scout)

    review, updated = await critic.review_candidate(cand)

    assert review.decision == "downgraded"
    assert updated.review_status == "downgraded"
    assert "all scholarly providers were unavailable" in review.rationale.lower()


@pytest.mark.asyncio
async def test_metadata_overlap_only_downgrades_never_rejects() -> None:
    now = _utc_now()
    cand = _make_candidate(now)

    fresh_paper = Paper(
        id="fresh-title-match",
        title="Scanner Domain Shift in MRI Reconstruction Survey",
        abstract="A survey of scanner domain shift in MRI.",
        source="openalex",
    )

    fake_scout = FakeScout(return_papers=[fresh_paper])
    critic = CriticService(fake_scout)

    review, updated = await critic.review_candidate(cand)

    assert review.decision == "downgraded"
    assert review.decision != "rejected"


@pytest.mark.asyncio
async def test_critic_downgrades_when_keywords_match_but_task_is_mismatched() -> None:
    now = _utc_now()
    # Candidate gap is about segmentation task
    cand = CandidateGap(
        id="gap-seg",
        title="Scanner domain shift in segmentation",
        description="Description",
        research_question="How to handle scanner domain shift in image segmentation?",
        supporting_papers=["p1"],
        evidence_count=1,
        search_scope="1 card",
        provenance=GapProvenance(retrievals=[], corpus_paper_ids=["p1"], corpus_description="s"),
        review_status="candidate",
        created_at=now,
    )

    # Memory card resolves scanner domain shift, but for reconstruction task!
    diff_task_card = StoredPaperCard(
        card=PaperCard(
            paper_id="p-recon",
            problem="Image Reconstruction",
            contributions=["We resolve scanner domain shift for image reconstruction."],
        ),
        source_url=None,
        document_sha256=None,
        selected_sections=(),
        llm_provider=None,
        llm_model=None,
        created_at=now,
        updated_at=now,
    )

    fake_scout = FakeScout(return_papers=[])
    critic = CriticService(fake_scout)

    review, updated = await critic.review_candidate(cand, memory_cards=(diff_task_card,))

    # Different task context must NOT reject!
    assert review.decision != "rejected"
    assert updated.review_status != "rejected"


@pytest.mark.asyncio
async def test_critic_stores_invalidating_evidence_ref_on_rejection() -> None:
    now = _utc_now()
    cand = _make_candidate(now)

    resolving_card = StoredPaperCard(
        card=PaperCard(
            paper_id="p-res",
            contributions=["We propose scanner domain shift adaptation for MRI."],
        ),
        source_url=None,
        document_sha256=None,
        selected_sections=(),
        llm_provider=None,
        llm_model=None,
        created_at=now,
        updated_at=now,
    )

    fake_scout = FakeScout(return_papers=[])
    critic = CriticService(fake_scout)

    review, updated = await critic.review_candidate(cand, memory_cards=(resolving_card,))

    assert review.decision == "rejected"
    assert len(updated.provenance.conflicting_evidence) == 1
    assert updated.provenance.conflicting_evidence[0].paper_id == "p-res"
