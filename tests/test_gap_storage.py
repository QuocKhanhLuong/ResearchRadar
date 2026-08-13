"""Tests for candidate gap and critic review storage operations in SQLite."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from research_radar.gap.service import GapService
from research_radar.models.gap import (
    CandidateGap,
    CriticReview,
    EvidenceRef,
    GapProvenance,
    RetrievalRecord,
)
from research_radar.models.paper import Paper
from research_radar.models.paper_card import PaperCard
from research_radar.research.scout import ScoutResult
from research_radar.storage.database import Database
from research_radar.storage.repositories import ResearchRepository


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class FakeScout:
    def __init__(self, return_papers: list[Paper] | None = None) -> None:
        self.return_papers = return_papers or []

    async def search(self, query: str, limit: int) -> ScoutResult:
        return ScoutResult(
            papers=self.return_papers,
            warnings=[],
            provider_counts={"fake": len(self.return_papers)},
        )


def test_candidate_gap_and_review_storage_roundtrip(tmp_path_factory: object) -> None:
    db_file = tmp_path_factory.mktemp("db") / "test.db"  # type: ignore[attr-defined]
    db = Database.create(f"sqlite:///{db_file}")
    db.initialize_schema()
    repo = ResearchRepository(db)

    now = _utc_now()
    ref = EvidenceRef(
        paper_id="p1",
        paper_title="Paper 1",
        evidence_kind="supporting",
        claim_or_field="limitations[0]",
        supporting_text="Limitation statement text",
    )
    rec = RetrievalRecord(
        query="MRI domain shift",
        query_purpose="critic",
        sources_searched=["openalex", "arxiv"],
        successful_sources=["openalex"],
        failed_sources=["arxiv"],
        retrieved_at=now,
        retrieved_paper_ids=["p1"],
        result_count=1,
    )
    prov = GapProvenance(
        retrievals=[rec],
        corpus_paper_ids=["p1", "p2"],
        corpus_description="Test scope",
        supporting_evidence=[ref],
        conflicting_evidence=[],
    )

    cand = CandidateGap(
        id="cand-123",
        title="Domain Shift Gap",
        description="Description text",
        gap_type="explicit",
        research_question="Question text?",
        supporting_papers=["p1"],
        evidence_count=1,
        evidence_score=0.6,
        confidence=0.5,
        search_scope="2 papers",
        caveats=["Caveat 1"],
        provenance=prov,
        review_status="candidate",
        created_at=now,
    )

    # Save candidate
    saved_cand = repo.save_candidate(cand)
    assert saved_cand.id == "cand-123"

    # Get candidate
    fetched_cand = repo.get_candidate("cand-123")
    assert fetched_cand is not None
    ev_text = fetched_cand.provenance.supporting_evidence[0].supporting_text
    assert ev_text == "Limitation statement text"

    # Save review
    review = CriticReview(
        candidate_id="cand-123",
        review_version=1,
        queries_used=["query 1"],
        retrieval_records=[rec],
        new_paper_ids=[],
        overlapping_paper_ids=[],
        decision="preserved",
        rationale="No overlapping work found.",
        caveats=["Caveat 1"],
        created_at=now,
    )
    repo.save_critic_review(review)

    # Update candidate status without erasing provenance
    repo.update_candidate_status("cand-123", "preserved", confidence=0.7)

    updated_cand = repo.get_candidate("cand-123")
    assert updated_cand is not None
    assert updated_cand.review_status == "preserved"
    assert updated_cand.confidence == 0.7
    assert len(updated_cand.provenance.supporting_evidence) == 1

    # List reviews
    reviews = repo.list_critic_reviews("cand-123")
    assert len(reviews) == 1
    assert reviews[0].decision == "preserved"

    db.dispose()


@pytest.mark.asyncio
async def test_repeated_gap_run_preserves_candidate_lineage_and_increments_review_version(
    tmp_path_factory: object,
) -> None:
    db_file = tmp_path_factory.mktemp("db") / "test_lineage.db"  # type: ignore[attr-defined]
    db = Database.create(f"sqlite:///{db_file}")
    db.initialize_schema()
    repo = ResearchRepository(db)

    p1 = Paper(id="p1", title="Spectral MRI under Scanner domain shift", source="arxiv")
    p2 = Paper(id="p2", title="CT Recon under Scanner domain shift", source="openalex")
    id1 = repo.upsert_merged_paper(p1)
    id2 = repo.upsert_merged_paper(p2)

    card1 = PaperCard(
        paper_id=id1,
        problem="Spectral MRI",
        limitations=["Scanner domain shift degrades performance."],
    )
    card2 = PaperCard(
        paper_id=id2,
        problem="CT Recon",
        limitations=["Scanner domain shift causes degradation."],
    )
    repo.upsert_paper_card(card1)
    repo.upsert_paper_card(card2)

    fake_scout = FakeScout(return_papers=[])
    gap_service = GapService(repo, fake_scout)

    # Pass 1
    res1 = await gap_service.analyze_gaps("Scanner domain shift", count=1)
    assert len(res1.candidates) == 1
    cand1 = res1.candidates[0]
    assert res1.reviews[0].review_version == 1

    # Pass 2 - repeated query for same topic
    res2 = await gap_service.analyze_gaps("Scanner domain shift", count=1)
    assert len(res2.candidates) == 1
    cand2 = res2.candidates[0]

    # Same candidate lineage (same ID)
    assert cand2.id == cand1.id
    # Review version incremented to 2
    assert res2.reviews[0].review_version == 2

    # Reviews history persisted in DB
    reviews_in_db = repo.list_critic_reviews(cand1.id)
    assert len(reviews_in_db) == 2
    assert [r.review_version for r in reviews_in_db] == [1, 2]

    db.dispose()
