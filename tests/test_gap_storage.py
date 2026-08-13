"""Tests for candidate gap and critic review storage operations in SQLite."""

from __future__ import annotations

from datetime import UTC, datetime

from research_radar.models.gap import (
    CandidateGap,
    CriticReview,
    EvidenceRef,
    GapProvenance,
    RetrievalRecord,
)
from research_radar.storage.database import Database
from research_radar.storage.repositories import ResearchRepository


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


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
