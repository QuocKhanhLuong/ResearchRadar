"""Tests for research gap domain models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from research_radar.models.gap import (
    CandidateGap,
    CriticReview,
    EvidenceRef,
    GapProvenance,
)


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def test_evidence_ref_validation() -> None:
    ref = EvidenceRef(
        paper_id="paper-1",
        paper_title="Title A",
        evidence_kind="supporting",
        claim_or_field="limitations[0]",
        supporting_text="Scanner shift causes degradation.",
    )
    assert ref.paper_id == "paper-1"
    assert ref.evidence_kind == "supporting"

    with pytest.raises(ValidationError):
        EvidenceRef(
            paper_id="",
            paper_title="Title",
            claim_or_field="field",
        )


def test_candidate_gap_score_bounds() -> None:
    now = _utc_now()
    prov = GapProvenance(
        retrievals=[],
        corpus_paper_ids=["p1", "p2"],
        corpus_description="Test corpus",
    )

    candidate = CandidateGap(
        id="gap-1",
        title="Test Gap",
        description="Description",
        research_question="Question?",
        supporting_papers=["p1", "p2"],
        evidence_count=2,
        evidence_score=0.7,
        confidence=0.5,
        search_scope="2 papers",
        provenance=prov,
        created_at=now,
    )
    assert candidate.evidence_score == 0.7

    with pytest.raises(ValidationError):
        CandidateGap(
            id="gap-1",
            title="Test",
            description="Desc",
            research_question="Q",
            evidence_count=1,
            evidence_score=1.5,  # > 1.0 invalid
            search_scope="scope",
            provenance=prov,
            created_at=now,
        )


def test_critic_review_version_must_be_positive() -> None:
    now = _utc_now()
    review = CriticReview(
        candidate_id="gap-1",
        review_version=1,
        queries_used=["query 1"],
        decision="preserved",
        rationale="No overlap found",
        created_at=now,
    )
    assert review.review_version == 1

    with pytest.raises(ValidationError):
        CriticReview(
            candidate_id="gap-1",
            review_version=0,  # invalid version
            decision="preserved",
            rationale="Rationale",
            created_at=now,
        )
