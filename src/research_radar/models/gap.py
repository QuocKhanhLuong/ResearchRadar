"""Domain models for research gap provenance and candidate review."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceRef(BaseModel):
    """Refers to a specific piece of evidence in a stored paper/card."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    paper_id: str = Field(min_length=1)
    paper_title: str = Field(min_length=1)
    evidence_kind: Literal["supporting", "conflicting", "context"] = "supporting"
    claim_or_field: str = Field(min_length=1)
    source_section: str | None = None
    supporting_text: str | None = None
    source_url: str | None = None


class RetrievalRecord(BaseModel):
    """Audit log of a single scholarly provider search execution."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1)
    query_purpose: str = Field(min_length=1)
    sources_searched: list[str] = Field(default_factory=list)
    successful_sources: list[str] = Field(default_factory=list)
    failed_sources: list[str] = Field(default_factory=list)
    retrieved_at: datetime
    retrieved_paper_ids: list[str] = Field(default_factory=list)
    result_count: int = Field(ge=0, default=0)


class GapProvenance(BaseModel):
    """Complete traceable lineage of evidence and searches backing a gap."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    retrievals: list[RetrievalRecord] = Field(default_factory=list)
    corpus_paper_ids: list[str] = Field(default_factory=list)
    corpus_description: str = Field(min_length=1)
    supporting_evidence: list[EvidenceRef] = Field(default_factory=list)
    conflicting_evidence: list[EvidenceRef] = Field(default_factory=list)


class CandidateGap(BaseModel):
    """A qualified candidate research question grounded in retrieved evidence."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    gap_type: Literal[
        "explicit", "coverage", "contradiction", "evaluation", "method_transfer"
    ] = "explicit"
    research_question: str = Field(min_length=1)
    supporting_papers: list[str] = Field(default_factory=list)
    conflicting_papers: list[str] = Field(default_factory=list)
    evidence_count: int = Field(ge=0)
    novelty_score: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    importance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    feasibility_score: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    search_scope: str = Field(min_length=1)
    caveats: list[str] = Field(default_factory=list)
    provenance: GapProvenance
    review_status: Literal["candidate", "preserved", "downgraded", "rejected"] = (
        "candidate"
    )
    created_at: datetime


class CriticReview(BaseModel):
    """Append-only audit record of a Critic verification pass on a candidate gap."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1)
    review_version: int = Field(ge=1)
    queries_used: list[str] = Field(default_factory=list)
    retrieval_records: list[RetrievalRecord] = Field(default_factory=list)
    new_paper_ids: list[str] = Field(default_factory=list)
    overlapping_paper_ids: list[str] = Field(default_factory=list)
    decision: Literal["preserved", "downgraded", "rejected"]
    rationale: str = Field(min_length=1)
    caveats: list[str] = Field(default_factory=list)
    created_at: datetime
