"""SQLAlchemy tables for ResearchRadar's small, single-user SQLite memory."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base owned exclusively by the storage package."""


class PaperTable(Base):
    """Canonical normalized paper metadata, without provider response payloads."""

    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    canonical_key: Mapped[str] = mapped_column(String(2048), unique=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    normalized_title: Mapped[str] = mapped_column(String(1000), index=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    authors: Mapped[list[str]] = mapped_column(JSON, default=list)
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str | None] = mapped_column(Text, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(512), nullable=True, index=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    primary_source: Mapped[str] = mapped_column(String(64))
    first_discovered_at: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

    sources: Mapped[list[PaperSourceTable]] = relationship(
        back_populates="paper",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    paper_card: Mapped[PaperCardTable | None] = relationship(
        back_populates="paper",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    watch_discoveries: Mapped[list[WatchPaperTable]] = relationship(
        back_populates="paper",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class PaperSourceTable(Base):
    """One provider-specific identifier and retrieval provenance for a paper."""

    __tablename__ = "paper_sources"

    __table_args__ = (
        UniqueConstraint("provider", "external_id", name="uq_paper_sources_provider_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(64))
    external_id: Mapped[str] = mapped_column(String(512))
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime)

    paper: Mapped[PaperTable] = relationship(back_populates="sources")


class WatchTopicTable(Base):
    """A saved single-user research query; intentionally has no user foreign key."""

    __tablename__ = "watch_topics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    normalized_name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    query: Mapped[str] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    last_scan_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    discoveries: Mapped[list[WatchPaperTable]] = relationship(
        back_populates="watch_topic",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class WatchPaperTable(Base):
    """Per-topic discovery and notification state for a canonical paper."""

    __tablename__ = "watch_papers"

    watch_topic_id: Mapped[str] = mapped_column(
        ForeignKey("watch_topics.id", ondelete="CASCADE"),
        primary_key=True,
    )
    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        primary_key=True,
    )
    first_seen_at: Mapped[datetime] = mapped_column(DateTime)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime)
    rank_score: Mapped[float] = mapped_column(Float)
    notified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    watch_topic: Mapped[WatchTopicTable] = relationship(back_populates="discoveries")
    paper: Mapped[PaperTable] = relationship(back_populates="watch_discoveries")


class PaperCardTable(Base):
    """Persisted validated PaperCard content and compact analysis provenance."""

    __tablename__ = "paper_cards"

    paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        primary_key=True,
    )
    problem: Mapped[str | None] = mapped_column(Text, nullable=True)
    motivation: Mapped[str | None] = mapped_column(Text, nullable=True)
    contributions: Mapped[list[str]] = mapped_column(JSON, default=list)
    methods: Mapped[list[str]] = mapped_column(JSON, default=list)
    datasets: Mapped[list[str]] = mapped_column(JSON, default=list)
    metrics: Mapped[list[str]] = mapped_column(JSON, default=list)
    main_claims: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    limitations: Mapped[list[str]] = mapped_column(JSON, default=list)
    future_work: Mapped[list[str]] = mapped_column(JSON, default=list)
    failure_cases: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    selected_sections: Mapped[list[str]] = mapped_column(JSON, default=list)
    llm_provider: Mapped[str | None] = mapped_column(String(128), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

    paper: Mapped[PaperTable] = relationship(back_populates="paper_card")


class DigestRunTable(Base):
    """Small idempotency/audit record for scheduled digest periods."""

    __tablename__ = "digest_runs"

    __table_args__ = (UniqueConstraint("period_start", "period_end", name="uq_digest_runs_period"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    period_start: Mapped[datetime] = mapped_column(DateTime, index=True)
    period_end: Mapped[datetime] = mapped_column(DateTime, index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    paper_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    safe_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class GapCandidateTable(Base):
    """Persisted research gap candidate and its provenance snapshot."""

    __tablename__ = "gap_candidates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    gap_type: Mapped[str] = mapped_column(String(64), index=True)
    research_question: Mapped[str] = mapped_column(Text)
    supporting_papers: Mapped[list[str]] = mapped_column(JSON, default=list)
    conflicting_papers: Mapped[list[str]] = mapped_column(JSON, default=list)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    novelty_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    importance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    feasibility_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    search_scope: Mapped[str] = mapped_column(Text)
    caveats: Mapped[list[str]] = mapped_column(JSON, default=list)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    review_status: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

    reviews: Mapped[list[GapReviewTable]] = relationship(
        back_populates="candidate",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class GapReviewTable(Base):
    """Append-only Critic review log for a candidate gap."""

    __tablename__ = "gap_reviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(
        ForeignKey("gap_candidates.id", ondelete="CASCADE"),
        index=True,
    )
    review_version: Mapped[int] = mapped_column(Integer)
    queries_used: Mapped[list[str]] = mapped_column(JSON, default=list)
    retrieval_records: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    new_paper_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    overlapping_paper_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    decision: Mapped[str] = mapped_column(String(32), index=True)
    rationale: Mapped[str] = mapped_column(Text)
    caveats: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime)

    candidate: Mapped[GapCandidateTable] = relationship(back_populates="reviews")

