"""Short-lived-session repository operations for ResearchRadar memory."""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import and_, desc, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload, sessionmaker

from research_radar.models import (
    CandidateGap,
    CriticReview,
    GapProvenance,
    Paper,
    PaperCard,
    Project,
    ProjectGapLink,
    ProjectPaperLink,
    RetrievalRecord,
    StructuredEvidence,
)
from research_radar.storage.database import Database
from research_radar.storage.tables import (
    DigestRunTable,
    GapCandidateTable,
    GapReviewTable,
    PaperCardTable,
    PaperSourceTable,
    PaperTable,
    ProjectGapTable,
    ProjectPaperTable,
    ProjectTable,
    WatchPaperTable,
    WatchTopicTable,
)

logger = logging.getLogger(__name__)


class StorageError(RuntimeError):
    """Raised when a repository transaction cannot safely complete."""


@dataclass(frozen=True, slots=True)
class PaperSource:
    """Provider-specific paper identity retained as compact provenance."""

    provider: str
    external_id: str
    source_url: str | None
    retrieved_at: datetime


@dataclass(frozen=True, slots=True)
class StoredPaper:
    """Canonical paper metadata plus the stable SQLite-assigned paper id."""

    id: str
    canonical_key: str
    title: str
    abstract: str | None
    authors: list[str]
    publication_year: int | None
    venue: str | None
    doi: str | None
    url: str | None
    citation_count: int | None
    primary_source: str
    sources: tuple[PaperSource, ...]
    first_discovered_at: datetime
    created_at: datetime
    updated_at: datetime

    @property
    def paper_id(self) -> str:
        """Alias that makes relationships such as ``PaperCard.paper_id`` clear."""

        return self.id

    @property
    def external_ids(self) -> dict[str, str]:
        """Return provider ids in the normalized domain-model shape."""

        return {source.provider: source.external_id for source in self.sources}

    @property
    def source(self) -> str:
        """Alias for callers that consume a provider-neutral paper shape."""

        return self.primary_source


@dataclass(frozen=True, slots=True)
class StoredPaperCard:
    """A validated PaperCard with provenance needed for later evidence work."""

    card: PaperCard
    source_url: str | None
    document_sha256: str | None
    selected_sections: tuple[str, ...]
    llm_provider: str | None
    llm_model: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class WatchTopic:
    """One single-user saved research query."""

    id: str
    name: str
    query: str
    enabled: bool
    created_at: datetime
    last_scan_at: datetime | None
    last_error: str | None


@dataclass(frozen=True, slots=True)
class PendingNotification:
    """A discovered paper that has not yet been successfully notified."""

    topic_id: str
    paper: StoredPaper
    rank_score: float
    first_seen_at: datetime
    last_seen_at: datetime


@dataclass(frozen=True, slots=True)
class DigestCandidate:
    """Persisted paper data selected from a completed discovery period."""

    paper: StoredPaper
    watch_topic_names: tuple[str, ...]
    highest_rank_score: float | None
    paper_card: PaperCard | None


@dataclass(frozen=True, slots=True)
class DigestRun:
    """A scheduled digest period and its idempotency state."""

    id: str
    period_start: datetime
    period_end: datetime
    status: str
    paper_count: int
    created_at: datetime
    sent_at: datetime | None
    safe_error: str | None


@dataclass(frozen=True, slots=True)
class ScopedCorpusResult:
    """The result of scoping persisted papers/cards by topic."""

    cards: tuple[StoredPaperCard, ...]
    papers: tuple[StoredPaper, ...]
    corpus_paper_ids: tuple[str, ...]
    missing_cards_paper_ids: tuple[str, ...]
    total_matching_papers: int


class ResearchRepository:
    """All SQLAlchemy query and transaction boundaries for V1 storage.

    Methods are synchronous by design and make a short-lived ``Session`` for
    every call. Async services can safely invoke a method with
    ``asyncio.to_thread`` without passing sessions between event-loop tasks.
    """

    def __init__(self, database: Database | sessionmaker[Session]) -> None:
        self._session_factory = (
            database.session_factory if isinstance(database, Database) else database
        )

    def upsert_merged_paper(self, paper: Paper) -> str:
        """Persist a normalized paper and return its stable storage id.

        Existing provider ids, DOI, canonical identity, and normalized title
        are checked in one transaction. If independently ingested records are
        later connected by a shared identifier, their source, watch, and card
        relationships are reconciled under one stable canonical record.
        """

        now = _utc_now()
        source_pairs = _paper_source_pairs(paper)
        identity_keys = _identity_keys(paper, source_pairs)
        canonical_key = identity_keys[0]
        normalized_title = _normalize_title(paper.title)
        title_identity = _title_identity(normalized_title)
        normalized_doi = _normalize_doi(paper.doi) or _source_doi(source_pairs)

        with self._session_scope() as session:
            candidates = self._find_paper_candidates(
                session,
                source_pairs=source_pairs,
                identity_keys=identity_keys,
                normalized_title=title_identity,
                normalized_doi=normalized_doi,
            )
            if candidates:
                target = candidates[0]
                if len(candidates) > 1:
                    self._merge_existing_papers(session, target, candidates[1:])
                self._merge_incoming_paper(
                    target,
                    paper,
                    normalized_title,
                    normalized_doi,
                    now,
                )
            else:
                target = PaperTable(
                    id=_new_id(),
                    canonical_key=canonical_key,
                    title=paper.title,
                    normalized_title=normalized_title,
                    abstract=paper.abstract,
                    authors=list(paper.authors),
                    publication_year=paper.publication_year,
                    venue=paper.venue,
                    doi=normalized_doi,
                    url=paper.url,
                    citation_count=paper.citation_count,
                    primary_source=_normalize_provider(paper.source),
                    first_discovered_at=now,
                    created_at=now,
                    updated_at=now,
                )
                session.add(target)
                session.flush()

            self._adopt_canonical_key(session, target, canonical_key)
            self._upsert_sources(session, target, source_pairs, paper.url, now)
            session.flush()
            return target.id

    def get_paper(self, paper_id: str) -> StoredPaper | None:
        """Return one canonical paper and its provider identity provenance."""

        with self._session_scope() as session:
            statement = (
                select(PaperTable)
                .where(PaperTable.id == paper_id)
                .options(selectinload(PaperTable.sources))
            )
            row = session.scalar(statement)
            return _to_stored_paper(row) if row is not None else None

    def list_paper_sources(self, paper_id: str) -> list[PaperSource]:
        """Return retained provider provenance for a canonical paper."""

        with self._session_scope() as session:
            rows = session.scalars(
                select(PaperSourceTable)
                .where(PaperSourceTable.paper_id == paper_id)
                .order_by(PaperSourceTable.provider, PaperSourceTable.external_id)
            ).all()
            return [_to_paper_source(row) for row in rows]

    def get_papers_for_local_lexical_search(self, query: str, limit: int = 20) -> list[StoredPaper]:
        """Return compact local metadata matches without an external search.

        This deliberately uses simple deterministic token matching. It is a
        storage primitive for a future bounded ``/ask`` flow, not a vector or
        full-text-search subsystem.
        """

        tokens = _lexical_tokens(query)
        if not tokens:
            return []
        _validate_limit(limit)

        with self._session_scope() as session:
            rows = session.scalars(
                select(PaperTable)
                .options(
                    selectinload(PaperTable.sources),
                    selectinload(PaperTable.paper_card),
                )
                .order_by(desc(PaperTable.first_discovered_at), PaperTable.id)
            ).all()

            scored_rows: list[tuple[int, PaperTable]] = []
            for row in rows:
                title_tokens = set(row.normalized_title.split())
                abstract_tokens = set(_lexical_tokens(row.abstract or ""))
                card_tokens = _paper_card_tokens(row.paper_card)
                title_hits = sum(token in title_tokens for token in tokens)
                abstract_hits = sum(token in abstract_tokens for token in tokens)
                card_hits = sum(token in card_tokens for token in tokens)
                score = (3 * title_hits) + (2 * abstract_hits) + card_hits
                if score:
                    scored_rows.append((score, row))

            scored_rows.sort(
                key=lambda item: (
                    -item[0],
                    -(item[1].publication_year or 0),
                    -(item[1].citation_count or 0),
                    item[1].normalized_title,
                )
            )
            return [_to_stored_paper(row) for _, row in scored_rows[:limit]]

    def upsert_paper_card(
        self,
        card: PaperCard,
        *,
        source_url: str | None = None,
        document_sha256: str | None = None,
        selected_sections: Iterable[str] | Mapping[str, object] | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> PaperCard:
        """Store one validated V1 analysis for an already persisted paper."""

        now = _utc_now()
        section_names = _normalize_selected_sections(selected_sections)
        with self._session_scope() as session:
            if session.get(PaperTable, card.paper_id) is None:
                raise ValueError(
                    f"Cannot store a PaperCard for unknown paper id {card.paper_id!r}."
                )

            row = session.get(PaperCardTable, card.paper_id)
            if row is None:
                row = PaperCardTable(
                    paper_id=card.paper_id,
                    created_at=now,
                    updated_at=now,
                    contributions=[],
                    methods=[],
                    datasets=[],
                    metrics=[],
                    main_claims=[],
                    limitations=[],
                    future_work=[],
                    failure_cases=[],
                    selected_sections=[],
                )
                session.add(row)

            _copy_card_values(row, card)
            if source_url is not None:
                row.source_url = source_url
            if document_sha256 is not None:
                row.document_sha256 = document_sha256
            if selected_sections is not None:
                row.selected_sections = section_names
            if llm_provider is not None:
                row.llm_provider = llm_provider
            if llm_model is not None:
                row.llm_model = llm_model
            row.updated_at = now
            session.flush()
            return _to_paper_card(row)

    def get_paper_card(self, paper_id: str) -> PaperCard | None:
        """Return a validated domain PaperCard without leaking ORM state."""

        record = self.get_paper_card_record(paper_id)
        return record.card if record is not None else None

    def get_paper_card_record(self, paper_id: str) -> StoredPaperCard | None:
        """Return a PaperCard together with compact reader/LLM provenance."""

        with self._session_scope() as session:
            row = session.get(PaperCardTable, paper_id)
            return _to_stored_paper_card(row) if row is not None else None

    def add_watch_topic(self, name: str, query: str) -> WatchTopic:
        """Add a saved query, rejecting duplicate normalized names."""

        clean_name = _require_text(name, "Watch topic name")
        clean_query = _require_text(query, "Watch topic query")
        normalized_name = _normalize_watch_name(clean_name)
        now = _utc_now()

        with self._session_scope() as session:
            existing = session.scalar(
                select(WatchTopicTable).where(WatchTopicTable.normalized_name == normalized_name)
            )
            if existing is not None:
                raise ValueError(f"A watch topic named {clean_name!r} already exists.")

            row = WatchTopicTable(
                id=_new_id(),
                name=clean_name,
                normalized_name=normalized_name,
                query=clean_query,
                enabled=True,
                created_at=now,
                last_scan_at=None,
                last_error=None,
            )
            session.add(row)
            session.flush()
            return _to_watch_topic(row)

    def get_watch_topic(self, topic_id: str) -> WatchTopic | None:
        """Return a saved topic by its stable id."""

        with self._session_scope() as session:
            row = session.get(WatchTopicTable, topic_id)
            return _to_watch_topic(row) if row is not None else None

    def list_watch_topics(self) -> list[WatchTopic]:
        """List all topics in stable creation order."""

        with self._session_scope() as session:
            rows = session.scalars(
                select(WatchTopicTable).order_by(WatchTopicTable.created_at, WatchTopicTable.id)
            ).all()
            return [_to_watch_topic(row) for row in rows]

    def list_enabled_watch_topics(self) -> list[WatchTopic]:
        """List enabled topics only, for the scheduled scan service."""

        with self._session_scope() as session:
            rows = session.scalars(
                select(WatchTopicTable)
                .where(WatchTopicTable.enabled.is_(True))
                .order_by(WatchTopicTable.created_at, WatchTopicTable.id)
            ).all()
            return [_to_watch_topic(row) for row in rows]

    def set_watch_topic_enabled(self, topic_id: str, enabled: bool) -> WatchTopic | None:
        """Enable or pause one topic without deleting its discovery history."""

        with self._session_scope() as session:
            row = session.get(WatchTopicTable, topic_id)
            if row is None:
                return None
            row.enabled = enabled
            session.flush()
            return _to_watch_topic(row)

    def remove_watch_topic(self, topic_id_or_name: str) -> bool:
        """Delete a topic by id or its normalized, unique display name."""

        identifier = _require_text(topic_id_or_name, "Watch topic identifier")
        with self._session_scope() as session:
            row = session.get(WatchTopicTable, identifier)
            if row is None:
                row = session.scalar(
                    select(WatchTopicTable).where(
                        WatchTopicTable.normalized_name == _normalize_watch_name(identifier)
                    )
                )
            if row is None:
                return False
            session.delete(row)
            return True

    def mark_watch_scan_success(
        self,
        topic_id: str,
        *,
        scanned_at: datetime | None = None,
    ) -> bool:
        """Record a successful scan and clear any prior safe error."""

        with self._session_scope() as session:
            row = session.get(WatchTopicTable, topic_id)
            if row is None:
                return False
            row.last_scan_at = _as_database_time(scanned_at or _utc_now())
            row.last_error = None
            return True

    def mark_watch_scan_failure(self, topic_id: str, safe_error: str) -> bool:
        """Record a bounded safe error without advancing ``last_scan_at``."""

        with self._session_scope() as session:
            row = session.get(WatchTopicTable, topic_id)
            if row is None:
                return False
            row.last_error = _bounded_error(safe_error)
            return True

    def record_watch_discovery(
        self,
        topic_id: str,
        paper_id: str,
        rank_score: float,
        *,
        seen_at: datetime | None = None,
    ) -> bool:
        """Record a topic-paper sighting and return whether it is newly seen."""

        if not math.isfinite(rank_score):
            raise ValueError("rank_score must be finite.")
        now = _as_database_time(seen_at or _utc_now())

        with self._session_scope() as session:
            if session.get(WatchTopicTable, topic_id) is None:
                raise ValueError(f"Unknown watch topic id {topic_id!r}.")
            if session.get(PaperTable, paper_id) is None:
                raise ValueError(f"Unknown paper id {paper_id!r}.")

            key = {"watch_topic_id": topic_id, "paper_id": paper_id}
            row = session.get(WatchPaperTable, key)
            if row is None:
                session.add(
                    WatchPaperTable(
                        watch_topic_id=topic_id,
                        paper_id=paper_id,
                        first_seen_at=now,
                        last_seen_at=now,
                        rank_score=float(rank_score),
                        notified_at=None,
                    )
                )
                return True

            row.first_seen_at = min(row.first_seen_at, now)
            row.last_seen_at = max(row.last_seen_at, now)
            row.rank_score = max(row.rank_score, float(rank_score))
            return False

    def list_pending_notifications(
        self,
        topic_id: str,
        cap: int = 3,
        *,
        minimum_rank_score: float | None = None,
    ) -> list[PendingNotification]:
        """Return highest-ranked unseen topic discoveries, capped for anti-spam."""

        _validate_limit(cap)
        with self._session_scope() as session:
            conditions = [
                WatchPaperTable.watch_topic_id == topic_id,
                WatchPaperTable.notified_at.is_(None),
            ]
            if minimum_rank_score is not None:
                if not math.isfinite(minimum_rank_score):
                    raise ValueError("minimum_rank_score must be finite.")
                conditions.append(WatchPaperTable.rank_score >= minimum_rank_score)

            rows = session.execute(
                select(WatchPaperTable, PaperTable)
                .join(PaperTable, PaperTable.id == WatchPaperTable.paper_id)
                .where(*conditions)
                .options(selectinload(PaperTable.sources))
                .order_by(desc(WatchPaperTable.rank_score), WatchPaperTable.first_seen_at)
                .limit(cap)
            ).all()
            return [
                PendingNotification(
                    topic_id=topic_id,
                    paper=_to_stored_paper(paper),
                    rank_score=watch_paper.rank_score,
                    first_seen_at=watch_paper.first_seen_at,
                    last_seen_at=watch_paper.last_seen_at,
                )
                for watch_paper, paper in rows
            ]

    def mark_notified(
        self,
        topic_id: str,
        paper_ids: str | Iterable[str],
        *,
        notified_at: datetime | None = None,
    ) -> int:
        """Mark only successfully delivered topic-paper notifications as sent."""

        ids = [paper_ids] if isinstance(paper_ids, str) else list(paper_ids)
        if not ids:
            return 0
        timestamp = _as_database_time(notified_at or _utc_now())

        with self._session_scope() as session:
            rows = session.scalars(
                select(WatchPaperTable).where(
                    WatchPaperTable.watch_topic_id == topic_id,
                    WatchPaperTable.paper_id.in_(ids),
                    WatchPaperTable.notified_at.is_(None),
                )
            ).all()
            for row in rows:
                row.notified_at = timestamp
            return len(rows)

    def list_digest_candidates(
        self,
        period_start: datetime,
        period_end: datetime,
        *,
        limit: int | None = None,
    ) -> list[DigestCandidate]:
        """Read persisted discoveries only; this method never triggers search."""

        start = _as_database_time(period_start)
        end = _as_database_time(period_end)
        if end <= start:
            raise ValueError("Digest period_end must be after period_start.")
        if limit is not None:
            _validate_limit(limit)

        with self._session_scope() as session:
            rows = session.scalars(
                select(PaperTable)
                .where(
                    or_(
                        and_(
                            PaperTable.first_discovered_at >= start,
                            PaperTable.first_discovered_at < end,
                        ),
                        PaperTable.watch_discoveries.any(
                            and_(
                                WatchPaperTable.first_seen_at >= start,
                                WatchPaperTable.first_seen_at < end,
                            )
                        ),
                    )
                )
                .options(
                    selectinload(PaperTable.sources),
                    selectinload(PaperTable.paper_card),
                    selectinload(PaperTable.watch_discoveries).selectinload(
                        WatchPaperTable.watch_topic
                    ),
                )
            ).all()

            candidates: list[DigestCandidate] = []
            for row in rows:
                discoveries = [
                    discovery
                    for discovery in row.watch_discoveries
                    if start <= discovery.first_seen_at < end
                ]
                topic_names = tuple(
                    sorted({discovery.watch_topic.name for discovery in discoveries})
                )
                highest_rank = (
                    max(discovery.rank_score for discovery in discoveries) if discoveries else None
                )
                candidates.append(
                    DigestCandidate(
                        paper=_to_stored_paper(row),
                        watch_topic_names=topic_names,
                        highest_rank_score=highest_rank,
                        paper_card=_to_paper_card(row.paper_card)
                        if row.paper_card is not None
                        else None,
                    )
                )

            candidates.sort(
                key=lambda candidate: (
                    -(candidate.highest_rank_score or -1.0),
                    -candidate.paper.first_discovered_at.timestamp(),
                    candidate.paper.title.casefold(),
                )
            )
            return candidates if limit is None else candidates[:limit]

    def get_last_successful_digest_end(self) -> datetime | None:
        """Return the end of the most recently sent digest period."""

        with self._session_scope() as session:
            return session.scalar(
                select(DigestRunTable.period_end)
                .where(DigestRunTable.status.in_(("sent", "success")))
                .order_by(desc(DigestRunTable.period_end))
                .limit(1)
            )

    def claim_digest_run(self, period_start: datetime, period_end: datetime) -> DigestRun | None:
        """Atomically claim a period unless it is already running or sent."""

        start = _as_database_time(period_start)
        end = _as_database_time(period_end)
        if end <= start:
            raise ValueError("Digest period_end must be after period_start.")
        now = _utc_now()

        with self._session_scope() as session:
            row = self._find_digest_run(session, start, end)
            if row is not None and row.status in {"running", "sent", "success"}:
                return None
            if row is None:
                row = DigestRunTable(
                    id=_new_id(),
                    period_start=start,
                    period_end=end,
                    status="running",
                    paper_count=0,
                    created_at=now,
                    sent_at=None,
                    safe_error=None,
                )
                session.add(row)
            else:
                row.status = "running"
                row.safe_error = None
                row.sent_at = None
            session.flush()
            return _to_digest_run(row)

    def record_digest_run(
        self,
        period_start: datetime,
        period_end: datetime,
        *,
        status: str,
        paper_count: int = 0,
        sent_at: datetime | None = None,
        safe_error: str | None = None,
    ) -> DigestRun:
        """Create or update a digest-run audit record for one exact period."""

        start = _as_database_time(period_start)
        end = _as_database_time(period_end)
        clean_status = _require_text(status, "Digest status")
        if end <= start:
            raise ValueError("Digest period_end must be after period_start.")
        if paper_count < 0:
            raise ValueError("paper_count cannot be negative.")
        now = _utc_now()

        with self._session_scope() as session:
            row = self._find_digest_run(session, start, end)
            if row is None:
                row = DigestRunTable(
                    id=_new_id(),
                    period_start=start,
                    period_end=end,
                    status=clean_status,
                    paper_count=paper_count,
                    created_at=now,
                    sent_at=None,
                    safe_error=None,
                )
                session.add(row)
            row.status = clean_status
            row.paper_count = paper_count
            row.sent_at = (
                _as_database_time(sent_at or now) if clean_status in {"sent", "success"} else None
            )
            row.safe_error = _bounded_error(safe_error) if safe_error else None
            session.flush()
            return _to_digest_run(row)

    def mark_digest_sent(
        self,
        run_id: str,
        paper_count: int,
        *,
        sent_at: datetime | None = None,
    ) -> bool:
        """Finalize a claimed digest only after its notification succeeds."""

        if paper_count < 0:
            raise ValueError("paper_count cannot be negative.")
        with self._session_scope() as session:
            row = session.get(DigestRunTable, run_id)
            if row is None:
                return False
            row.status = "sent"
            row.paper_count = paper_count
            row.sent_at = _as_database_time(sent_at or _utc_now())
            row.safe_error = None
            return True

    def mark_digest_failed(self, run_id: str, safe_error: str) -> bool:
        """Keep a failed digest retryable while retaining a concise safe error."""

        with self._session_scope() as session:
            row = session.get(DigestRunTable, run_id)
            if row is None:
                return False
            row.status = "failed"
            row.sent_at = None
            row.safe_error = _bounded_error(safe_error)
            return True

    def get_scoped_corpus(self, topic: str, limit: int = 50) -> ScopedCorpusResult:
        """Deterministically select stored PaperCards and papers matching a topic query."""

        tokens = _lexical_tokens(topic)
        if not tokens:
            return ScopedCorpusResult((), (), (), (), 0)
        _validate_limit(limit)

        with self._session_scope() as session:
            rows = session.scalars(
                select(PaperTable)
                .options(
                    selectinload(PaperTable.sources),
                    selectinload(PaperTable.paper_card),
                )
                .order_by(desc(PaperTable.first_discovered_at), PaperTable.id)
            ).all()

            scored: list[tuple[int, PaperTable]] = []
            for row in rows:
                title_tokens = set(row.normalized_title.split())
                abstract_tokens = set(_lexical_tokens(row.abstract or ""))
                card_tokens = _paper_card_tokens(row.paper_card)
                title_hits = sum(token in title_tokens for token in tokens)
                abstract_hits = sum(token in abstract_tokens for token in tokens)
                card_hits = sum(token in card_tokens for token in tokens)
                score = (3 * title_hits) + (2 * abstract_hits) + card_hits
                if score > 0:
                    scored.append((score, row))

            scored.sort(
                key=lambda item: (
                    -item[0],
                    -(item[1].publication_year or 0),
                    -(item[1].citation_count or 0),
                    item[1].normalized_title,
                )
            )
            top_rows = scored[:limit]

            cards: list[StoredPaperCard] = []
            papers: list[StoredPaper] = []
            corpus_paper_ids: list[str] = []
            missing_cards: list[str] = []

            for _, paper_row in top_rows:
                stored_paper = _to_stored_paper(paper_row)
                papers.append(stored_paper)
                corpus_paper_ids.append(stored_paper.id)
                if paper_row.paper_card is not None:
                    cards.append(_to_stored_paper_card(paper_row.paper_card))
                else:
                    missing_cards.append(stored_paper.id)

            return ScopedCorpusResult(
                cards=tuple(cards),
                papers=tuple(papers),
                corpus_paper_ids=tuple(corpus_paper_ids),
                missing_cards_paper_ids=tuple(missing_cards),
                total_matching_papers=len(scored),
            )

    def save_candidate(self, candidate: CandidateGap) -> CandidateGap:
        """Persist or update a CandidateGap and its provenance snapshot."""

        now = _utc_now()
        with self._session_scope() as session:
            row = session.get(GapCandidateTable, candidate.id)
            prov_dict = candidate.provenance.model_dump(mode="json")
            if row is None:
                row = GapCandidateTable(
                    id=candidate.id,
                    title=candidate.title,
                    description=candidate.description,
                    gap_type=candidate.gap_type,
                    research_question=candidate.research_question,
                    supporting_papers=list(candidate.supporting_papers),
                    conflicting_papers=list(candidate.conflicting_papers),
                    evidence_count=candidate.evidence_count,
                    novelty_score=candidate.novelty_score,
                    evidence_score=candidate.evidence_score,
                    importance_score=candidate.importance_score,
                    feasibility_score=candidate.feasibility_score,
                    confidence=candidate.confidence,
                    search_scope=candidate.search_scope,
                    caveats=list(candidate.caveats),
                    provenance=prov_dict,
                    review_status=candidate.review_status,
                    created_at=candidate.created_at or now,
                    updated_at=now,
                )
                session.add(row)
            else:
                row.title = candidate.title
                row.description = candidate.description
                row.gap_type = candidate.gap_type
                row.research_question = candidate.research_question
                row.supporting_papers = list(candidate.supporting_papers)
                row.conflicting_papers = list(candidate.conflicting_papers)
                row.evidence_count = candidate.evidence_count
                row.novelty_score = candidate.novelty_score
                row.evidence_score = candidate.evidence_score
                row.importance_score = candidate.importance_score
                row.feasibility_score = candidate.feasibility_score
                row.confidence = candidate.confidence
                row.search_scope = candidate.search_scope
                row.caveats = list(candidate.caveats)
                row.provenance = prov_dict
                row.review_status = candidate.review_status
                row.updated_at = now
            return candidate

    def get_candidate(self, candidate_id: str) -> CandidateGap | None:
        """Retrieve a candidate gap by its ID."""

        with self._session_scope() as session:
            row = session.get(GapCandidateTable, candidate_id)
            if row is None:
                return None
            return _to_candidate_gap(row)

    def list_candidates(
        self, gap_type: str | None = None, limit: int = 50
    ) -> list[CandidateGap]:
        """List candidate gaps ordered by creation time."""

        _validate_limit(limit)
        with self._session_scope() as session:
            stmt = select(GapCandidateTable).order_by(desc(GapCandidateTable.created_at))
            if gap_type:
                stmt = stmt.where(GapCandidateTable.gap_type == gap_type)
            rows = session.scalars(stmt.limit(limit)).all()
            return [_to_candidate_gap(row) for row in rows]

    def save_critic_review(self, review: CriticReview) -> CriticReview:
        """Append a CriticReview record to the audit trail."""

        now = _utc_now()
        with self._session_scope() as session:
            candidate = session.get(GapCandidateTable, review.candidate_id)
            if candidate is None:
                raise StorageError(f"CandidateGap {review.candidate_id} does not exist.")
            review_id = _new_id()
            row = GapReviewTable(
                id=review_id,
                candidate_id=review.candidate_id,
                review_version=review.review_version,
                queries_used=list(review.queries_used),
                retrieval_records=[r.model_dump(mode="json") for r in review.retrieval_records],
                new_paper_ids=list(review.new_paper_ids),
                overlapping_paper_ids=list(review.overlapping_paper_ids),
                decision=review.decision,
                rationale=review.rationale,
                caveats=list(review.caveats),
                created_at=review.created_at or now,
            )
            session.add(row)
        return review

    def list_critic_reviews(self, candidate_id: str) -> list[CriticReview]:
        """List all CriticReviews for a candidate gap in version order."""

        with self._session_scope() as session:
            rows = session.scalars(
                select(GapReviewTable)
                .where(GapReviewTable.candidate_id == candidate_id)
                .order_by(GapReviewTable.review_version)
            ).all()
            return [_to_critic_review(row) for row in rows]

    def update_candidate_status(
        self,
        candidate_id: str,
        status: str,
        *,
        caveats: list[str] | None = None,
        confidence: float | None = None,
        provenance: GapProvenance | None = None,
    ) -> bool:
        """Update candidate review status, caveats, and confidence without destroying provenance."""

        now = _utc_now()
        with self._session_scope() as session:
            row = session.get(GapCandidateTable, candidate_id)
            if row is None:
                return False
            row.review_status = status
            row.updated_at = now
            if caveats is not None:
                row.caveats = list(caveats)
            if confidence is not None:
                row.confidence = confidence
            if provenance is not None:
                row.provenance = provenance.model_dump(mode="json")
            return True

    @contextmanager
    def _session_scope(self) -> Iterable[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as error:
            session.rollback()
            logger.exception("ResearchRadar storage transaction failed.")
            raise StorageError("Research memory could not be updated.") from error
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _find_paper_candidates(
        self,
        session: Session,
        *,
        source_pairs: tuple[tuple[str, str], ...],
        identity_keys: tuple[str, ...],
        normalized_title: str | None,
        normalized_doi: str | None,
    ) -> list[PaperTable]:
        source_conditions = [
            and_(PaperSourceTable.provider == provider, PaperSourceTable.external_id == external_id)
            for provider, external_id in source_pairs
        ]
        source_paper_ids: set[str] = set()
        if source_conditions:
            source_paper_ids.update(
                session.scalars(
                    select(PaperSourceTable.paper_id).where(or_(*source_conditions))
                ).all()
            )

        paper_conditions = [PaperTable.canonical_key.in_(identity_keys)]
        if source_paper_ids:
            paper_conditions.append(PaperTable.id.in_(source_paper_ids))
        if normalized_doi:
            paper_conditions.append(PaperTable.doi == normalized_doi)
        if normalized_title is not None:
            paper_conditions.append(PaperTable.normalized_title == normalized_title)

        rows = session.scalars(select(PaperTable).where(or_(*paper_conditions))).all()
        return sorted(rows, key=lambda row: (row.created_at, row.id))

    def _merge_existing_papers(
        self,
        session: Session,
        target: PaperTable,
        duplicates: Iterable[PaperTable],
    ) -> None:
        for duplicate in duplicates:
            self._merge_paper_row_values(target, duplicate)
            self._move_sources(session, duplicate.id, target.id)
            self._move_watch_discoveries(session, duplicate.id, target.id)
            self._move_paper_card(session, duplicate.id, target.id)
            session.delete(duplicate)
        session.flush()

    def _merge_incoming_paper(
        self,
        target: PaperTable,
        paper: Paper,
        normalized_title: str,
        normalized_doi: str | None,
        now: datetime,
    ) -> None:
        target.title = _prefer_richer_text(target.title, paper.title)
        target.normalized_title = _normalize_title(target.title) or normalized_title
        target.abstract = _prefer_richer_text(target.abstract, paper.abstract)
        if len(paper.authors) > len(target.authors or []):
            target.authors = list(paper.authors)
        if target.publication_year is None:
            target.publication_year = paper.publication_year
        if target.venue is None and paper.venue:
            target.venue = paper.venue
        if target.doi is None:
            target.doi = normalized_doi
        if target.url is None and paper.url:
            target.url = paper.url
        if paper.citation_count is not None:
            target.citation_count = max(target.citation_count or 0, paper.citation_count)
        target.updated_at = now

    def _merge_paper_row_values(self, target: PaperTable, duplicate: PaperTable) -> None:
        target.title = _prefer_richer_text(target.title, duplicate.title)
        target.normalized_title = _normalize_title(target.title)
        target.abstract = _prefer_richer_text(target.abstract, duplicate.abstract)
        if len(duplicate.authors or []) > len(target.authors or []):
            target.authors = list(duplicate.authors)
        if target.publication_year is None:
            target.publication_year = duplicate.publication_year
        if target.venue is None:
            target.venue = duplicate.venue
        if target.doi is None:
            target.doi = duplicate.doi
        if target.url is None:
            target.url = duplicate.url
        if duplicate.citation_count is not None:
            target.citation_count = max(target.citation_count or 0, duplicate.citation_count)
        target.first_discovered_at = min(target.first_discovered_at, duplicate.first_discovered_at)
        target.updated_at = max(target.updated_at, duplicate.updated_at)

    def _move_sources(self, session: Session, from_paper_id: str, to_paper_id: str) -> None:
        rows = session.scalars(
            select(PaperSourceTable).where(PaperSourceTable.paper_id == from_paper_id)
        ).all()
        for row in rows:
            existing = session.scalar(
                select(PaperSourceTable).where(
                    PaperSourceTable.paper_id == to_paper_id,
                    PaperSourceTable.provider == row.provider,
                    PaperSourceTable.external_id == row.external_id,
                )
            )
            if existing is None:
                row.paper_id = to_paper_id
            else:
                existing.retrieved_at = max(existing.retrieved_at, row.retrieved_at)
                existing.source_url = existing.source_url or row.source_url
                session.delete(row)

    def _move_watch_discoveries(
        self,
        session: Session,
        from_paper_id: str,
        to_paper_id: str,
    ) -> None:
        rows = session.scalars(
            select(WatchPaperTable).where(WatchPaperTable.paper_id == from_paper_id)
        ).all()
        for row in rows:
            key = {"watch_topic_id": row.watch_topic_id, "paper_id": to_paper_id}
            existing = session.get(WatchPaperTable, key)
            if existing is None:
                row.paper_id = to_paper_id
                continue
            existing.first_seen_at = min(existing.first_seen_at, row.first_seen_at)
            existing.last_seen_at = max(existing.last_seen_at, row.last_seen_at)
            existing.rank_score = max(existing.rank_score, row.rank_score)
            if existing.notified_at is None:
                existing.notified_at = row.notified_at
            elif row.notified_at is not None:
                existing.notified_at = min(existing.notified_at, row.notified_at)
            session.delete(row)

    def _move_paper_card(self, session: Session, from_paper_id: str, to_paper_id: str) -> None:
        source_card = session.get(PaperCardTable, from_paper_id)
        if source_card is None:
            return
        target_card = session.get(PaperCardTable, to_paper_id)
        if target_card is None:
            source_card.paper_id = to_paper_id
            return
        if source_card.updated_at > target_card.updated_at:
            _copy_card_row(target_card, source_card)
        session.delete(source_card)

    def _adopt_canonical_key(
        self,
        session: Session,
        target: PaperTable,
        canonical_key: str,
    ) -> None:
        if target.canonical_key == canonical_key:
            return
        conflict = session.scalar(
            select(PaperTable.id).where(
                PaperTable.canonical_key == canonical_key,
                PaperTable.id != target.id,
            )
        )
        if conflict is None:
            target.canonical_key = canonical_key

    def _upsert_sources(
        self,
        session: Session,
        target: PaperTable,
        source_pairs: Iterable[tuple[str, str]],
        source_url: str | None,
        retrieved_at: datetime,
    ) -> None:
        for provider, external_id in source_pairs:
            row = session.scalar(
                select(PaperSourceTable).where(
                    PaperSourceTable.provider == provider,
                    PaperSourceTable.external_id == external_id,
                )
            )
            if row is None:
                session.add(
                    PaperSourceTable(
                        id=_new_id(),
                        paper_id=target.id,
                        provider=provider,
                        external_id=external_id,
                        source_url=source_url,
                        retrieved_at=retrieved_at,
                    )
                )
                continue
            if row.paper_id != target.id:
                # Candidate reconciliation should already have handled this.
                # Do not reassign blindly if a future caller bypasses it.
                raise StorageError("Paper source identity is associated with another paper.")
            row.retrieved_at = retrieved_at
            if source_url:
                row.source_url = source_url

    def create_project(
        self,
        name: str,
        *,
        description: str | None = None,
        goal: str | None = None,
        keywords: list[str] | None = None,
        constraints: list[str] | None = None,
        hypotheses: list[str] | None = None,
        rejected_ideas: list[str] | None = None,
    ) -> Project:
        clean_name = _require_text(name, "Project name")
        norm_name = _normalize_project_name(clean_name)
        now = _utc_now()
        with self._session_scope() as session:
            existing = session.scalar(
                select(ProjectTable).where(ProjectTable.normalized_name == norm_name)
            )
            if existing is not None:
                raise ValueError(f"Project with name '{clean_name}' already exists.")

            proj_row = ProjectTable(
                id=_new_id(),
                name=clean_name,
                normalized_name=norm_name,
                description=description.strip() if description else None,
                goal=goal.strip() if goal else None,
                keywords=list(keywords or []),
                constraints=list(constraints or []),
                hypotheses=list(hypotheses or []),
                rejected_ideas=list(rejected_ideas or []),
                created_at=now,
                updated_at=now,
            )
            session.add(proj_row)
            session.flush()
            return _to_project(proj_row)

    def get_project(self, project_id_or_name: str) -> Project | None:
        clean = project_id_or_name.strip()
        if not clean:
            return None
        with self._session_scope() as session:
            proj_row = session.scalar(
                select(ProjectTable).where(
                    or_(
                        ProjectTable.id == clean,
                        ProjectTable.normalized_name == _normalize_project_name(clean),
                    )
                )
            )
            return _to_project(proj_row) if proj_row is not None else None

    def list_projects(self) -> list[Project]:
        with self._session_scope() as session:
            rows = session.scalars(
                select(ProjectTable).order_by(desc(ProjectTable.updated_at))
            ).all()
            return [_to_project(row) for row in rows]

    def update_project(
        self,
        project_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        goal: str | None = None,
        keywords: list[str] | None = None,
        constraints: list[str] | None = None,
        hypotheses: list[str] | None = None,
        rejected_ideas: list[str] | None = None,
    ) -> Project:
        with self._session_scope() as session:
            row = session.scalar(
                select(ProjectTable).where(
                    or_(
                        ProjectTable.id == project_id.strip(),
                        ProjectTable.normalized_name == _normalize_project_name(project_id),
                    )
                )
            )
            if row is None:
                raise ValueError(f"Project '{project_id}' not found.")

            if name is not None:
                clean_name = _require_text(name, "Project name")
                row.name = clean_name
                row.normalized_name = _normalize_project_name(clean_name)
            if description is not None:
                row.description = description.strip() or None
            if goal is not None:
                row.goal = goal.strip() or None
            if keywords is not None:
                row.keywords = list(keywords)
            if constraints is not None:
                row.constraints = list(constraints)
            if hypotheses is not None:
                row.hypotheses = list(hypotheses)
            if rejected_ideas is not None:
                row.rejected_ideas = list(rejected_ideas)

            row.updated_at = _utc_now()
            session.flush()
            return _to_project(row)

    def add_paper_to_project(
        self,
        project_id: str,
        paper_id: str,
        *,
        relation: str = "relevant",
        note: str | None = None,
    ) -> ProjectPaperLink:
        now = _utc_now()
        with self._session_scope() as session:
            proj = session.scalar(
                select(ProjectTable).where(
                    or_(
                        ProjectTable.id == project_id.strip(),
                        ProjectTable.normalized_name == _normalize_project_name(project_id),
                    )
                )
            )
            if proj is None:
                raise ValueError(f"Project '{project_id}' not found.")

            paper = session.scalar(select(PaperTable).where(PaperTable.id == paper_id.strip()))
            if paper is None:
                raise ValueError(f"Paper '{paper_id}' not found.")

            link_row = session.scalar(
                select(ProjectPaperTable).where(
                    ProjectPaperTable.project_id == proj.id,
                    ProjectPaperTable.paper_id == paper.id,
                )
            )
            if link_row is None:
                link_row = ProjectPaperTable(
                    project_id=proj.id,
                    paper_id=paper.id,
                    relation=relation.strip() or "relevant",
                    note=note.strip() if note else None,
                    created_at=now,
                )
                session.add(link_row)
            else:
                link_row.relation = relation.strip() or "relevant"
                if note is not None:
                    link_row.note = note.strip() or None

            proj.updated_at = now
            session.flush()
            return ProjectPaperLink(
                project_id=link_row.project_id,
                paper_id=link_row.paper_id,
                relation=link_row.relation,
                note=link_row.note,
                created_at=link_row.created_at,
            )

    def list_project_papers(self, project_id: str) -> list[ProjectPaperLink]:
        with self._session_scope() as session:
            proj = session.scalar(
                select(ProjectTable).where(
                    or_(
                        ProjectTable.id == project_id.strip(),
                        ProjectTable.normalized_name == _normalize_project_name(project_id),
                    )
                )
            )
            if proj is None:
                return []
            rows = session.scalars(
                select(ProjectPaperTable).where(ProjectPaperTable.project_id == proj.id)
            ).all()
            return [
                ProjectPaperLink(
                    project_id=row.project_id,
                    paper_id=row.paper_id,
                    relation=row.relation,
                    note=row.note,
                    created_at=row.created_at,
                )
                for row in rows
            ]

    def add_gap_to_project(
        self,
        project_id: str,
        candidate_id: str,
        *,
        status: str = "active",
    ) -> ProjectGapLink:
        now = _utc_now()
        with self._session_scope() as session:
            proj = session.scalar(
                select(ProjectTable).where(
                    or_(
                        ProjectTable.id == project_id.strip(),
                        ProjectTable.normalized_name == _normalize_project_name(project_id),
                    )
                )
            )
            if proj is None:
                raise ValueError(f"Project '{project_id}' not found.")

            cand = session.scalar(
                select(GapCandidateTable).where(GapCandidateTable.id == candidate_id.strip())
            )
            if cand is None:
                raise ValueError(f"CandidateGap '{candidate_id}' not found.")

            link_row = session.scalar(
                select(ProjectGapTable).where(
                    ProjectGapTable.project_id == proj.id,
                    ProjectGapTable.candidate_id == cand.id,
                )
            )
            if link_row is None:
                link_row = ProjectGapTable(
                    project_id=proj.id,
                    candidate_id=cand.id,
                    status=status.strip() or "active",
                    created_at=now,
                )
                session.add(link_row)
            else:
                link_row.status = status.strip() or "active"

            proj.updated_at = now
            session.flush()
            return ProjectGapLink(
                project_id=link_row.project_id,
                candidate_id=link_row.candidate_id,
                status=link_row.status,
                created_at=link_row.created_at,
            )

    def list_project_gaps(self, project_id: str) -> list[ProjectGapLink]:
        with self._session_scope() as session:
            proj = session.scalar(
                select(ProjectTable).where(
                    or_(
                        ProjectTable.id == project_id.strip(),
                        ProjectTable.normalized_name == _normalize_project_name(project_id),
                    )
                )
            )
            if proj is None:
                return []
            rows = session.scalars(
                select(ProjectGapTable).where(ProjectGapTable.project_id == proj.id)
            ).all()
            return [
                ProjectGapLink(
                    project_id=row.project_id,
                    candidate_id=row.candidate_id,
                    status=row.status,
                    created_at=row.created_at,
                )
                for row in rows
            ]

    @staticmethod
    def _find_digest_run(
        session: Session,
        period_start: datetime,
        period_end: datetime,
    ) -> DigestRunTable | None:
        return session.scalar(
            select(DigestRunTable).where(
                DigestRunTable.period_start == period_start,
                DigestRunTable.period_end == period_end,
            )
        )


def _to_paper_source(row: PaperSourceTable) -> PaperSource:
    return PaperSource(
        provider=row.provider,
        external_id=row.external_id,
        source_url=row.source_url,
        retrieved_at=row.retrieved_at,
    )


def _to_stored_paper(row: PaperTable) -> StoredPaper:
    sources = tuple(
        _to_paper_source(source)
        for source in sorted(row.sources, key=lambda item: (item.provider, item.external_id))
    )
    return StoredPaper(
        id=row.id,
        canonical_key=row.canonical_key,
        title=row.title,
        abstract=row.abstract,
        authors=list(row.authors or []),
        publication_year=row.publication_year,
        venue=row.venue,
        doi=row.doi,
        url=row.url,
        citation_count=row.citation_count,
        primary_source=row.primary_source,
        sources=sources,
        first_discovered_at=row.first_discovered_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_project(row: ProjectTable) -> Project:
    return Project(
        id=row.id,
        name=row.name,
        description=row.description,
        goal=row.goal,
        keywords=list(row.keywords or []),
        constraints=list(row.constraints or []),
        hypotheses=list(row.hypotheses or []),
        rejected_ideas=list(row.rejected_ideas or []),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _normalize_project_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not cleaned:
        raise ValueError("Project name must contain valid alphanumeric characters.")
    return cleaned


def _parse_structured_evidence(items: Iterable[object] | None) -> list[StructuredEvidence]:
    parsed: list[StructuredEvidence] = []
    for item in items or []:
        if isinstance(item, Mapping):
            parsed.append(StructuredEvidence.model_validate(item))
    return parsed


def _to_paper_card(row: PaperCardTable) -> PaperCard:
    return PaperCard(
        paper_id=row.paper_id,
        problem=row.problem,
        motivation=row.motivation,
        contributions=list(row.contributions or []),
        methods=list(row.methods or []),
        datasets=list(row.datasets or []),
        metrics=list(row.metrics or []),
        tasks=_parse_structured_evidence(getattr(row, "tasks", None)),
        modalities=_parse_structured_evidence(getattr(row, "modalities", None)),
        evaluation_conditions=_parse_structured_evidence(
            getattr(row, "evaluation_conditions", None)
        ),
        main_claims=list(row.main_claims or []),
        limitations=list(row.limitations or []),
        future_work=list(row.future_work or []),
        failure_cases=list(row.failure_cases or []),
    )


def _to_stored_paper_card(row: PaperCardTable) -> StoredPaperCard:
    return StoredPaperCard(
        card=_to_paper_card(row),
        source_url=row.source_url,
        document_sha256=row.document_sha256,
        selected_sections=tuple(row.selected_sections or []),
        llm_provider=row.llm_provider,
        llm_model=row.llm_model,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_candidate_gap(row: GapCandidateTable) -> CandidateGap:
    return CandidateGap(
        id=row.id,
        title=row.title,
        description=row.description,
        gap_type=row.gap_type,  # type: ignore[arg-type]
        research_question=row.research_question,
        supporting_papers=list(row.supporting_papers or []),
        conflicting_papers=list(row.conflicting_papers or []),
        evidence_count=row.evidence_count,
        novelty_score=row.novelty_score,
        evidence_score=row.evidence_score,
        importance_score=row.importance_score,
        feasibility_score=row.feasibility_score,
        confidence=row.confidence,
        search_scope=row.search_scope,
        caveats=list(row.caveats or []),
        provenance=GapProvenance.model_validate(row.provenance),
        review_status=row.review_status,  # type: ignore[arg-type]
        created_at=row.created_at,
    )


def _to_critic_review(row: GapReviewTable) -> CriticReview:
    return CriticReview(
        candidate_id=row.candidate_id,
        review_version=row.review_version,
        queries_used=list(row.queries_used or []),
        retrieval_records=[
            RetrievalRecord.model_validate(item)
            for item in (row.retrieval_records or [])
        ],
        new_paper_ids=list(row.new_paper_ids or []),
        overlapping_paper_ids=list(row.overlapping_paper_ids or []),
        decision=row.decision,  # type: ignore[arg-type]
        rationale=row.rationale,
        caveats=list(row.caveats or []),
        created_at=row.created_at,
    )


def _to_watch_topic(row: WatchTopicTable) -> WatchTopic:
    return WatchTopic(
        id=row.id,
        name=row.name,
        query=row.query,
        enabled=row.enabled,
        created_at=row.created_at,
        last_scan_at=row.last_scan_at,
        last_error=row.last_error,
    )


def _to_digest_run(row: DigestRunTable) -> DigestRun:
    return DigestRun(
        id=row.id,
        period_start=row.period_start,
        period_end=row.period_end,
        status=row.status,
        paper_count=row.paper_count,
        created_at=row.created_at,
        sent_at=row.sent_at,
        safe_error=row.safe_error,
    )


def _copy_card_values(row: PaperCardTable, card: PaperCard) -> None:
    row.problem = card.problem
    row.motivation = card.motivation
    row.contributions = list(card.contributions)
    row.methods = list(card.methods)
    row.datasets = list(card.datasets)
    row.metrics = list(card.metrics)
    row.tasks = [item.model_dump(mode="json") for item in card.tasks]
    row.modalities = [item.model_dump(mode="json") for item in card.modalities]
    row.evaluation_conditions = [
        item.model_dump(mode="json") for item in card.evaluation_conditions
    ]
    row.main_claims = [claim.model_dump(mode="json") for claim in card.main_claims]
    row.limitations = list(card.limitations)
    row.future_work = list(card.future_work)
    row.failure_cases = list(card.failure_cases)


def _copy_card_row(target: PaperCardTable, source: PaperCardTable) -> None:
    target.problem = source.problem
    target.motivation = source.motivation
    target.contributions = list(source.contributions or [])
    target.methods = list(source.methods or [])
    target.datasets = list(source.datasets or [])
    target.metrics = list(source.metrics or [])
    target.tasks = list(getattr(source, "tasks", None) or [])
    target.modalities = list(getattr(source, "modalities", None) or [])
    target.evaluation_conditions = list(getattr(source, "evaluation_conditions", None) or [])
    target.main_claims = list(source.main_claims or [])
    target.limitations = list(source.limitations or [])
    target.future_work = list(source.future_work or [])
    target.failure_cases = list(source.failure_cases or [])
    target.source_url = source.source_url
    target.document_sha256 = source.document_sha256
    target.selected_sections = list(source.selected_sections or [])
    target.llm_provider = source.llm_provider
    target.llm_model = source.llm_model
    target.created_at = min(target.created_at, source.created_at)
    target.updated_at = source.updated_at


def _paper_source_pairs(paper: Paper) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    primary_provider = _normalize_provider(paper.source)
    primary_id = _normalize_external_id(primary_provider, paper.id)
    if primary_id:
        pairs.append((primary_provider, primary_id))

    for provider, external_id in paper.external_ids.items():
        normalized_provider = _normalize_provider(provider)
        normalized_id = _normalize_external_id(normalized_provider, external_id)
        if normalized_id:
            pairs.append((normalized_provider, normalized_id))

    normalized_doi = _normalize_doi(paper.doi)
    if normalized_doi:
        pairs.append(("doi", normalized_doi))

    deduplicated: dict[tuple[str, str], None] = {}
    for pair in pairs:
        deduplicated[pair] = None
    return tuple(deduplicated)


def _identity_keys(paper: Paper, source_pairs: Iterable[tuple[str, str]]) -> tuple[str, ...]:
    """Return all deterministic identities useful for cross-scan reconciliation."""

    pairs = tuple(source_pairs)
    keys: list[str] = []
    normalized_doi = _normalize_doi(paper.doi) or _source_doi(pairs)
    if normalized_doi:
        keys.append(f"doi:{normalized_doi}")
    for provider, external_id in pairs:
        if provider == "arxiv":
            keys.append(f"arxiv:{_normalize_arxiv_identity(external_id)}")
    for provider, external_id in pairs:
        if provider not in {"doi", "arxiv"}:
            keys.append(f"source:{provider}:{external_id.casefold()}")
    if not keys:
        keys.append(f"title:{_normalize_title(paper.title)}")
    return tuple(dict.fromkeys(keys))


def _source_doi(source_pairs: Iterable[tuple[str, str]]) -> str | None:
    for provider, external_id in source_pairs:
        if provider == "doi":
            return _normalize_doi(external_id)
    return None


def _normalize_provider(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", unicodedata.normalize("NFKC", value).casefold())
    normalized = normalized.strip("_")
    aliases = {
        "arxiv_id": "arxiv",
        "open_alex": "openalex",
        "semantic_scholar_id": "semantic_scholar",
    }
    return aliases.get(normalized, normalized) or "unknown"


def _normalize_external_id(provider: str, value: str) -> str:
    cleaned = unicodedata.normalize("NFKC", str(value)).strip()
    if not cleaned:
        return ""
    if provider == "doi":
        return _normalize_doi(cleaned) or ""
    if provider == "openalex":
        cleaned = re.sub(r"^https?://openalex\.org/", "", cleaned, flags=re.IGNORECASE)
    if provider == "arxiv":
        cleaned = re.sub(
            r"^https?://arxiv\.org/(?:abs|pdf)/",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
    prefixes = (
        f"{provider}:",
        f"{provider.replace('_', '-')}:",
        f"{provider.replace('_', ' ')}:",
    )
    for prefix in prefixes:
        if cleaned.casefold().startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break
    return re.sub(r"\s+", " ", cleaned)


def _normalize_doi(value: str | None) -> str | None:
    if value is None:
        return None
    doi = unicodedata.normalize("NFKC", value).strip()
    if not doi:
        return None
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    doi = doi.strip().rstrip("/.,;")
    return doi.casefold() or None


def _normalize_arxiv_identity(value: str) -> str:
    arxiv_id = unicodedata.normalize("NFKC", value).strip()
    arxiv_id = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", arxiv_id, flags=re.IGNORECASE)
    arxiv_id = re.sub(r"^arxiv:\s*", "", arxiv_id, flags=re.IGNORECASE)
    arxiv_id = arxiv_id.removesuffix(".pdf")
    return re.sub(r"v\d+$", "", arxiv_id, flags=re.IGNORECASE).casefold()


def _normalize_title(value: str) -> str:
    pieces: list[str] = []
    for character in unicodedata.normalize("NFKC", value).casefold():
        pieces.append(" " if unicodedata.category(character).startswith("P") else character)
    return " ".join("".join(pieces).split())


def _title_identity(normalized_title: str) -> str | None:
    """Avoid merging unrelated records on very short, generic-looking titles."""

    return normalized_title if len(normalized_title) >= 8 else None


def _normalize_watch_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _normalize_search_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _lexical_tokens(query: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(token for token in _normalize_search_text(query).split() if token))


def _paper_card_tokens(card: PaperCardTable | None) -> set[str]:
    if card is None:
        return set()
    values: list[str] = [card.problem or "", card.motivation or ""]
    for field in (
        card.contributions,
        card.methods,
        card.datasets,
        card.metrics,
        card.limitations,
        card.future_work,
        card.failure_cases,
    ):
        values.extend(str(item) for item in field or [])
    for struct_field in (
        getattr(card, "tasks", None),
        getattr(card, "modalities", None),
        getattr(card, "evaluation_conditions", None),
    ):
        for item in struct_field or []:
            if isinstance(item, Mapping):
                values.append(str(item.get("value") or ""))
    for claim in card.main_claims or []:
        if isinstance(claim, Mapping):
            values.extend(str(claim.get(key) or "") for key in ("claim", "supporting_text"))
    return set(_lexical_tokens(" ".join(values)))


def _normalize_selected_sections(
    selected_sections: Iterable[str] | Mapping[str, object] | None,
) -> list[str]:
    if selected_sections is None:
        return []
    values = (
        selected_sections.keys() if isinstance(selected_sections, Mapping) else selected_sections
    )
    return list(dict.fromkeys(_require_text(str(value), "Section name") for value in values))


def _prefer_richer_text(existing: str | None, incoming: str | None) -> str | None:
    if not existing:
        return incoming
    if not incoming:
        return existing
    return incoming if len(incoming.strip()) > len(existing.strip()) else existing


def _require_text(value: str, label: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{label} cannot be empty.")
    return cleaned


def _validate_limit(value: int) -> None:
    if value < 1:
        raise ValueError("limit must be at least 1.")


def _bounded_error(value: str) -> str:
    return " ".join(str(value).split())[:1000] or "Unknown storage operation failure."


def _utc_now() -> datetime:
    """Return a naive UTC timestamp, matching SQLite's portable DateTime form."""

    return datetime.now(UTC).replace(tzinfo=None)


def _as_database_time(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _new_id() -> str:
    return str(uuid4())
