"""Short-lived-session storage accessors for ingestion runs and document artifacts."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import desc, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from research_radar.artifacts.base import ArtifactRef
from research_radar.storage.database import Database
from research_radar.storage.repositories import StorageError
from research_radar.storage.tables import (
    DocumentArtifactTable,
    IngestionRunTable,
    ProviderRetrievalTable,
)

logger = logging.getLogger(__name__)

_MAX_EXTERNAL_IDS = 200
_MAX_SAFE_ERROR_LENGTH = 500
_MAX_NORMALIZED_QUERY_LENGTH = 1000


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """A persisted content-addressed artifact reference for one paper version."""

    id: str
    paper_id: str
    sha256: str
    artifact_type: str
    backend: str
    object_key: str
    mime_type: str
    byte_size: int
    source_url: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class IngestionRunRecord:
    """One bounded discovery run with aggregate, non-sensitive outcome data."""

    id: str
    query: str
    normalized_query: str
    status: str
    started_at: datetime
    completed_at: datetime | None
    requested_limit: int
    discovered_count: int
    canonical_count: int
    providers: list[str]
    project_id: str | None
    safe_error: str | None


@dataclass(frozen=True, slots=True)
class ProviderRetrievalRecord:
    """Per-provider retrieval provenance recorded during one ingestion run."""

    id: str
    run_id: str
    provider: str
    query: str
    external_ids: list[str]
    retrieved_at: datetime
    status: str
    result_count: int
    safe_error: str | None


class IngestionRepository:
    """SQLAlchemy query and transaction boundaries for ingestion memory.

    Mirrors ``ResearchRepository``: every method opens a short-lived session,
    commits on success, and rolls back into a ``StorageError`` on failure.
    """

    def __init__(self, database: Database | sessionmaker[Session]) -> None:
        self._session_factory = (
            database.session_factory if isinstance(database, Database) else database
        )

    def record_artifact(
        self,
        ref: ArtifactRef,
        *,
        created_at: datetime | None = None,
    ) -> StoredArtifact:
        """Idempotently persist one artifact reference keyed by its content hash.

        An existing row for ``(paper_id, sha256, artifact_type)`` has its
        location fields refreshed in place; no duplicate row is created.
        """

        timestamp = _as_database_time(created_at or _utc_now())
        with self._session_scope() as session:
            row = session.scalar(
                select(DocumentArtifactTable).where(
                    DocumentArtifactTable.paper_id == ref.paper_id,
                    DocumentArtifactTable.sha256 == ref.sha256,
                    DocumentArtifactTable.artifact_type == ref.artifact_type,
                )
            )
            if row is None:
                row = DocumentArtifactTable(
                    id=uuid4().hex,
                    paper_id=ref.paper_id,
                    sha256=ref.sha256,
                    artifact_type=ref.artifact_type,
                    backend=ref.backend,
                    object_key=ref.object_key,
                    mime_type=ref.mime_type,
                    byte_size=ref.byte_size,
                    source_url=ref.source_url,
                    created_at=timestamp,
                )
                session.add(row)
            else:
                row.object_key = ref.object_key
                row.mime_type = ref.mime_type
                row.byte_size = ref.byte_size
                row.source_url = ref.source_url
            session.flush()
            return _to_stored_artifact(row)

    def get_artifact(self, paper_id: str, sha256: str, artifact_type: str) -> StoredArtifact | None:
        """Return one stored artifact by its identity, or ``None`` when absent."""

        with self._session_scope() as session:
            row = session.scalar(
                select(DocumentArtifactTable).where(
                    DocumentArtifactTable.paper_id == paper_id,
                    DocumentArtifactTable.sha256 == sha256,
                    DocumentArtifactTable.artifact_type == artifact_type,
                )
            )
            return _to_stored_artifact(row) if row is not None else None

    def list_artifacts(self, paper_id: str) -> list[StoredArtifact]:
        """List every stored artifact version for a paper in stable order."""

        with self._session_scope() as session:
            rows = session.scalars(
                select(DocumentArtifactTable)
                .where(DocumentArtifactTable.paper_id == paper_id)
                .order_by(DocumentArtifactTable.created_at, DocumentArtifactTable.artifact_type)
            ).all()
            return [_to_stored_artifact(row) for row in rows]

    def count_artifacts(self) -> int:
        """Return the total number of stored document-artifact rows."""

        with self._session_scope() as session:
            return int(session.scalar(select(func.count()).select_from(DocumentArtifactTable)) or 0)

    def start_ingestion_run(
        self,
        *,
        query: str,
        requested_limit: int,
        providers: list[str],
        project_id: str | None = None,
    ) -> IngestionRunRecord:
        """Open a new discovery run in the ``running`` state and return it."""

        now = _utc_now()
        normalized_query = " ".join(query.casefold().split())[:_MAX_NORMALIZED_QUERY_LENGTH]
        with self._session_scope() as session:
            row = IngestionRunTable(
                id=uuid4().hex,
                query=query,
                normalized_query=normalized_query,
                status="running",
                started_at=now,
                completed_at=None,
                requested_limit=requested_limit,
                discovered_count=0,
                canonical_count=0,
                providers=list(providers),
                project_id=project_id,
                safe_error=None,
            )
            session.add(row)
            session.flush()
            return _to_ingestion_run_record(row)

    def complete_ingestion_run(
        self,
        run_id: str,
        *,
        discovered_count: int,
        canonical_count: int,
    ) -> IngestionRunRecord | None:
        """Finalize a run as ``completed`` with its aggregate counts."""

        now = _utc_now()
        with self._session_scope() as session:
            row = session.get(IngestionRunTable, run_id)
            if row is None:
                return None
            row.status = "completed"
            row.completed_at = now
            row.discovered_count = discovered_count
            row.canonical_count = canonical_count
            session.flush()
            return _to_ingestion_run_record(row)

    def fail_ingestion_run(self, run_id: str, *, safe_error: str) -> IngestionRunRecord | None:
        """Mark a run ``failed`` while retaining only a bounded safe error."""

        now = _utc_now()
        with self._session_scope() as session:
            row = session.get(IngestionRunTable, run_id)
            if row is None:
                return None
            row.status = "failed"
            row.completed_at = now
            row.safe_error = _bounded_error(safe_error)
            session.flush()
            return _to_ingestion_run_record(row)

    def record_provider_retrieval(
        self,
        *,
        run_id: str,
        provider: str,
        query: str,
        external_ids: list[str],
        status: str,
        result_count: int,
        safe_error: str | None = None,
    ) -> ProviderRetrievalRecord:
        """Append one per-provider provenance row for an ingestion run.

        Only normalized external identifiers are retained; identifiers beyond
        the first two hundred and error text beyond five hundred characters
        are dropped defensively.
        """

        now = _utc_now()
        bounded_ids = list(external_ids)[:_MAX_EXTERNAL_IDS]
        bounded_error = _bounded_error(safe_error) if safe_error else None
        with self._session_scope() as session:
            row = ProviderRetrievalTable(
                id=uuid4().hex,
                run_id=run_id,
                provider=provider,
                query=query,
                external_ids=bounded_ids,
                retrieved_at=now,
                status=status,
                result_count=result_count,
                safe_error=bounded_error,
            )
            session.add(row)
            session.flush()
            return _to_provider_retrieval_record(row)

    def list_provider_retrievals(self, run_id: str) -> list[ProviderRetrievalRecord]:
        """List recorded provider retrievals for one run in retrieval order."""

        with self._session_scope() as session:
            rows = session.scalars(
                select(ProviderRetrievalTable)
                .where(ProviderRetrievalTable.run_id == run_id)
                .order_by(ProviderRetrievalTable.retrieved_at, ProviderRetrievalTable.id)
            ).all()
            return [_to_provider_retrieval_record(row) for row in rows]

    def list_recent_runs(self, limit: int = 10) -> list[IngestionRunRecord]:
        """List the most recently started runs, newest first."""

        clamped_limit = min(max(limit, 1), 100)
        with self._session_scope() as session:
            rows = session.scalars(
                select(IngestionRunTable)
                .order_by(desc(IngestionRunTable.started_at), IngestionRunTable.id)
                .limit(clamped_limit)
            ).all()
            return [_to_ingestion_run_record(row) for row in rows]

    def count_ingestion_runs(self) -> int:
        """Return the total number of recorded ingestion runs."""

        with self._session_scope() as session:
            return int(session.scalar(select(func.count()).select_from(IngestionRunTable)) or 0)

    @contextmanager
    def _session_scope(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as error:
            session.rollback()
            logger.exception("ResearchRadar ingestion storage transaction failed.")
            raise StorageError("Ingestion memory could not be updated.") from error
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def _to_stored_artifact(row: DocumentArtifactTable) -> StoredArtifact:
    return StoredArtifact(
        id=row.id,
        paper_id=row.paper_id,
        sha256=row.sha256,
        artifact_type=row.artifact_type,
        backend=row.backend,
        object_key=row.object_key,
        mime_type=row.mime_type,
        byte_size=row.byte_size,
        source_url=row.source_url,
        created_at=row.created_at,
    )


def _to_ingestion_run_record(row: IngestionRunTable) -> IngestionRunRecord:
    return IngestionRunRecord(
        id=row.id,
        query=row.query,
        normalized_query=row.normalized_query,
        status=row.status,
        started_at=row.started_at,
        completed_at=row.completed_at,
        requested_limit=row.requested_limit,
        discovered_count=row.discovered_count,
        canonical_count=row.canonical_count,
        providers=list(row.providers or []),
        project_id=row.project_id,
        safe_error=row.safe_error,
    )


def _to_provider_retrieval_record(row: ProviderRetrievalTable) -> ProviderRetrievalRecord:
    return ProviderRetrievalRecord(
        id=row.id,
        run_id=row.run_id,
        provider=row.provider,
        query=row.query,
        external_ids=list(row.external_ids or []),
        retrieved_at=row.retrieved_at,
        status=row.status,
        result_count=row.result_count,
        safe_error=row.safe_error,
    )


def _bounded_error(value: str) -> str:
    """Collapse whitespace and cap a caller-supplied safe error summary."""

    return (
        " ".join(str(value).split())[:_MAX_SAFE_ERROR_LENGTH] or "Unknown ingestion failure."
    )


def _as_database_time(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _utc_now() -> datetime:
    """Return a naive UTC timestamp, matching SQLite's portable DateTime form."""

    return datetime.now(UTC).replace(tzinfo=None)
