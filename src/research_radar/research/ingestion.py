"""Topic ingestion that fans out to providers, canonicalizes, persists, and audits."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from research_radar.models.paper import Paper
from research_radar.research.canonical import CanonicalPaper, canonicalize
from research_radar.research.scout import ScoutResult, ScoutService
from research_radar.storage.ingestion_repository import IngestionRepository
from research_radar.storage.repositories import ResearchRepository

logger = logging.getLogger(__name__)

_MAX_AUTO_READ = 5
_MAX_RECORDED_EXTERNAL_IDS = 200


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Aggregate, safe-to-display outcome of one ingestion run."""

    run_id: str
    query: str
    discovered_count: int
    canonical_count: int
    paper_ids: list[str]
    warnings: list[str]
    provider_counts: dict[str, int]
    read_paper_ids: list[str]


class PaperReader(Protocol):
    """Anything that can read one public PDF URL into a persisted analysis."""

    async def read_url(self, url: str) -> object:
        """Return the outcome of reading the paper at ``url``."""


class IngestionService:
    """Orchestrate bounded discovery, canonicalization, persistence, and audit."""

    def __init__(
        self,
        *,
        scout: ScoutService,
        repository: ResearchRepository,
        ingestion_repository: IngestionRepository,
        reader_service: PaperReader | None = None,
        metadata_limit: int = 50,
    ) -> None:
        self._scout = scout
        self._repository = repository
        self._ingestion_repository = ingestion_repository
        self._reader_service = reader_service
        self._metadata_limit = metadata_limit

    async def ingest_research_topic(
        self,
        query: str,
        *,
        limit: int = 20,
        project_id: str | None = None,
        auto_read: int = 0,
    ) -> IngestionResult:
        """Discover, canonicalize, and persist papers for one research topic."""

        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("Research topic query cannot be empty.")
        clamped_limit = max(1, min(limit, self._metadata_limit))
        if auto_read < 0:
            raise ValueError("auto_read cannot be negative.")
        clamped_auto_read = min(auto_read, _MAX_AUTO_READ)
        provider_names = list(self._scout.provider_names)
        resolved_project_id = await self._resolve_project_id(project_id)

        run = await asyncio.to_thread(
            self._ingestion_repository.start_ingestion_run,
            query=normalized_query,
            requested_limit=clamped_limit,
            providers=provider_names,
            project_id=resolved_project_id,
        )
        try:
            raw = await self._scout.search(normalized_query, clamped_limit)
            await self._record_provider_retrievals(
                run.id, normalized_query, provider_names, raw
            )
            canonical = canonicalize(raw.papers)
            paper_ids: list[str] = []
            for entry in canonical:
                paper_ids.append(
                    await asyncio.to_thread(self._repository.upsert_merged_paper, entry.paper)
                )
            warnings = list(raw.warnings)
            if resolved_project_id is not None:
                await self._link_papers_to_project(resolved_project_id, paper_ids, warnings)
            read_paper_ids = await self._auto_read_canonical_papers(
                canonical, paper_ids, clamped_auto_read, warnings
            )
            await asyncio.to_thread(
                self._ingestion_repository.complete_ingestion_run,
                run.id,
                discovered_count=len(raw.papers),
                canonical_count=len(canonical),
            )
        except Exception as error:
            logger.warning("Ingestion run %s failed: %s", run.id, type(error).__name__)
            await asyncio.to_thread(
                self._ingestion_repository.fail_ingestion_run,
                run.id,
                safe_error=f"Ingestion failed: {type(error).__name__}.",
            )
            raise
        return IngestionResult(
            run_id=run.id,
            query=normalized_query,
            discovered_count=len(raw.papers),
            canonical_count=len(canonical),
            paper_ids=paper_ids,
            warnings=warnings,
            provider_counts=dict(raw.provider_counts),
            read_paper_ids=read_paper_ids,
        )

    async def _resolve_project_id(self, project_id_or_name: str | None) -> str | None:
        """Resolve a project reference to its storage id before the run opens.

        ``ingestion_runs.project_id`` is a foreign key, so a user-supplied
        project NAME has to become an id first. Resolving up front also fails
        an unknown project loudly instead of opening a run that cannot be
        linked to anything.
        """

        if project_id_or_name is None:
            return None
        project = await asyncio.to_thread(self._repository.get_project, project_id_or_name)
        if project is None:
            raise ValueError(f"Project '{project_id_or_name}' was not found.")
        return project.id

    async def _record_provider_retrievals(
        self,
        run_id: str,
        normalized_query: str,
        provider_names: list[str],
        raw: ScoutResult,
    ) -> None:
        """Write one provenance row per configured provider for this run."""

        for name in provider_names:
            if name in raw.provider_counts:
                status = "ok"
                result_count = raw.provider_counts[name]
                safe_error = None
                external_ids = _bare_external_ids(raw.papers, name)
            else:
                warning = next(
                    (item for item in raw.warnings if item.startswith(f"{name} ")), None
                )
                status = "failed" if warning is not None else "ok"
                result_count = 0
                safe_error = warning
                external_ids = []
            await asyncio.to_thread(
                self._ingestion_repository.record_provider_retrieval,
                run_id=run_id,
                provider=name,
                query=normalized_query,
                external_ids=external_ids[:_MAX_RECORDED_EXTERNAL_IDS],
                status=status,
                result_count=result_count,
                safe_error=safe_error,
            )

    async def _link_papers_to_project(
        self,
        project_id: str,
        paper_ids: list[str],
        warnings: list[str],
    ) -> None:
        """Link every persisted paper to a project in one bounded transaction.

        A whole run is linked at once rather than one statement per paper, and
        a link failure degrades to a warning so it never aborts an otherwise
        successful ingestion.
        """

        if not paper_ids:
            return
        try:
            linked = await asyncio.to_thread(
                self._repository.add_papers_to_project, project_id, list(paper_ids)
            )
        except Exception:
            logger.warning("Project linking failed for this ingestion run.")
            warnings.append("Papers could not be linked to the project.")
            return
        skipped = len(paper_ids) - len(linked)
        if skipped > 0:
            logger.info("Project linking skipped %d already-linked paper(s).", skipped)

    async def _auto_read_canonical_papers(
        self,
        canonical: list[CanonicalPaper],
        paper_ids: list[str],
        auto_read: int,
        warnings: list[str],
    ) -> list[str]:
        """Read at most ``auto_read`` persisted papers that expose a direct PDF URL."""

        if auto_read <= 0 or self._reader_service is None:
            return []
        read_paper_ids: list[str] = []
        for entry, paper_id in zip(canonical, paper_ids, strict=True):
            if len(read_paper_ids) >= auto_read:
                break
            pdf_url = entry.paper.external_ids.get("pdf_url")
            if not pdf_url:
                continue
            try:
                await self._reader_service.read_url(pdf_url)
            except Exception:
                logger.warning("Auto-read failed for one paper.")
                warnings.append("One selected paper could not be auto-read.")
                continue
            read_paper_ids.append(paper_id)
        return read_paper_ids


def _bare_external_ids(papers: Sequence[Paper], provider: str) -> list[str]:
    """Collect deduplicated bare identifiers contributed by ONE provider, in order.

    Only papers this provider actually returned are considered. Attributing
    another provider's identifiers to this row would make the retrieval record
    a false provenance claim.
    """

    ids: list[str] = []
    for paper in papers:
        if paper.source != provider:
            continue
        candidate = paper.external_ids.get(provider) or paper.id.split(":", maxsplit=1)[-1]
        if candidate and candidate not in ids:
            ids.append(candidate)
    return ids
