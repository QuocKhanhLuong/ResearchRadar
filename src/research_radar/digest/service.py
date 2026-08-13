"""Persisted-research daily digest assembly and delivery orchestration.

This module deliberately has no Discord, provider, or scheduler dependency.
It reads only the normalized data already held by :class:`ResearchRepository`.
The application layer may schedule ``run_scheduled_digest`` and adapt the
notification protocol to Discord (or another private delivery mechanism).
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from research_radar.storage import DigestCandidate, ResearchRepository

logger = logging.getLogger(__name__)

DEFAULT_DIGEST_WINDOW = timedelta(days=1)
DEFAULT_TOP_PAPER_LIMIT = 5
MAX_TOP_PAPER_LIMIT = 10
DEFAULT_RENDER_CHARACTER_LIMIT = 1_800


@dataclass(frozen=True, slots=True)
class PaperCardInsight:
    """One concise, deterministic insight already stored in a PaperCard.

    ``kind`` describes the structured field that supplied the text.  The
    service never asks an LLM to create or extend a digest insight.
    """

    kind: str
    text: str
    source_section: str | None = None


@dataclass(frozen=True, slots=True)
class DigestPaper:
    """A presentation-ready normalized paper selected for one digest."""

    paper_id: str
    title: str
    authors: tuple[str, ...]
    publication_year: int | None
    venue: str | None
    citation_count: int | None
    canonical_link: str | None
    rank_score: float | None
    watch_topic_names: tuple[str, ...]
    insight: PaperCardInsight | None = None


@dataclass(frozen=True, slots=True)
class WatchActivity:
    """The count of unique digest papers newly associated with one topic."""

    topic_name: str
    paper_count: int


@dataclass(frozen=True, slots=True)
class ResearchDigest:
    """Typed, provider-independent data for an on-demand or scheduled digest."""

    period_start: datetime
    period_end: datetime
    paper_count: int
    top_papers: tuple[DigestPaper, ...]
    watch_activity: tuple[WatchActivity, ...]

    def render_text(self, *, max_characters: int = DEFAULT_RENDER_CHARACTER_LIMIT) -> str:
        """Render a bounded plain-text digest suitable for a thin UI adapter.

        The result intentionally remains generic text instead of a Discord
        embed.  ``max_characters`` defaults below Discord's message limit but
        callers can choose another transport's budget.  Metadata is compacted
        rather than allowing provider-originated text to overflow a message.
        """

        if max_characters < 256:
            raise ValueError("max_characters must be at least 256.")

        header = [
            "ResearchRadar Daily Digest",
            (
                "Window: "
                f"{_format_timestamp(self.period_start)} to {_format_timestamp(self.period_end)}"
            ),
            f"New papers discovered: {self.paper_count}",
        ]
        sections: list[str] = ["\n".join(header)]

        if not self.top_papers:
            sections.append("No new persisted papers were found in this window.")
        else:
            paper_lines = ["Top papers:"]
            for index, paper in enumerate(self.top_papers, start=1):
                paper_lines.extend(_render_paper_lines(index, paper))
            sections.append("\n".join(paper_lines))

        if self.watch_activity:
            topic_lines = ["Watch topics with activity:"]
            topic_lines.extend(
                f"- {_truncate(activity.topic_name, 120)}: "
                f"{activity.paper_count} new paper(s)"
                for activity in self.watch_activity
            )
            sections.append("\n".join(topic_lines))

        return _render_with_budget(sections, max_characters)


class DigestNotificationSink(Protocol):
    """Neutral async delivery boundary for a completed scheduled digest."""

    async def notify_digest(self, digest: ResearchDigest) -> None:
        """Deliver one digest after it has been atomically claimed."""


@dataclass(frozen=True, slots=True)
class ScheduledDigestResult:
    """The safe outcome of one scheduled digest attempt."""

    status: str
    digest: ResearchDigest | None = None
    run_id: str | None = None
    error: str | None = None
    skipped_due_to_overlap: bool = False
    skipped_due_to_claim: bool = False

    @property
    def sent(self) -> bool:
        """Whether notification completion was durably recorded."""

        return self.status == "sent"


class DigestService:
    """Build persisted-data digests and safely deliver scheduled ones.

    On-demand calls are intentionally read-only with respect to digest-run
    state. Scheduled calls derive a period from the last sent cursor, claim an
    exact period, and only mark it sent after the notification sink succeeds.
    Failed delivery leaves the period retryable through the repository's
    existing claim API.
    """

    def __init__(
        self,
        repository: ResearchRepository,
        *,
        notification_sink: DigestNotificationSink | None = None,
        default_window: timedelta = DEFAULT_DIGEST_WINDOW,
        top_paper_limit: int = DEFAULT_TOP_PAPER_LIMIT,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        _validate_window(default_window, label="default_window")
        _validate_top_paper_limit(top_paper_limit)

        self._repository = repository
        self._notification_sink = notification_sink
        self._default_window = default_window
        self._top_paper_limit = top_paper_limit
        self._clock = clock or _utc_now
        self._scheduled_lock = asyncio.Lock()
        self._retry_period: tuple[datetime, datetime] | None = None

    async def build_on_demand(
        self,
        *,
        period_end: datetime | None = None,
        window: timedelta | None = None,
    ) -> ResearchDigest:
        """Build a recent persisted-memory digest without changing run state.

        This is the intended ``/digest`` entry point.  It never reads or writes
        a scheduled-digest cursor, so manually asking for a digest cannot make
        the next scheduled digest omit papers.
        """

        selected_window = self._default_window if window is None else window
        _validate_window(selected_window, label="window")
        end = _as_utc_naive(period_end or self._clock())
        return await self.build_digest(end - selected_window, end)

    async def build_digest(self, period_start: datetime, period_end: datetime) -> ResearchDigest:
        """Build typed digest data from an explicit persisted-data window."""

        start, end = _validate_period(period_start, period_end)
        candidates = await asyncio.to_thread(
            self._repository.list_digest_candidates,
            start,
            end,
        )
        return _build_digest(
            start,
            end,
            candidates,
            top_paper_limit=self._top_paper_limit,
        )

    async def run_scheduled_digest(
        self,
        *,
        period_end: datetime | None = None,
    ) -> ScheduledDigestResult:
        """Claim, build, send, and finalize exactly one cursor-based digest.

        The process-local lock avoids overlapping invocations with slightly
        different clock values.  The durable repository claim independently
        protects restarts and future multi-process deployment from duplicate
        sends for the same exact period.
        """

        if self._scheduled_lock.locked():
            logger.warning("Skipped digest because another scheduled run is active.")
            return ScheduledDigestResult(
                status="skipped",
                skipped_due_to_overlap=True,
            )

        if self._notification_sink is None:
            logger.warning("Skipped scheduled digest because no notification sink is configured.")
            return ScheduledDigestResult(status="not_configured")

        async with self._scheduled_lock:
            retry_period = self._retry_period
            if retry_period is not None:
                start, end = retry_period
                logger.info("Retrying failed scheduled digest window.")
            else:
                end = _as_utc_naive(period_end or self._clock())
                try:
                    last_successful_end = await asyncio.to_thread(
                        self._repository.get_last_successful_digest_end
                    )
                except Exception as error:
                    logger.exception("Unable to read the scheduled digest cursor.")
                    return ScheduledDigestResult(status="failed", error=_safe_error(error))

                start = (
                    _as_utc_naive(last_successful_end)
                    if last_successful_end is not None
                    else end - self._default_window
                )
            if end <= start:
                logger.info("Skipped digest with non-advancing cursor window.")
                return ScheduledDigestResult(status="skipped", skipped_due_to_claim=True)

            try:
                run = await asyncio.to_thread(self._repository.claim_digest_run, start, end)
            except Exception as error:
                logger.exception("Unable to claim scheduled digest period.")
                return ScheduledDigestResult(status="failed", error=_safe_error(error))

            if run is None:
                logger.info("Skipped already claimed or sent digest period.")
                return ScheduledDigestResult(status="skipped", skipped_due_to_claim=True)

            try:
                digest = await self.build_digest(start, end)
            except Exception as error:
                safe_error = _safe_error(error)
                logger.exception("Unable to build scheduled digest %s.", run.id)
                await self._record_scheduled_failure(run.id, safe_error)
                self._retry_period = (start, end)
                return ScheduledDigestResult(status="failed", run_id=run.id, error=safe_error)

            try:
                await self._notification_sink.notify_digest(digest)
            except Exception as error:
                safe_error = _safe_error(error)
                logger.exception("Scheduled digest notification failed for %s.", run.id)
                await self._record_scheduled_failure(run.id, safe_error)
                self._retry_period = (start, end)
                return ScheduledDigestResult(
                    status="failed",
                    digest=digest,
                    run_id=run.id,
                    error=safe_error,
                )

            try:
                marked_sent = await asyncio.to_thread(
                    self._repository.mark_digest_sent,
                    run.id,
                    digest.paper_count,
                )
            except Exception as error:
                # A delivery that completed but cannot be marked sent is
                # intentionally visible as failure. This favors a possible
                # duplicate over silently advancing the durable cursor.
                safe_error = _safe_error(error)
                logger.exception("Unable to finalize scheduled digest %s.", run.id)
                await self._record_scheduled_failure(run.id, safe_error)
                self._retry_period = (start, end)
                return ScheduledDigestResult(
                    status="failed",
                    digest=digest,
                    run_id=run.id,
                    error=safe_error,
                )

            if not marked_sent:
                safe_error = "Digest run could not be finalized."
                logger.error("Scheduled digest run %s disappeared before finalization.", run.id)
                self._retry_period = (start, end)
                await self._record_scheduled_failure(run.id, safe_error)
                return ScheduledDigestResult(
                    status="failed",
                    digest=digest,
                    run_id=run.id,
                    error=safe_error,
                )

            logger.info(
                "Sent scheduled digest %s containing %d persisted paper(s).",
                run.id,
                digest.paper_count,
            )
            self._retry_period = None
            return ScheduledDigestResult(status="sent", digest=digest, run_id=run.id)

    async def _record_scheduled_failure(self, run_id: str, safe_error: str) -> None:
        """Best-effort failure state: logs retain any storage-level detail."""

        try:
            await asyncio.to_thread(self._repository.mark_digest_failed, run_id, safe_error)
        except Exception:
            logger.exception("Unable to record scheduled digest failure for %s.", run_id)


def _build_digest(
    period_start: datetime,
    period_end: datetime,
    candidates: list[DigestCandidate],
    *,
    top_paper_limit: int,
) -> ResearchDigest:
    """Convert repository candidates into bounded, UI-independent digest data."""

    top_papers = tuple(
        _to_digest_paper(candidate) for candidate in candidates[:top_paper_limit]
    )
    activity_counts: Counter[str] = Counter()
    for candidate in candidates:
        activity_counts.update(candidate.watch_topic_names)
    activity = tuple(
        WatchActivity(topic_name=name, paper_count=count)
        for name, count in sorted(
            activity_counts.items(),
            key=lambda item: (-item[1], item[0].casefold()),
        )
    )
    return ResearchDigest(
        period_start=period_start,
        period_end=period_end,
        paper_count=len(candidates),
        top_papers=top_papers,
        watch_activity=activity,
    )


def _to_digest_paper(candidate: DigestCandidate) -> DigestPaper:
    paper = candidate.paper
    canonical_link = f"https://doi.org/{paper.doi}" if paper.doi else paper.url
    return DigestPaper(
        paper_id=paper.id,
        title=paper.title,
        authors=tuple(paper.authors),
        publication_year=paper.publication_year,
        venue=paper.venue,
        citation_count=paper.citation_count,
        canonical_link=canonical_link,
        rank_score=candidate.highest_rank_score,
        watch_topic_names=candidate.watch_topic_names,
        insight=_paper_card_insight(candidate),
    )


def _paper_card_insight(candidate: DigestCandidate) -> PaperCardInsight | None:
    """Select one stored, evidence-aware field without fabricating a summary."""

    card = candidate.paper_card
    if card is None:
        return None
    if card.contributions:
        return PaperCardInsight(kind="Contribution", text=_compact_text(card.contributions[0]))
    if card.main_claims:
        claim = card.main_claims[0]
        return PaperCardInsight(
            kind="Stored claim",
            text=_compact_text(claim.claim),
            source_section=claim.source_section,
        )
    if card.methods:
        return PaperCardInsight(kind="Method", text=_compact_text(card.methods[0]))
    if card.limitations:
        return PaperCardInsight(kind="Limitation", text=_compact_text(card.limitations[0]))
    return None


def _render_paper_lines(index: int, paper: DigestPaper) -> list[str]:
    metadata: list[str] = []
    if paper.publication_year is not None:
        metadata.append(str(paper.publication_year))
    if paper.venue:
        metadata.append(_truncate(_compact_text(paper.venue), 80))
    if paper.citation_count is not None:
        metadata.append(f"{paper.citation_count} citations")

    lines = [f"{index}. {_truncate(_compact_text(paper.title), 180)}"]
    if metadata:
        lines.append(f"   {'; '.join(metadata)}")
    if paper.authors:
        lines.append(f"   Authors: {_truncate(', '.join(map(_compact_text, paper.authors)), 160)}")
    if paper.watch_topic_names:
        lines.append(
            "   Watch: "
            f"{_truncate(', '.join(map(_compact_text, paper.watch_topic_names)), 160)}"
        )
    if paper.insight is not None:
        suffix = (
            f" ({_compact_text(paper.insight.source_section)})"
            if paper.insight.source_section
            else ""
        )
        lines.append(
            f"   {paper.insight.kind}{suffix}: {_truncate(paper.insight.text, 220)}"
        )
    if paper.canonical_link:
        lines.append(f"   {_truncate(paper.canonical_link, 220)}")
    return lines


def _render_with_budget(sections: list[str], max_characters: int) -> str:
    """Keep complete early sections where possible and mark compact truncation."""

    rendered = "\n\n".join(sections)
    if len(rendered) <= max_characters:
        return rendered
    marker = "\n\n[Digest truncated; use the on-demand command for full details.]"
    available = max_characters - len(marker)
    if available <= 0:
        return _truncate(rendered, max_characters)
    return f"{rendered[:available].rstrip()}{marker}"


def _validate_period(period_start: datetime, period_end: datetime) -> tuple[datetime, datetime]:
    start = _as_utc_naive(period_start)
    end = _as_utc_naive(period_end)
    if end <= start:
        raise ValueError("period_end must be after period_start.")
    return start, end


def _validate_window(window: timedelta, *, label: str) -> None:
    if window <= timedelta(0):
        raise ValueError(f"{label} must be positive.")


def _validate_top_paper_limit(limit: int) -> None:
    if not 1 <= limit <= MAX_TOP_PAPER_LIMIT:
        raise ValueError(f"top_paper_limit must be between 1 and {MAX_TOP_PAPER_LIMIT}.")


def _utc_now() -> datetime:
    """Return a SQLite-compatible naive UTC timestamp."""

    return datetime.now(UTC).replace(tzinfo=None)


def _as_utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _format_timestamp(value: datetime) -> str:
    return f"{_as_utc_naive(value).isoformat(timespec='minutes')} UTC"


def _compact_text(value: str) -> str:
    return " ".join(value.split())


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: max(1, limit - 1)].rstrip()}…"


def _safe_error(error: Exception) -> str:
    """Expose only an exception category; detailed errors remain in logs."""

    return f"Digest operation failed ({type(error).__name__})."
