"""Discord-independent watchlist orchestration and paper monitoring."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Protocol

from research_radar.research.service import ResearchService
from research_radar.storage import PendingNotification, ResearchRepository, WatchTopic

logger = logging.getLogger(__name__)


class NotificationSink(Protocol):
    """Deliver a bounded watch update through an application boundary.

    The watch domain deliberately knows nothing about Discord, channels, or
    embeds. A Discord adapter can implement this protocol, as can a test sink
    or a future non-Discord delivery mechanism.
    """

    async def notify(
        self,
        topic: WatchTopic,
        papers: Sequence[PendingNotification],
    ) -> None:
        """Deliver one topic's pending paper notification."""


@dataclass(frozen=True, slots=True)
class TopicScanResult:
    """The outcome of one watch topic scan, including safe partial failures."""

    topic: WatchTopic
    papers_retrieved: int = 0
    papers_persisted: int = 0
    new_discoveries: int = 0
    notifications_delivered: int = 0
    warnings: tuple[str, ...] = ()
    error: str | None = None
    notification_error: str | None = None


@dataclass(frozen=True, slots=True)
class WatchScanResult:
    """A non-overlapping scan run across all currently enabled topics."""

    topics: tuple[TopicScanResult, ...] = ()
    skipped_due_to_overlap: bool = False

    @property
    def failed_topics(self) -> tuple[TopicScanResult, ...]:
        """Return only topic scans whose discovery/persistence stage failed."""

        return tuple(result for result in self.topics if result.error is not None)


class WatchService:
    """Manage saved watch topics and run their bounded discovery scans.

    Repository operations intentionally remain synchronous and are dispatched
    through :func:`asyncio.to_thread`. Topics are scanned sequentially so a
    single-user process cannot create an uncontrolled provider burst; each
    individual ``ResearchService.search`` call still fans providers out
    concurrently.
    """

    def __init__(
        self,
        repository: ResearchRepository,
        research_service: ResearchService,
        *,
        notification_sink: NotificationSink | None = None,
        search_limit: int = 10,
        notification_cap: int = 3,
        minimum_notification_score: float | None = 0.35,
    ) -> None:
        if search_limit < 1:
            raise ValueError("search_limit must be at least 1.")
        if notification_cap < 1:
            raise ValueError("notification_cap must be at least 1.")
        if minimum_notification_score is not None and (
            not math.isfinite(minimum_notification_score)
            or not 0 <= minimum_notification_score <= 1
        ):
            raise ValueError("minimum_notification_score must be between 0 and 1.")

        self._repository = repository
        self._research_service = research_service
        self._notification_sink = notification_sink
        self._search_limit = search_limit
        self._notification_cap = notification_cap
        self._minimum_notification_score = minimum_notification_score
        self._scan_lock = asyncio.Lock()

    async def add_topic(self, name: str, query: str) -> WatchTopic:
        """Persist one single-user watch topic without blocking the event loop."""

        return await asyncio.to_thread(self._repository.add_watch_topic, name, query)

    async def list_topics(self) -> list[WatchTopic]:
        """List saved watch topics in their stable repository order."""

        return await asyncio.to_thread(self._repository.list_watch_topics)

    async def remove_topic(self, topic_id_or_name: str) -> bool:
        """Remove a topic by its stable id or unique normalized name."""

        return await asyncio.to_thread(self._repository.remove_watch_topic, topic_id_or_name)

    async def set_topic_enabled(self, topic_id: str, enabled: bool) -> WatchTopic | None:
        """Enable or pause a topic while retaining its discovery history."""

        return await asyncio.to_thread(
            self._repository.set_watch_topic_enabled,
            topic_id,
            enabled,
        )

    async def scan_enabled_topics(self) -> WatchScanResult:
        """Scan enabled topics once, refusing an overlapping process-local run.

        A topic failure is recorded and returned without preventing later
        topics from scanning. An unexpected failure while listing topics is
        logged and re-raised so APScheduler's error listener can report it.
        """

        if self._scan_lock.locked():
            logger.warning("Skipped watch scan because another scan is already running.")
            return WatchScanResult(skipped_due_to_overlap=True)

        async with self._scan_lock:
            try:
                topics = await asyncio.to_thread(self._repository.list_enabled_watch_topics)
            except Exception:
                logger.exception("Unable to list enabled watch topics.")
                raise

            results: list[TopicScanResult] = []
            for topic in topics:
                results.append(await self._scan_topic(topic))

            logger.info(
                "Completed watch scan for %d topic(s); %d topic scan(s) failed.",
                len(results),
                sum(result.error is not None for result in results),
            )
            return WatchScanResult(topics=tuple(results))

    async def _scan_topic(self, topic: WatchTopic) -> TopicScanResult:
        """Search, persist, and optionally notify for a single topic."""

        try:
            discovery = await self._research_service.search(topic.query, limit=self._search_limit)
            papers_persisted = 0
            new_discoveries = 0
            for ranked_paper in discovery.papers:
                paper_id = await asyncio.to_thread(
                    self._repository.upsert_merged_paper,
                    ranked_paper.paper,
                )
                is_new = await asyncio.to_thread(
                    self._repository.record_watch_discovery,
                    topic.id,
                    paper_id,
                    ranked_paper.score,
                )
                papers_persisted += 1
                new_discoveries += int(is_new)

            await asyncio.to_thread(self._repository.mark_watch_scan_success, topic.id)
        except Exception as error:
            safe_error = _safe_scan_error(error)
            logger.exception("Watch scan failed for topic %s (%s).", topic.name, topic.id)
            try:
                await asyncio.to_thread(
                    self._repository.mark_watch_scan_failure,
                    topic.id,
                    safe_error,
                )
            except Exception:
                logger.exception(
                    "Unable to record watch scan failure for topic %s (%s).",
                    topic.name,
                    topic.id,
                )
            return TopicScanResult(topic=topic, error=safe_error)

        delivered, notification_error = await self._notify_pending(topic)
        return TopicScanResult(
            topic=topic,
            papers_retrieved=len(discovery.papers),
            papers_persisted=papers_persisted,
            new_discoveries=new_discoveries,
            notifications_delivered=delivered,
            warnings=tuple(discovery.warnings),
            notification_error=notification_error,
        )

    async def _notify_pending(self, topic: WatchTopic) -> tuple[int, str | None]:
        """Deliver one bounded pending batch and mark only confirmed sends."""

        if self._notification_sink is None:
            return 0, None

        try:
            pending = await asyncio.to_thread(
                partial(
                    self._repository.list_pending_notifications,
                    topic.id,
                    cap=self._notification_cap,
                    minimum_rank_score=self._minimum_notification_score,
                )
            )
            if not pending:
                return 0, None

            await self._notification_sink.notify(topic, tuple(pending))
            return (
                await asyncio.to_thread(
                    self._repository.mark_notified,
                    topic.id,
                    [item.paper.id for item in pending],
                ),
                None,
            )
        except Exception as error:
            # The repository remains untouched after a delivery exception, so
            # the same pending papers can be retried by a later successful run.
            safe_error = _safe_notification_error(error)
            logger.exception(
                "Watch notification failed for topic %s (%s).",
                topic.name,
                topic.id,
            )
            return 0, safe_error


def _safe_scan_error(error: Exception) -> str:
    """Return a non-sensitive, bounded status string while logs retain detail."""

    return f"Research scan failed ({type(error).__name__})."


def _safe_notification_error(error: Exception) -> str:
    """Return a concise notification status without exposing transport details."""

    return f"Notification delivery failed ({type(error).__name__})."
