"""Focused tests for Discord-independent watchlist monitoring."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from research_radar.models import Paper
from research_radar.research.ranker import RankedPaper
from research_radar.research.service import SearchResult
from research_radar.storage import (
    Database,
    PendingNotification,
    ResearchRepository,
    WatchTopic,
)
from research_radar.watch.service import NotificationSink, WatchService


def _paper(identifier: str, title: str) -> Paper:
    return Paper(
        id=f"openalex:{identifier}",
        title=title,
        abstract=f"{title} abstract",
        authors=["Ada Lovelace"],
        publication_year=2026,
        venue="Research Venue",
        url=f"https://example.test/{identifier}",
        citation_count=3,
        source="openalex",
        external_ids={"openalex": identifier},
    )


def _ranked(identifier: str, title: str, score: float) -> RankedPaper:
    return RankedPaper(
        paper=_paper(identifier, title),
        score=score,
        title_overlap=score,
        abstract_overlap=0.0,
        recency=1.0,
        citations=0.0,
        completeness=1.0,
    )


class _ResearchService:
    def __init__(self, outcomes: dict[str, SearchResult | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, limit: int) -> SearchResult:
        self.calls.append((query, limit))
        outcome = self.outcomes[query]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _RecordingSink(NotificationSink):
    def __init__(self) -> None:
        self.deliveries: list[tuple[WatchTopic, tuple[str, ...]]] = []

    async def notify(self, topic: WatchTopic, papers: Sequence[PendingNotification]) -> None:
        self.deliveries.append((topic, tuple(item.paper.title for item in papers)))


class _FailingSink(NotificationSink):
    async def notify(self, topic: WatchTopic, papers: Sequence[PendingNotification]) -> None:
        raise RuntimeError("delivery unavailable")


@pytest.fixture
def repository(tmp_path: Path) -> ResearchRepository:
    database = Database.create(f"sqlite:///{tmp_path / 'research_radar.db'}")
    database.initialize_schema()
    try:
        yield ResearchRepository(database)
    finally:
        database.dispose()


@pytest.mark.asyncio
async def test_scan_persists_all_results_and_only_notifies_bounded_ranked_papers(
    repository: ResearchRepository,
) -> None:
    sink = _RecordingSink()
    research = _ResearchService(
        {
            "medical reconstruction": SearchResult(
                papers=[
                    _ranked("high", "High relevance reconstruction", 0.92),
                    _ranked("low", "Low relevance reconstruction", 0.21),
                ],
                warnings=["arxiv was unavailable; results may be partial."],
            )
        }
    )
    service = WatchService(
        repository,
        research,  # type: ignore[arg-type]
        notification_sink=sink,
        notification_cap=1,
        minimum_notification_score=0.5,
    )
    topic = await service.add_topic("Medical reconstruction", "medical reconstruction")

    result = await service.scan_enabled_topics()

    assert result.skipped_due_to_overlap is False
    assert len(result.topics) == 1
    topic_result = result.topics[0]
    assert topic_result.papers_retrieved == 2
    assert topic_result.papers_persisted == 2
    assert topic_result.new_discoveries == 2
    assert topic_result.notifications_delivered == 1
    assert topic_result.warnings == ("arxiv was unavailable; results may be partial.",)
    assert sink.deliveries == [(topic, ("High relevance reconstruction",))]
    assert research.calls == [("medical reconstruction", 10)]

    pending = repository.list_pending_notifications(topic.id, cap=10)
    stored_topic = repository.get_watch_topic(topic.id)
    assert [item.paper.title for item in pending] == ["Low relevance reconstruction"]
    assert stored_topic is not None
    assert stored_topic.last_scan_at is not None
    assert stored_topic.last_error is None


@pytest.mark.asyncio
async def test_failed_notification_is_left_pending_without_failing_the_discovery_scan(
    repository: ResearchRepository,
) -> None:
    research = _ResearchService(
        {
            "visual anomaly": SearchResult(
                papers=[_ranked("notification", "Visual anomaly detection", 0.95)]
            )
        }
    )
    service = WatchService(
        repository,
        research,  # type: ignore[arg-type]
        notification_sink=_FailingSink(),
    )
    topic = await service.add_topic("Visual anomaly", "visual anomaly")

    result = await service.scan_enabled_topics()

    topic_result = result.topics[0]
    assert topic_result.error is None
    assert topic_result.notifications_delivered == 0
    assert topic_result.notification_error == "Notification delivery failed (RuntimeError)."
    assert [item.paper.title for item in repository.list_pending_notifications(topic.id)] == [
        "Visual anomaly detection"
    ]
    stored_topic = repository.get_watch_topic(topic.id)
    assert stored_topic is not None
    assert stored_topic.last_scan_at is not None
    assert stored_topic.last_error is None


@pytest.mark.asyncio
async def test_topic_failures_are_recorded_and_do_not_stop_later_topics(
    repository: ResearchRepository,
) -> None:
    research = _ResearchService(
        {
            "broken query": RuntimeError("provider outage"),
            "working query": SearchResult(papers=[_ranked("working", "Working paper", 0.8)]),
        }
    )
    service = WatchService(repository, research)  # type: ignore[arg-type]
    broken = await service.add_topic("Broken", "broken query")
    working = await service.add_topic("Working", "working query")

    result = await service.scan_enabled_topics()

    by_topic = {scan.topic.id: scan for scan in result.topics}
    assert by_topic[broken.id].error == "Research scan failed (RuntimeError)."
    assert by_topic[working.id].error is None
    assert by_topic[working.id].new_discoveries == 1
    failed_topic = repository.get_watch_topic(broken.id)
    successful_topic = repository.get_watch_topic(working.id)
    assert failed_topic is not None
    assert failed_topic.last_scan_at is None
    assert failed_topic.last_error == "Research scan failed (RuntimeError)."
    assert successful_topic is not None
    assert successful_topic.last_scan_at is not None


class _ThreadRecordingRepository:
    def __init__(self) -> None:
        created_at = datetime.now(UTC).replace(tzinfo=None)
        self.topics = [
            WatchTopic("one", "One", "one", True, created_at, None, None),
            WatchTopic("two", "Two", "two", True, created_at, None, None),
        ]
        self.thread_ids: list[int] = []
        self._paper_number = 0

    def _record(self) -> None:
        self.thread_ids.append(threading.get_ident())

    def list_enabled_watch_topics(self) -> list[WatchTopic]:
        self._record()
        return self.topics

    def upsert_merged_paper(self, paper: Paper) -> str:
        self._record()
        self._paper_number += 1
        return f"paper-{self._paper_number}"

    def record_watch_discovery(self, topic_id: str, paper_id: str, rank_score: float) -> bool:
        self._record()
        return True

    def mark_watch_scan_success(self, topic_id: str) -> bool:
        self._record()
        return True


class _BlockingResearchService:
    def __init__(self) -> None:
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.calls: list[str] = []
        self.active = 0
        self.max_active = 0

    async def search(self, query: str, limit: int) -> SearchResult:
        self.calls.append(query)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if query == "one":
                self.first_started.set()
                await self.release_first.wait()
            return SearchResult(papers=[_ranked(query, f"Paper {query}", 0.8)])
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_scans_topics_sequentially_skips_overlap_and_moves_sync_work_off_loop() -> None:
    repository = _ThreadRecordingRepository()
    research = _BlockingResearchService()
    service = WatchService(repository, research)  # type: ignore[arg-type]
    main_thread = threading.get_ident()

    running_scan = asyncio.create_task(service.scan_enabled_topics())
    await research.first_started.wait()
    overlap = await service.scan_enabled_topics()

    assert overlap.skipped_due_to_overlap is True
    assert research.calls == ["one"]
    assert research.max_active == 1

    research.release_first.set()
    completed = await running_scan

    assert [item.topic.id for item in completed.topics] == ["one", "two"]
    assert research.calls == ["one", "two"]
    assert research.max_active == 1
    assert repository.thread_ids
    assert all(thread_id != main_thread for thread_id in repository.thread_ids)


@pytest.mark.asyncio
async def test_watchlist_crud_wrappers_use_repository_state(repository: ResearchRepository) -> None:
    service = WatchService(repository, _ResearchService({}))  # type: ignore[arg-type]

    topic = await service.add_topic("State space", "spectral state space models")
    paused = await service.set_topic_enabled(topic.id, False)

    assert [item.id for item in await service.list_topics()] == [topic.id]
    assert paused is not None
    assert paused.enabled is False
    assert await service.remove_topic("state space") is True
    assert await service.list_topics() == []
