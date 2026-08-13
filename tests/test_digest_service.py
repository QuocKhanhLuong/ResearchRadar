"""Focused tests for persisted-memory digest assembly and delivery behavior."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from research_radar.digest import DigestService, ResearchDigest
from research_radar.models import Paper, PaperCard
from research_radar.storage import Database, ResearchRepository


def _paper(identifier: str, title: str) -> Paper:
    return Paper(
        id=f"openalex:{identifier}",
        title=title,
        abstract=f"{title} abstract",
        authors=["Ada Lovelace", "Grace Hopper"],
        publication_year=2026,
        venue="Research Venue",
        url=f"https://example.test/{identifier}",
        citation_count=7,
        source="openalex",
        external_ids={"openalex": identifier},
    )


@pytest.fixture
def repository(tmp_path: Path) -> ResearchRepository:
    database = Database.create(f"sqlite:///{tmp_path / 'research_radar.db'}")
    database.initialize_schema()
    try:
        yield ResearchRepository(database)
    finally:
        database.dispose()


class _RecordingSink:
    def __init__(self) -> None:
        self.deliveries: list[ResearchDigest] = []

    async def notify_digest(self, digest: ResearchDigest) -> None:
        self.deliveries.append(digest)


class _FailOnceSink:
    def __init__(self) -> None:
        self.attempts = 0
        self.deliveries: list[ResearchDigest] = []

    async def notify_digest(self, digest: ResearchDigest) -> None:
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("private transport details must not escape")
        self.deliveries.append(digest)


class _BlockingSink:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def notify_digest(self, digest: ResearchDigest) -> None:
        del digest
        self.entered.set()
        await self.release.wait()


def _window() -> tuple[datetime, datetime]:
    end = datetime(2030, 1, 2, 8, 0, tzinfo=UTC)
    return end - timedelta(days=1), end


def _record_discovery(
    repository: ResearchRepository,
    *,
    topic_name: str,
    query: str,
    identifier: str,
    title: str,
    score: float,
    seen_at: datetime,
) -> str:
    topic = repository.add_watch_topic(topic_name, query)
    paper_id = repository.upsert_merged_paper(_paper(identifier, title))
    repository.record_watch_discovery(topic.id, paper_id, score, seen_at=seen_at)
    return paper_id


@pytest.mark.asyncio
async def test_on_demand_digest_uses_only_persisted_candidates_and_has_no_run_state(
    repository: ResearchRepository,
) -> None:
    start, end = _window()
    high_id = _record_discovery(
        repository,
        topic_name="Medical reconstruction",
        query="medical reconstruction",
        identifier="high",
        title="High-ranked Reconstruction",
        score=0.93,
        seen_at=start + timedelta(hours=2),
    )
    low_id = _record_discovery(
        repository,
        topic_name="Visual anomaly",
        query="visual anomaly detection",
        identifier="low",
        title="Lower-ranked Anomaly Detection",
        score=0.51,
        seen_at=start + timedelta(hours=3),
    )
    second_topic = repository.add_watch_topic("Robustness", "robust imaging")
    repository.record_watch_discovery(
        second_topic.id,
        high_id,
        0.77,
        seen_at=start + timedelta(hours=4),
    )
    repository.upsert_paper_card(
        PaperCard(
            paper_id=high_id,
            contributions=["A compact deterministic reconstruction method."],
        )
    )

    service = DigestService(repository, top_paper_limit=2)
    digest = await service.build_on_demand(period_end=end, window=timedelta(days=1))

    assert digest.paper_count == 2
    assert [paper.paper_id for paper in digest.top_papers] == [high_id, low_id]
    assert digest.top_papers[0].watch_topic_names == (
        "Medical reconstruction",
        "Robustness",
    )
    assert digest.top_papers[0].insight is not None
    assert digest.top_papers[0].insight.kind == "Contribution"
    assert [
        (activity.topic_name, activity.paper_count) for activity in digest.watch_activity
    ] == [
        ("Medical reconstruction", 1),
        ("Robustness", 1),
        ("Visual anomaly", 1),
    ]
    assert repository.get_last_successful_digest_end() is None
    assert (
        repository.claim_digest_run(start.replace(tzinfo=None), end.replace(tzinfo=None))
        is not None
    )

    text = digest.render_text()
    assert "New papers discovered: 2" in text
    assert "High-ranked Reconstruction" in text
    assert "Contribution: A compact deterministic reconstruction method." in text
    assert "Watch topics with activity:" in text


@pytest.mark.asyncio
async def test_explicit_digest_window_normalizes_aware_times_and_rejects_invalid_boundaries(
    repository: ResearchRepository,
) -> None:
    service = DigestService(repository)
    start, end = _window()

    digest = await service.build_digest(start, end)

    assert digest.period_start.tzinfo is None
    assert digest.period_end.tzinfo is None
    with pytest.raises(ValueError, match="period_end"):
        await service.build_digest(end, start)
    with pytest.raises(ValueError, match="window"):
        await service.build_on_demand(period_end=end, window=timedelta())
    with pytest.raises(ValueError, match="top_paper_limit"):
        DigestService(repository, top_paper_limit=0)


@pytest.mark.asyncio
async def test_scheduled_digest_marks_cursor_sent_only_after_neutral_sink_succeeds(
    repository: ResearchRepository,
) -> None:
    start, end = _window()
    _record_discovery(
        repository,
        topic_name="Imaging",
        query="medical imaging",
        identifier="scheduled",
        title="Scheduled Paper",
        score=0.8,
        seen_at=start + timedelta(hours=1),
    )
    sink = _RecordingSink()
    service = DigestService(repository, notification_sink=sink)

    result = await service.run_scheduled_digest(period_end=end)

    assert result.sent is True
    assert result.status == "sent"
    assert result.digest is not None
    assert result.digest.paper_count == 1
    assert len(sink.deliveries) == 1
    assert repository.get_last_successful_digest_end() == end.replace(tzinfo=None)


@pytest.mark.asyncio
async def test_failed_scheduled_delivery_is_retryable_for_the_same_claimed_window(
    repository: ResearchRepository,
) -> None:
    start, end = _window()
    _record_discovery(
        repository,
        topic_name="Anomaly",
        query="anomaly detection",
        identifier="retry",
        title="Retryable Paper",
        score=0.9,
        seen_at=start + timedelta(hours=1),
    )
    sink = _FailOnceSink()
    service = DigestService(repository, notification_sink=sink)

    failed = await service.run_scheduled_digest(period_end=end)

    assert failed.status == "failed"
    assert failed.error == "Digest operation failed (RuntimeError)."
    assert repository.get_last_successful_digest_end() is None

    retried = await service.run_scheduled_digest(period_end=end + timedelta(hours=12))

    assert retried.sent is True
    assert len(sink.deliveries) == 1
    assert retried.digest is not None
    assert retried.digest.period_end == end.replace(tzinfo=None)
    assert repository.get_last_successful_digest_end() == end.replace(tzinfo=None)


@pytest.mark.asyncio
async def test_scheduled_digest_without_a_sink_does_not_claim_or_advance_state(
    repository: ResearchRepository,
) -> None:
    start, end = _window()
    service = DigestService(repository)

    result = await service.run_scheduled_digest(period_end=end)

    assert result.status == "not_configured"
    assert repository.get_last_successful_digest_end() is None
    assert (
        repository.claim_digest_run(start.replace(tzinfo=None), end.replace(tzinfo=None))
        is not None
    )


@pytest.mark.asyncio
async def test_scheduled_digest_refuses_a_process_local_overlap(
    repository: ResearchRepository,
) -> None:
    start, end = _window()
    _record_discovery(
        repository,
        topic_name="Robustness",
        query="robustness",
        identifier="overlap",
        title="Overlap Paper",
        score=0.85,
        seen_at=start + timedelta(hours=1),
    )
    sink = _BlockingSink()
    service = DigestService(repository, notification_sink=sink)

    first = asyncio.create_task(service.run_scheduled_digest(period_end=end))
    await sink.entered.wait()
    overlapping = await service.run_scheduled_digest(period_end=end)
    sink.release.set()
    completed = await first

    assert overlapping.status == "skipped"
    assert overlapping.skipped_due_to_overlap is True
    assert completed.sent is True


@pytest.mark.asyncio
async def test_rendered_digest_is_bounded_without_losing_its_header(
    repository: ResearchRepository,
) -> None:
    start, end = _window()
    _record_discovery(
        repository,
        topic_name="Very long topic name " * 20,
        query="long query",
        identifier="long",
        title="Very long paper title " * 40,
        score=0.91,
        seen_at=start + timedelta(hours=1),
    )
    digest = await DigestService(repository).build_digest(start, end)

    text = digest.render_text(max_characters=500)

    assert len(text) <= 500
    assert text.startswith("ResearchRadar Daily Digest")
    assert "[Digest truncated;" in text
