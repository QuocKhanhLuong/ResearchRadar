"""Focused tests for APScheduler registration and lifecycle behavior."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from research_radar.watch.scheduler import WATCH_SCAN_JOB_ID, WatchScheduler
from research_radar.watch.service import WatchScanResult


class _RecordingScheduler:
    def __init__(self) -> None:
        self.running = False
        self.listeners: list[tuple[object, int]] = []
        self.jobs: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.start_calls = 0
        self.shutdown_calls: list[bool] = []

    def add_listener(self, callback: object, mask: int) -> None:
        self.listeners.append((callback, mask))

    def add_job(self, *args: object, **kwargs: object) -> None:
        self.jobs.append((args, kwargs))

    def start(self) -> None:
        self.running = True
        self.start_calls += 1

    def shutdown(self, wait: bool = True) -> None:
        self.running = False
        self.shutdown_calls.append(wait)


@pytest.mark.asyncio
async def test_scheduler_registers_bounded_coalesced_job_and_has_clean_lifecycle() -> None:
    service = SimpleNamespace(scan_enabled_topics=AsyncMock(return_value=WatchScanResult()))
    scheduler = _RecordingScheduler()
    watch_scheduler = WatchScheduler(
        service,  # type: ignore[arg-type]
        scan_hours=6,
        scheduler=scheduler,  # type: ignore[arg-type]
    )

    watch_scheduler.register()
    watch_scheduler.register()
    watch_scheduler.start()
    watch_scheduler.start()
    run_result = await watch_scheduler._run_scan()
    await watch_scheduler.shutdown()

    assert run_result == WatchScanResult()
    assert len(scheduler.listeners) == 1
    assert len(scheduler.jobs) == 1
    args, kwargs = scheduler.jobs[0]
    assert callable(args[0])
    assert kwargs["trigger"] == "interval"
    assert kwargs["hours"] == 6
    assert kwargs["id"] == WATCH_SCAN_JOB_ID
    assert kwargs["max_instances"] == 1
    assert kwargs["coalesce"] is True
    assert scheduler.start_calls == 1
    assert scheduler.shutdown_calls == [False]
    service.scan_enabled_topics.assert_awaited_once_with()


def test_scheduler_error_listener_logs_only_its_own_job(caplog: pytest.LogCaptureFixture) -> None:
    service = SimpleNamespace(scan_enabled_topics=AsyncMock(return_value=WatchScanResult()))
    scheduler = _RecordingScheduler()
    watch_scheduler = WatchScheduler(
        service,  # type: ignore[arg-type]
        scan_hours=6,
        scheduler=scheduler,  # type: ignore[arg-type]
    )
    watch_scheduler.register()
    listener = scheduler.listeners[0][0]

    with caplog.at_level(logging.ERROR):
        listener(SimpleNamespace(job_id="another-job", exception=RuntimeError("ignore")))  # type: ignore[operator]
        listener(SimpleNamespace(job_id=WATCH_SCAN_JOB_ID, exception=RuntimeError("boom")))  # type: ignore[operator]

    assert "Scheduled watch scan failed" in caplog.text
    assert caplog.text.count("Scheduled watch scan failed") == 1


def test_scheduler_rejects_invalid_intervals() -> None:
    service = SimpleNamespace(scan_enabled_topics=AsyncMock(return_value=WatchScanResult()))

    with pytest.raises(ValueError, match="scan_hours"):
        WatchScheduler(service, scan_hours=0)  # type: ignore[arg-type]
