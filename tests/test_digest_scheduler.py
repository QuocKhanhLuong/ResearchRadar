"""Focused tests for daily digest scheduler registration and lifecycle behavior."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from apscheduler.triggers.cron import CronTrigger

from research_radar.digest.scheduler import DAILY_DIGEST_JOB_ID, DigestScheduler
from research_radar.digest.service import ScheduledDigestResult


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
async def test_scheduler_registers_daily_cron_job_and_has_clean_lifecycle() -> None:
    service = SimpleNamespace(
        run_scheduled_digest=AsyncMock(return_value=ScheduledDigestResult("sent"))
    )
    scheduler = _RecordingScheduler()
    digest_scheduler = DigestScheduler(
        service,  # type: ignore[arg-type]
        digest_hour=8,
        timezone="Asia/Bangkok",
        scheduler=scheduler,  # type: ignore[arg-type]
    )

    digest_scheduler.register()
    digest_scheduler.register()
    digest_scheduler.start()
    digest_scheduler.start()
    run_result = await digest_scheduler._run_digest()
    await digest_scheduler.shutdown()
    await digest_scheduler.shutdown()

    assert run_result == ScheduledDigestResult("sent")
    assert len(scheduler.listeners) == 1
    assert len(scheduler.jobs) == 1
    args, kwargs = scheduler.jobs[0]
    assert callable(args[0])
    assert kwargs["id"] == DAILY_DIGEST_JOB_ID
    assert kwargs["max_instances"] == 1
    assert kwargs["coalesce"] is True
    trigger = kwargs["trigger"]
    assert isinstance(trigger, CronTrigger)
    assert trigger.timezone == ZoneInfo("Asia/Bangkok")
    assert str(trigger) == "cron[hour='8', minute='0']"
    assert scheduler.start_calls == 1
    assert scheduler.shutdown_calls == [False]
    service.run_scheduled_digest.assert_awaited_once_with()


def test_scheduler_error_listener_logs_only_its_own_job(caplog: pytest.LogCaptureFixture) -> None:
    service = SimpleNamespace(
        run_scheduled_digest=AsyncMock(return_value=ScheduledDigestResult("sent"))
    )
    scheduler = _RecordingScheduler()
    digest_scheduler = DigestScheduler(
        service,  # type: ignore[arg-type]
        digest_hour=8,
        timezone=ZoneInfo("Asia/Bangkok"),
        scheduler=scheduler,  # type: ignore[arg-type]
    )
    digest_scheduler.register()
    listener = scheduler.listeners[0][0]

    with caplog.at_level(logging.ERROR):
        listener(SimpleNamespace(job_id="another-job", exception=RuntimeError("ignore")))  # type: ignore[operator]
        listener(SimpleNamespace(job_id=DAILY_DIGEST_JOB_ID, exception=RuntimeError("boom")))  # type: ignore[operator]

    assert "Scheduled digest failed" in caplog.text
    assert caplog.text.count("Scheduled digest failed") == 1


@pytest.mark.parametrize(
    ("digest_hour", "timezone", "message"),
    [
        (-1, "Asia/Bangkok", "digest_hour"),
        (24, "Asia/Bangkok", "digest_hour"),
        (8, "Not/AZone", "IANA timezone"),
    ],
)
def test_scheduler_rejects_invalid_schedule_configuration(
    digest_hour: int,
    timezone: str,
    message: str,
) -> None:
    service = SimpleNamespace(
        run_scheduled_digest=AsyncMock(return_value=ScheduledDigestResult("sent"))
    )

    with pytest.raises(ValueError, match=message):
        DigestScheduler(service, digest_hour=digest_hour, timezone=timezone)  # type: ignore[arg-type]
