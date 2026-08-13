"""APScheduler integration for one-process daily ResearchRadar digests."""

from __future__ import annotations

import logging
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.events import EVENT_JOB_ERROR, JobExecutionEvent
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from research_radar.digest.service import DigestService, ScheduledDigestResult

logger = logging.getLogger(__name__)

DAILY_DIGEST_JOB_ID = "research_radar.daily_digest"


class DigestScheduler:
    """Register and own the daily digest job for the main process.

    The digest service owns durable period claims and its process-local overlap
    protection. APScheduler adds ``max_instances=1`` and ``coalesce=True`` so
    delayed scheduler ticks cannot create a burst of concurrent executions.
    """

    def __init__(
        self,
        digest_service: DigestService,
        *,
        digest_hour: int,
        timezone: str | ZoneInfo,
        scheduler: AsyncIOScheduler | None = None,
    ) -> None:
        if not 0 <= digest_hour <= 23:
            raise ValueError("digest_hour must be between 0 and 23.")

        self._digest_service = digest_service
        self._digest_hour = digest_hour
        self._timezone = _as_zoneinfo(timezone)
        self._scheduler = scheduler
        self._registered = False

    @property
    def scheduler(self) -> AsyncIOScheduler:
        """Return the owned/injected scheduler, constructing it lazily."""

        if self._scheduler is None:
            self._scheduler = AsyncIOScheduler(timezone=self._timezone)
        return self._scheduler

    def register(self) -> None:
        """Register one daily cron job and its failure listener exactly once."""

        if self._registered:
            return

        scheduler = self.scheduler
        scheduler.add_listener(self._handle_job_error, EVENT_JOB_ERROR)
        scheduler.add_job(
            self._run_digest,
            trigger=CronTrigger(
                hour=self._digest_hour,
                minute=0,
                timezone=self._timezone,
            ),
            id=DAILY_DIGEST_JOB_ID,
            name="ResearchRadar daily digest",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self._registered = True
        logger.info(
            "Registered daily digest job for %02d:00 %s.",
            self._digest_hour,
            self._timezone.key,
        )

    def start(self) -> None:
        """Start the scheduler if it is not already running."""

        self.register()
        if self.scheduler.running:
            return
        self.scheduler.start()
        logger.info("Started ResearchRadar digest scheduler.")

    async def shutdown(self) -> None:
        """Request clean, non-blocking scheduler shutdown during app teardown."""

        if self._scheduler is None or not self._scheduler.running:
            return
        self._scheduler.shutdown(wait=False)
        logger.info("Stopped ResearchRadar digest scheduler.")

    async def _run_digest(self) -> ScheduledDigestResult:
        """Run the service job; unexpected failures reach APScheduler's listener."""

        result = await self._digest_service.run_scheduled_digest()
        if result.status == "failed":
            logger.warning("Scheduled digest completed with a recoverable failure.")
        return result

    def _handle_job_error(self, event: JobExecutionEvent) -> None:
        """Log a scheduled job failure without terminating the bot process."""

        if event.job_id != DAILY_DIGEST_JOB_ID or event.exception is None:
            return
        logger.error(
            "Scheduled digest failed for job %s: %s",
            event.job_id,
            event.exception,
            exc_info=(
                type(event.exception),
                event.exception,
                event.exception.__traceback__,
            ),
        )


def _as_zoneinfo(value: str | ZoneInfo) -> ZoneInfo:
    """Return a validated IANA zone without accepting generic tzinfo values."""

    if isinstance(value, ZoneInfo):
        return value
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Unknown IANA timezone: {value}") from error
