"""APScheduler integration for one-process ResearchRadar watch monitoring."""

from __future__ import annotations

import logging

from apscheduler.events import EVENT_JOB_ERROR, JobExecutionEvent
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from research_radar.watch.service import WatchScanResult, WatchService

logger = logging.getLogger(__name__)

WATCH_SCAN_JOB_ID = "research_radar.watch_scan"


class WatchScheduler:
    """Register and own the periodic watch scan job for the main process.

    ``start()`` must run on the application's asyncio event loop. The service
    itself prevents overlap, while APScheduler additionally uses
    ``max_instances=1`` and ``coalesce=True`` to protect scheduled executions.
    """

    def __init__(
        self,
        watch_service: WatchService,
        *,
        scan_hours: int,
        scheduler: AsyncIOScheduler | None = None,
    ) -> None:
        if scan_hours < 1:
            raise ValueError("scan_hours must be at least 1.")

        self._watch_service = watch_service
        self._scan_hours = scan_hours
        self._scheduler = scheduler
        self._registered = False

    @property
    def scheduler(self) -> AsyncIOScheduler:
        """Return the owned/injected scheduler, constructing it lazily."""

        if self._scheduler is None:
            self._scheduler = AsyncIOScheduler()
        return self._scheduler

    def register(self) -> None:
        """Register one interval job and its explicit failure listener once."""

        if self._registered:
            return

        scheduler = self.scheduler
        scheduler.add_listener(self._handle_job_error, EVENT_JOB_ERROR)
        scheduler.add_job(
            self._run_scan,
            trigger="interval",
            hours=self._scan_hours,
            id=WATCH_SCAN_JOB_ID,
            name="ResearchRadar watch scan",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self._registered = True
        logger.info("Registered watch scan job with a %d-hour interval.", self._scan_hours)

    def start(self) -> None:
        """Start the scheduler if it is not already running."""

        self.register()
        if self.scheduler.running:
            return
        self.scheduler.start()
        logger.info("Started ResearchRadar watch scheduler.")

    async def shutdown(self) -> None:
        """Request clean, non-blocking scheduler shutdown during app teardown."""

        if not self._scheduler or not self._scheduler.running:
            return
        self._scheduler.shutdown(wait=False)
        logger.info("Stopped ResearchRadar watch scheduler.")

    async def _run_scan(self) -> WatchScanResult:
        """Run the service job; unhandled global failures reach the listener."""

        result = await self._watch_service.scan_enabled_topics()
        if result.skipped_due_to_overlap:
            logger.warning("Scheduled watch scan skipped because a run was already active.")
        return result

    def _handle_job_error(self, event: JobExecutionEvent) -> None:
        """Log scheduler job errors without terminating the bot process."""

        if event.job_id != WATCH_SCAN_JOB_ID or event.exception is None:
            return
        logger.error(
            "Scheduled watch scan failed for job %s: %s",
            event.job_id,
            event.exception,
            exc_info=(
                type(event.exception),
                event.exception,
                event.exception.__traceback__,
            ),
        )
