"""Discord-independent daily digest domain services and scheduling."""

from research_radar.digest.scheduler import DAILY_DIGEST_JOB_ID, DigestScheduler
from research_radar.digest.service import (
    DigestNotificationSink,
    DigestPaper,
    DigestService,
    PaperCardInsight,
    ResearchDigest,
    ScheduledDigestResult,
    WatchActivity,
)

__all__ = [
    "DAILY_DIGEST_JOB_ID",
    "DigestNotificationSink",
    "DigestPaper",
    "DigestScheduler",
    "DigestService",
    "PaperCardInsight",
    "ResearchDigest",
    "ScheduledDigestResult",
    "WatchActivity",
]
