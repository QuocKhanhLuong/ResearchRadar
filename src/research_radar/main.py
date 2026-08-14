"""Application composition root and Discord bot launcher."""

from __future__ import annotations

import logging
from zoneinfo import ZoneInfo

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from research_radar.bot.client import ResearchRadarBot, create_bot
from research_radar.bot.notifications import DiscordNotificationSink
from research_radar.config import Settings, get_settings
from research_radar.digest.scheduler import DigestScheduler
from research_radar.digest.service import DigestService
from research_radar.gap.service import GapService
from research_radar.logging import configure_logging
from research_radar.providers.arxiv import ArxivProvider
from research_radar.providers.base import PaperProvider
from research_radar.providers.openalex import OpenAlexProvider
from research_radar.providers.semantic_scholar import SemanticScholarProvider
from research_radar.reader.fetcher import DirectPDFFetcher
from research_radar.reader.llm.base import LLMProvider
from research_radar.reader.llm.mock import MockLLMProvider
from research_radar.reader.llm.remote import RemoteLLMProvider
from research_radar.reader.parser import PDFParser
from research_radar.reader.service import ReaderService
from research_radar.research.ask import AskService
from research_radar.research.scout import ScoutService
from research_radar.research.service import ResearchService
from research_radar.storage.database import Database
from research_radar.storage.repositories import ResearchRepository
from research_radar.watch.scheduler import WatchScheduler
from research_radar.watch.service import WatchService

logger = logging.getLogger(__name__)


def build_application_bot(settings: Settings | None = None) -> ResearchRadarBot:
    """Compose all ResearchRadar services into a configured Discord bot instance.

    This function does NOT connect to Discord or require a DISCORD_TOKEN, making it
    fully constructible and testable in offline/unit environments.
    """
    settings = settings or get_settings()

    db = Database.create(settings.database_url)
    db.initialize_schema()
    repository = ResearchRepository(db)

    http_client = httpx.AsyncClient(timeout=httpx.Timeout(settings.http_timeout_seconds))

    openalex_api_key = (
        settings.openalex_api_key.get_secret_value()
        if settings.openalex_api_key
        else None
    )
    s2_api_key = (
        settings.semantic_scholar_api_key.get_secret_value()
        if settings.semantic_scholar_api_key
        else None
    )

    providers: list[PaperProvider] = [
        ArxivProvider(http_client, timeout_seconds=settings.http_timeout_seconds),
        OpenAlexProvider(
            http_client,
            email=settings.openalex_email,
            api_key=openalex_api_key,
            timeout_seconds=settings.http_timeout_seconds,
        ),
        SemanticScholarProvider(
            http_client,
            api_key=s2_api_key,
            timeout_seconds=settings.http_timeout_seconds,
        ),
    ]
    scout = ScoutService(providers)
    research_service = ResearchService(scout)

    llm: LLMProvider
    if (
        settings.llm_provider == "remote"
        and settings.llm_base_url
        and settings.llm_model
    ):
        llm_api_key = (
            settings.llm_api_key.get_secret_value()
            if settings.llm_api_key
            else None
        )
        llm = RemoteLLMProvider(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=llm_api_key,
            client=http_client,
            timeout_seconds=settings.http_timeout_seconds,
        )
    else:
        llm = MockLLMProvider()

    fetcher = DirectPDFFetcher(client=http_client)
    parser = PDFParser()
    reader_service = ReaderService(
        fetcher=fetcher,
        parser=parser,
        llm=llm,
        repository=repository,
        llm_provider_name=settings.llm_provider,
        llm_model=settings.llm_model,
    )

    notification_sink: DiscordNotificationSink | None = None
    if settings.discord_channel_id is not None:
        notification_sink = DiscordNotificationSink(settings.discord_channel_id)

    watch_service = WatchService(
        repository,
        research_service,
        notification_sink=notification_sink,
    )
    digest_service = DigestService(
        repository,
        notification_sink=notification_sink,
    )

    scheduler_tz = ZoneInfo(settings.timezone)
    apscheduler = AsyncIOScheduler(timezone=scheduler_tz)
    watch_scheduler = WatchScheduler(
        watch_service,
        scan_hours=settings.watch_scan_hours,
        scheduler=apscheduler,
    )
    digest_scheduler = DigestScheduler(
        digest_service,
        digest_hour=settings.digest_hour,
        timezone=scheduler_tz,
        scheduler=apscheduler,
    )
    watch_scheduler.register()
    digest_scheduler.register()

    gap_service = GapService(repository=repository, scout=scout)
    ask_service = AskService(repository=repository, llm_provider=llm)

    bot: ResearchRadarBot | None = None

    async def on_startup() -> None:
        if notification_sink is not None and bot is not None:
            notification_sink.bind_client(bot)
        if not apscheduler.running:
            apscheduler.start()

    async def on_shutdown() -> None:
        if apscheduler.running:
            apscheduler.shutdown(wait=False)
        await http_client.aclose()

    bot = create_bot(
        settings,
        startup_hooks=[on_startup],
        shutdown_hooks=[on_shutdown],
        research_service=research_service,
        watch_service=watch_service,
        reader_service=reader_service,
        digest_service=digest_service,
        gap_service=gap_service,
        project_service=repository,
        ask_service=ask_service,
    )
    return bot


def main() -> None:
    """Launch the ResearchRadar Discord bot with all composed services."""
    configure_logging()
    settings = get_settings()
    logger.info("Starting ResearchRadar (database=%s)...", settings.database_url)
    token = settings.require_discord_token()
    bot = build_application_bot(settings)
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
