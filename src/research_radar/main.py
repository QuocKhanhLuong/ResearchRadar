"""Application composition root and Discord bot launcher."""

from __future__ import annotations

import logging
from zoneinfo import ZoneInfo

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from research_radar.artifacts.local import LocalArtifactStore
from research_radar.bot.client import ResearchRadarBot, create_bot
from research_radar.bot.commands.memory import register_memory_commands
from research_radar.bot.notifications import DiscordNotificationSink
from research_radar.chat.router import ChatRouter
from research_radar.chat.service import ChatBudget, ChatService
from research_radar.config import Settings, get_settings
from research_radar.digest.scheduler import DigestScheduler
from research_radar.digest.service import DigestService
from research_radar.gap.service import GapService
from research_radar.logging import configure_logging
from research_radar.memory.base import UserMemoryStore
from research_radar.memory.capture import MemoryCapturePolicy
from research_radar.memory.disabled import DisabledUserMemoryStore
from research_radar.providers.arxiv import ArxivProvider
from research_radar.providers.base import PaperProvider
from research_radar.providers.openalex import OpenAlexProvider
from research_radar.providers.semantic_scholar import SemanticScholarProvider
from research_radar.reader.cache import DocumentCache
from research_radar.reader.fetcher import DirectPDFFetcher
from research_radar.reader.llm.base import LLMProvider
from research_radar.reader.llm.mock import MockLLMProvider
from research_radar.reader.llm.remote import RemoteLLMProvider
from research_radar.reader.llm.telemetry import InMemoryUsageSink
from research_radar.reader.parser import PDFParser
from research_radar.reader.service import ReaderService
from research_radar.research.ask import AskService
from research_radar.research.ingestion import IngestionService
from research_radar.research.scout import ScoutService
from research_radar.research.service import ResearchService
from research_radar.semantic.base import EmbeddingProvider, SemanticIndex
from research_radar.semantic.embedding import LocalEmbeddingProvider
from research_radar.semantic.index import DisabledSemanticIndex, PineconeSemanticIndex
from research_radar.storage.database import Database
from research_radar.storage.ingestion_repository import IngestionRepository
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
    ingestion_repository = IngestionRepository(db)
    artifact_store = LocalArtifactStore(settings.artifact_root_path())
    document_cache = DocumentCache(
        store=artifact_store,
        ingestion_repository=ingestion_repository,
    )

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
    usage_sink = InMemoryUsageSink()
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
            usage_sink=usage_sink,
            provider_name=settings.llm_provider,
        )
    else:
        llm = MockLLMProvider()

    embedding_provider = _build_embedding_provider(settings)
    semantic_index = _build_semantic_index(settings)

    fetcher = DirectPDFFetcher(client=http_client)
    parser = PDFParser()
    reader_service = ReaderService(
        fetcher=fetcher,
        parser=parser,
        llm=llm,
        repository=repository,
        llm_provider_name=settings.llm_provider,
        llm_model=settings.llm_model,
        document_cache=document_cache,
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
    ask_service = AskService(
        repository=repository,
        llm_provider=llm,
        embedding_provider=embedding_provider,
        semantic_index=semantic_index,
    )
    ingestion_service = IngestionService(
        scout=scout,
        repository=repository,
        ingestion_repository=ingestion_repository,
        reader_service=reader_service,
        metadata_limit=settings.ingestion_metadata_limit,
    )

    user_memory = _build_user_memory_store(settings)
    chat_service = ChatService(
        repository=repository,
        router=ChatRouter(llm_provider=llm),
        user_memory=user_memory,
        capture_policy=MemoryCapturePolicy(enabled=settings.user_memory_capture),
        llm_provider=llm,
        ingestion_service=ingestion_service,
        embedding_provider=embedding_provider,
        semantic_index=semantic_index,
        budget=ChatBudget(max_discovery_results=settings.chat_live_discovery_limit),
    )

    bot: ResearchRadarBot | None = None

    async def on_startup() -> None:
        if notification_sink is not None and bot is not None:
            notification_sink.bind_client(bot)
        if not apscheduler.running:
            apscheduler.start()

    async def on_shutdown() -> None:
        if apscheduler.running:
            apscheduler.shutdown(wait=False)
        await user_memory.close()
        await http_client.aclose()
        db.dispose()

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
        ingestion_service=ingestion_service,
        chat_service=chat_service,
    )
    register_memory_commands(bot.tree, user_memory, settings)
    return bot



def _build_user_memory_store(settings: Settings) -> UserMemoryStore:
    """Return the personal-memory backend, defaulting to a total no-op.

    Personal memory is optional. ``USER_MEMORY_BACKEND=disabled`` (the default)
    keeps chat fully functional, and an unimportable Graphiti install degrades
    to the same disabled store rather than failing startup.
    """

    if settings.user_memory_backend != "graphiti":
        return DisabledUserMemoryStore()
    from research_radar.memory.graphiti_store import GraphitiUserMemoryStore

    return GraphitiUserMemoryStore(settings)


def _build_embedding_provider(settings: Settings) -> EmbeddingProvider | None:
    """Return a configured embedding provider, or None when embeddings are off.

    The local backend loads its model lazily, so constructing it here neither
    imports sentence-transformers nor downloads anything.
    """

    if settings.embedding_provider != "local":
        return None
    return LocalEmbeddingProvider(model_id=settings.embedding_model)


def _build_semantic_index(settings: Settings) -> SemanticIndex:
    """Return the derived semantic index, defaulting to a total no-op.

    Pinecone is optional and derived. When it is not fully configured the
    application keeps working on lexical retrieval alone.
    """

    if settings.semantic_index != "pinecone":
        return DisabledSemanticIndex()
    if settings.pinecone_api_key is None or not settings.pinecone_index:
        logger.warning(
            "SEMANTIC_INDEX=pinecone requires PINECONE_API_KEY and PINECONE_INDEX; "
            "continuing with semantic retrieval disabled."
        )
        return DisabledSemanticIndex()
    return PineconeSemanticIndex(
        api_key=settings.pinecone_api_key.get_secret_value(),
        index_name=settings.pinecone_index,
        namespace=settings.pinecone_namespace,
    )


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
