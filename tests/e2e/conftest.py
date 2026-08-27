"""Fixtures for the offline personal-research-chat end-to-end harness.

Production modules under ``research_radar.chat`` / ``research_radar.memory`` /
``research_radar.bot.mention`` are being written concurrently by other workers,
so every fixture imports them lazily inside its body. The test module itself
gates collection with ``pytest.importorskip``, keeping this suite green until
the coordinator integrates the other branches.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import text

from e2e.fakes import (
    DEFAULT_CHAT_TOPIC,
    FakeSemanticIndex,
    RecordingLLMProvider,
    build_fake_user_memory,
    provider_trio_for_topic,
)
from research_radar.research.ingestion import IngestionService
from research_radar.research.scout import ScoutService
from research_radar.storage.database import Database, create_database, initialize_schema
from research_radar.storage.ingestion_repository import IngestionRepository
from research_radar.storage.repositories import ResearchRepository


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    """Create a real temporary file-backed SQLite database with schema applied."""

    database = create_database(f"sqlite:///{tmp_path / 'chat_e2e.db'}")
    initialize_schema(database)
    yield database
    database.dispose()


@pytest.fixture
def repository(database: Database) -> ResearchRepository:
    """Return the canonical research repository over the temporary database."""

    return ResearchRepository(database)


@pytest.fixture
def ingestion_repository(database: Database) -> IngestionRepository:
    """Return the ingestion audit repository over the temporary database."""

    return IngestionRepository(database)


@pytest.fixture
def capture_policy():
    """Return the production memory capture policy with defaults enabled."""

    from research_radar.memory.capture import MemoryCapturePolicy

    return MemoryCapturePolicy()


@pytest.fixture
def settings():
    """Return application settings with safe offline defaults."""

    from research_radar.config import Settings

    return Settings()


@pytest.fixture
def mention_policy(settings):
    """Return the production mention admission policy."""

    from research_radar.bot.mention import MentionPolicy

    return MentionPolicy(settings)


@pytest.fixture
def chat_stack(
    repository: ResearchRepository,
    ingestion_repository: IngestionRepository,
    capture_policy,
) -> Callable[..., SimpleNamespace]:
    """Build a fully wired offline ChatService stack with per-call overrides."""

    def build(
        *,
        user_memory=None,
        llm=None,
        semantic_index: FakeSemanticIndex | None = None,
        topic: str = DEFAULT_CHAT_TOPIC,
        works: int = 3,
        ingestion_service: IngestionService | None = None,
        embedding_provider: Any = None,
        budget: Any = None,
    ) -> SimpleNamespace:
        from research_radar.chat.router import ChatRouter
        from research_radar.chat.service import ChatService

        resolved_llm = llm if llm is not None else RecordingLLMProvider()
        resolved_memory = user_memory if user_memory is not None else build_fake_user_memory()
        providers = provider_trio_for_topic(topic, works=works)
        resolved_ingestion = ingestion_service
        if resolved_ingestion is None:
            resolved_ingestion = IngestionService(
                scout=ScoutService(list(providers)),
                repository=repository,
                ingestion_repository=ingestion_repository,
                reader_service=None,
                metadata_limit=50,
            )
        service = ChatService(
            repository=repository,
            router=ChatRouter(llm_provider=resolved_llm),
            user_memory=resolved_memory,
            capture_policy=capture_policy,
            llm_provider=resolved_llm,
            ingestion_service=resolved_ingestion,
            embedding_provider=embedding_provider,
            semantic_index=semantic_index,
            budget=budget,
        )
        return SimpleNamespace(
            service=service,
            llm=resolved_llm,
            user_memory=resolved_memory,
            providers=providers,
            ingestion_service=resolved_ingestion,
            repository=repository,
            ingestion_repository=ingestion_repository,
        )

    return build


@pytest.fixture
def direct_ingestion(
    repository: ResearchRepository, ingestion_repository: IngestionRepository
) -> Callable[..., tuple[list[Any], IngestionService]]:
    """Build a standalone ingestion service over a fresh provider trio."""

    def build(topic: str, works: int = 3):
        providers = provider_trio_for_topic(topic, works=works)
        service = IngestionService(
            scout=ScoutService(list(providers)),
            repository=repository,
            ingestion_repository=ingestion_repository,
            reader_service=None,
            metadata_limit=50,
        )
        return providers, service

    return build


@pytest.fixture
def semantic_index() -> FakeSemanticIndex:
    """Return a switchable fake semantic index starting fully available."""

    return FakeSemanticIndex()


@pytest.fixture
def count_rows(database: Database) -> Callable[[str], int]:
    """Return a helper reading raw row counts from the temporary database."""

    def read(table_name: str) -> int:
        with database.engine.connect() as connection:
            return int(connection.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())

    return read
