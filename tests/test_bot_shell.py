"""Tests for the Discord boundary that do not require a token or gateway."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from research_radar.bot.client import create_bot
from research_radar.bot.commands.ping import PING_RESPONSE, ping
from research_radar.config import Settings
from research_radar.models import Paper
from research_radar.research.ranker import RankedPaper
from research_radar.research.service import SearchResult


class _FakeResponse:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_message(self, content: str) -> None:
        self.messages.append(content)


class _FakeInteraction:
    def __init__(self) -> None:
        self.response = _FakeResponse()


@pytest.mark.asyncio
async def test_ping_returns_exact_online_message() -> None:
    interaction = _FakeInteraction()

    await ping(interaction)  # type: ignore[arg-type]

    assert interaction.response.messages == [PING_RESPONSE]


@pytest.mark.asyncio
async def test_bot_factory_registers_only_ping_without_message_content_intent() -> None:
    bot = create_bot(Settings())
    try:
        commands = bot.tree.get_commands()

        assert [command.name for command in commands] == ["ping"]
        assert bot.intents.guilds is True
        assert bot.intents.message_content is False
    finally:
        await bot.close()


@pytest.mark.asyncio
async def test_command_sync_is_global_without_a_development_guild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = create_bot(Settings())
    sync = AsyncMock(return_value=[])
    monkeypatch.setattr(bot.tree, "sync", sync)
    try:
        commands = await bot.sync_application_commands()

        assert commands == []
        sync.assert_awaited_once_with()
    finally:
        await bot.close()


@pytest.mark.asyncio
async def test_command_sync_targets_configured_development_guild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = create_bot(Settings(discord_guild_id=123456))
    copy_global_to = Mock()
    sync = AsyncMock(return_value=[])
    monkeypatch.setattr(bot.tree, "copy_global_to", copy_global_to)
    monkeypatch.setattr(bot.tree, "sync", sync)
    try:
        commands = await bot.sync_application_commands()

        assert commands == []
        guild = copy_global_to.call_args.kwargs["guild"]
        assert guild.id == 123456
        sync.assert_awaited_once_with(guild=guild)
    finally:
        await bot.close()


@pytest.mark.asyncio
async def test_shutdown_hooks_are_run_once_in_reverse_registration_order() -> None:
    calls: list[str] = []

    async def close_first() -> None:
        calls.append("first")

    async def close_second() -> None:
        calls.append("second")

    bot = create_bot(Settings(), shutdown_hooks=(close_first, close_second))
    try:
        await bot.close_owned_resources()
        await bot.close_owned_resources()

        assert calls == ["second", "first"]
        with pytest.raises(RuntimeError, match="shutdown hook"):
            bot.add_shutdown_hook(close_first)
    finally:
        await bot.close()


@pytest.mark.asyncio
async def test_setup_hook_syncs_commands_before_starting_application_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def start_resource() -> None:
        calls.append("start")

    bot = create_bot(Settings(), startup_hooks=(start_resource,))
    sync = AsyncMock(side_effect=lambda: calls.append("sync") or [])
    monkeypatch.setattr(bot, "sync_application_commands", sync)
    try:
        await bot.setup_hook()

        assert calls == ["sync", "start"]
        sync.assert_awaited_once_with()
    finally:
        await bot.close()


@pytest.mark.asyncio
async def test_paper_command_is_registered_only_when_a_service_is_provided() -> None:
    class _ResearchService:
        async def search(self, query: str, limit: int) -> SearchResult:
            return SearchResult(papers=[])

    bot = create_bot(Settings(), research_service=_ResearchService())  # type: ignore[arg-type]
    try:
        assert [command.name for command in bot.tree.get_commands()] == ["ping", "paper"]
    finally:
        await bot.close()


@pytest.mark.asyncio
async def test_bot_registers_all_optional_service_command_adapters() -> None:
    class _ResearchService:
        async def search(self, query: str, limit: int) -> SearchResult:
            return SearchResult(papers=[])

    class _WatchService:
        async def add_topic(self, name: str, query: str) -> object:
            return object()

        async def list_topics(self) -> list[object]:
            return []

        async def remove_topic(self, topic_id_or_name: str) -> bool:
            return False

    class _ReaderService:
        async def read_url(self, url: str) -> object:
            return object()

    class _DigestService:
        async def build_on_demand(self) -> object:
            return object()

    bot = create_bot(
        Settings(),
        research_service=_ResearchService(),  # type: ignore[arg-type]
        watch_service=_WatchService(),
        reader_service=_ReaderService(),
        digest_service=_DigestService(),
    )
    try:
        assert [command.name for command in bot.tree.get_commands()] == [
            "ping",
            "paper",
            "watch",
            "read",
            "digest",
        ]
    finally:
        await bot.close()


def test_paper_search_embeds_render_normalized_metadata() -> None:
    from research_radar.bot.embeds import paper_search_embeds

    paper = Paper(
        id="openalex:1",
        title="Useful paper",
        authors=["Ada"],
        publication_year=2025,
        venue="Venue",
        doi="10.1/example",
        citation_count=3,
        source="openalex",
    )
    result = SearchResult(
        papers=[RankedPaper(paper, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0)],
        warnings=["arxiv was unavailable; results may be partial."],
    )

    embed = paper_search_embeds(result)[0]

    assert embed.title == "1. Useful paper"
    assert "Ada" in (embed.description or "")
    assert embed.url == "https://doi.org/10.1/example"


def test_paper_search_embeds_bound_untrusted_provider_metadata() -> None:
    from research_radar.bot.embeds import (
        MAX_EMBED_DESCRIPTION_CHARS,
        MAX_EMBED_TITLE_CHARS,
        paper_search_embeds,
    )

    paper = Paper(
        id="openalex:huge",
        title="T" * 2_000,
        authors=["A" * 1_000] * 30,
        venue="V" * 1_000,
        url="https://example.test/" + "x" * 3_000,
        source="openalex",
    )
    results = [
        RankedPaper(paper.model_copy(update={"id": f"openalex:{index}"}), 1, 1, 1, 1, 1, 1)
        for index in range(12)
    ]

    embeds = paper_search_embeds(SearchResult(papers=results))

    assert len(embeds) == 10
    assert all(len(embed.title or "") <= MAX_EMBED_TITLE_CHARS for embed in embeds)
    assert all(len(embed.description or "") <= MAX_EMBED_DESCRIPTION_CHARS for embed in embeds)
    assert all(embed.url is None for embed in embeds)
