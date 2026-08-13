"""Tests for the Discord boundary that do not require a token or gateway."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from research_radar.bot.client import create_bot
from research_radar.bot.commands.ping import PING_RESPONSE, ping
from research_radar.config import Settings


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
