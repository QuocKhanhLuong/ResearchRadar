"""Automated tests for the application composition root and bot construction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from research_radar.config import Settings
from research_radar.errors import ConfigurationError
from research_radar.main import build_application_bot, main


@pytest.mark.asyncio
async def test_build_application_bot_constructs_all_services_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_file = tmp_path / "test_composition.db"
    settings = Settings(
        database_url=f"sqlite:///{db_file}",
        discord_guild_id=987654321,
    )

    bot = build_application_bot(settings)
    try:
        command_names = {command.name for command in bot.tree.get_commands()}
        expected_commands = {
            "ping",
            "paper",
            "watch",
            "read",
            "digest",
            "gap",
            "project-create",
            "project-list",
            "project-show",
            "project-add-paper",
            "project-add-gap",
            "ask",
        }
        assert expected_commands.issubset(command_names)

        # Verify startup and shutdown hooks execute cleanly without Discord gateway
        monkeypatch.setattr(bot, "sync_application_commands", AsyncMock(return_value=[]))
        await bot.setup_hook()
        await bot.close_owned_resources()
    finally:
        await bot.close()


@pytest.mark.asyncio
async def test_build_application_bot_with_remote_llm_and_notification_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_file = tmp_path / "test_composition_remote.db"
    settings = Settings(
        database_url=f"sqlite:///{db_file}",
        discord_channel_id=123456789,
        llm_provider="remote",
        llm_base_url="https://api.openai.com/v1",
        llm_model="gpt-4o-mini",
        llm_api_key=SecretStr("sk-test-key-123"),
    )

    bot = build_application_bot(settings)
    try:
        command_names = {command.name for command in bot.tree.get_commands()}
        assert "ask" in command_names
        assert "read" in command_names

        monkeypatch.setattr(bot, "sync_application_commands", AsyncMock(return_value=[]))
        await bot.setup_hook()
        await bot.close_owned_resources()
    finally:
        await bot.close()


def test_main_fails_fast_when_discord_token_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "research_radar.main.get_settings",
        lambda: Settings(discord_token=None, _env_file=None),
    )
    with pytest.raises(ConfigurationError, match="DISCORD_TOKEN is required"):
        main()
