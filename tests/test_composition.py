"""Automated tests for the application composition root, lifecycle, and bot construction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from research_radar.bot.commands.memory import OWNER_REFUSAL_MESSAGE
from research_radar.config import Settings
from research_radar.errors import ConfigurationError
from research_radar.main import (
    _build_embedding_provider,
    _build_semantic_index,
    _build_user_memory_store,
    build_application_bot,
    main,
)
from research_radar.memory.disabled import DisabledUserMemoryStore
from research_radar.semantic.embedding import LocalEmbeddingProvider
from research_radar.semantic.index import DisabledSemanticIndex, PineconeSemanticIndex

ALL_EXPECTED_COMMANDS = {
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
    "ingest",
    "memory-status",
    "memory-search",
}


@pytest.mark.asyncio
async def test_build_application_bot_constructs_all_services_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The composition root wires all 15 commands, services, and chat mention handling."""
    db_file = tmp_path / "test_composition.db"
    settings = Settings(
        database_url=f"sqlite:///{db_file}",
        discord_guild_id=987654321,
        chat_live_discovery_limit=10,
    )

    bot = build_application_bot(settings)
    try:
        command_names = {command.name for command in bot.tree.get_commands()}
        assert ALL_EXPECTED_COMMANDS.issubset(command_names)
        assert bot._chat_service is not None
        assert bot._chat_service._budget.max_discovery_results == 10
        assert hasattr(bot, "on_message")

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
    """The composition root binds remote LLM and notification sink when configured."""
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
        assert "memory-status" in command_names
        assert "memory-search" in command_names

        monkeypatch.setattr(bot, "sync_application_commands", AsyncMock(return_value=[]))
        await bot.setup_hook()
        await bot.close_owned_resources()
    finally:
        await bot.close()


def test_composition_builder_helpers(tmp_path: Path) -> None:
    """Internal builder helpers build the correct stores or degrade gracefully."""
    # User memory store builder
    disabled_settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        user_memory_backend="disabled",
    )
    store = _build_user_memory_store(disabled_settings)
    assert isinstance(store, DisabledUserMemoryStore)

    # Embedding provider builder
    emb_disabled = Settings(
        database_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        embedding_provider="disabled",
    )
    assert _build_embedding_provider(emb_disabled) is None

    emb_local = Settings(
        database_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        embedding_provider="local",
    )
    provider = _build_embedding_provider(emb_local)
    assert isinstance(provider, LocalEmbeddingProvider)

    # Semantic index builder
    sem_disabled = Settings(
        database_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        semantic_index="disabled",
    )
    assert isinstance(_build_semantic_index(sem_disabled), DisabledSemanticIndex)

    # Pinecone without key degrades safely to DisabledSemanticIndex
    sem_pinecone_unconfigured = Settings(
        database_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        semantic_index="pinecone",
        pinecone_api_key=None,
        pinecone_index=None,
    )
    assert isinstance(
        _build_semantic_index(sem_pinecone_unconfigured), DisabledSemanticIndex
    )

    # Pinecone configured
    sem_pinecone_configured = Settings(
        database_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        semantic_index="pinecone",
        pinecone_api_key=SecretStr("pcsk_test-api-key"),
        pinecone_index="test-index",
    )
    idx = _build_semantic_index(sem_pinecone_configured)
    assert isinstance(idx, PineconeSemanticIndex)


@pytest.mark.asyncio
async def test_composed_bot_tree_memory_commands_enforce_owner_at_runtime(
    tmp_path: Path,
) -> None:
    """Slash commands in the bot command tree enforce owner permissions at runtime."""
    owner_id = 111_222_333
    stranger_id = 999_888_777
    db_file = tmp_path / "test_composition_owner.db"
    settings = Settings(
        database_url=f"sqlite:///{db_file}",
        discord_owner_user_id=owner_id,
    )

    bot = build_application_bot(settings)
    try:
        commands_by_name = {cmd.name: cmd for cmd in bot.tree.get_commands()}
        assert "memory-status" in commands_by_name
        assert "memory-search" in commands_by_name

        status_cmd = commands_by_name["memory-status"]
        search_cmd = commands_by_name["memory-search"]

        # Helper to create fake interaction
        def make_interaction(user_id: int) -> SimpleNamespace:
            resp = SimpleNamespace(
                send_message=AsyncMock(),
                defer=AsyncMock(),
                is_done=lambda: False,
            )
            followup = SimpleNamespace(send=AsyncMock())
            return SimpleNamespace(
                user=SimpleNamespace(id=user_id),
                response=resp,
                followup=followup,
            )

        # Stranger calls memory-status -> rejected immediately
        stranger_interaction = make_interaction(stranger_id)
        await status_cmd.callback(stranger_interaction)
        stranger_interaction.response.send_message.assert_awaited_once_with(
            OWNER_REFUSAL_MESSAGE, ephemeral=True
        )
        stranger_interaction.response.defer.assert_not_awaited()

        # Owner calls memory-status -> accepted and deferred
        owner_interaction = make_interaction(owner_id)
        await status_cmd.callback(owner_interaction)
        owner_interaction.response.defer.assert_awaited_once_with(
            thinking=True, ephemeral=True
        )
        owner_interaction.followup.send.assert_awaited_once()
        embed = owner_interaction.followup.send.call_args.kwargs.get("embed")
        assert embed is not None
        assert "Personal memory status" in embed.title

        # Stranger calls memory-search -> rejected immediately
        stranger_search = make_interaction(stranger_id)
        await search_cmd.callback(stranger_search, query="test")
        stranger_search.response.send_message.assert_awaited_once_with(
            OWNER_REFUSAL_MESSAGE, ephemeral=True
        )

        # Owner calls memory-search -> accepted and answered
        owner_search = make_interaction(owner_id)
        await search_cmd.callback(owner_search, query="test")
        owner_search.response.defer.assert_awaited_once_with(
            thinking=True, ephemeral=True
        )
        owner_search.followup.send.assert_awaited_once()
        search_embed = owner_search.followup.send.call_args.kwargs.get("embed")
        assert search_embed is not None
        assert "Personal memory search" in search_embed.title
    finally:
        await bot.close_owned_resources()
        await bot.close()


def test_main_fails_fast_when_discord_token_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoking main() without DISCORD_TOKEN raises ConfigurationError immediately."""
    monkeypatch.setattr(
        "research_radar.main.get_settings",
        lambda: Settings(discord_token=None, _env_file=None),
    )
    with pytest.raises(ConfigurationError, match="DISCORD_TOKEN is required"):
        main()

