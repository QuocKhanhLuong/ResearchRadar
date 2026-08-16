"""Unit tests for Discord interaction helper and global error handler."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest
from discord import app_commands

from research_radar.bot.interactions import (
    ALREADY_ACKNOWLEDGED_CODE,
    UNKNOWN_INTERACTION_CODE,
    is_already_acknowledged_error,
    is_unknown_interaction_error,
    on_app_command_error,
    safe_defer,
)


def _make_http_exception(status: int, code: int, message: str) -> discord.HTTPException:
    resp = MagicMock(status=status, reason="Error")
    return discord.HTTPException(resp, {"code": code, "message": message})


def _make_not_found_exception(
    code: int = 10062,
    message: str = "Unknown interaction",
) -> discord.NotFound:
    resp = MagicMock(status=404, reason="Not Found")
    return discord.NotFound(resp, {"code": code, "message": message})


class _FakeInteraction:
    def __init__(self, *, is_done: bool = False, interaction_id: int = 12345) -> None:
        self.id = interaction_id
        self._is_done = is_done
        self.response = SimpleNamespace(
            defer=AsyncMock(),
            is_done=lambda: self._is_done,
            send_message=AsyncMock(),
        )
        self.followup = SimpleNamespace(
            send=AsyncMock(),
        )
        self.command = SimpleNamespace(name="test-command")


@pytest.mark.asyncio
async def test_safe_defer_normal_success() -> None:
    interaction = _FakeInteraction(is_done=False)

    result = await safe_defer(interaction, thinking=True)

    assert result is True
    interaction.response.defer.assert_awaited_once_with(thinking=True)


@pytest.mark.asyncio
async def test_safe_defer_ephemeral_success() -> None:
    interaction = _FakeInteraction(is_done=False)

    result = await safe_defer(interaction, thinking=True, ephemeral=True)

    assert result is True
    interaction.response.defer.assert_awaited_once_with(thinking=True, ephemeral=True)


@pytest.mark.asyncio
async def test_safe_defer_already_done_skips_defer() -> None:
    interaction = _FakeInteraction(is_done=True)

    result = await safe_defer(interaction, thinking=True)

    assert result is True
    interaction.response.defer.assert_not_called()


@pytest.mark.asyncio
async def test_safe_defer_handles_interaction_responded() -> None:
    interaction = _FakeInteraction(is_done=False)
    interaction.response.defer.side_effect = discord.InteractionResponded(MagicMock())

    result = await safe_defer(interaction, thinking=True)

    assert result is True


@pytest.mark.asyncio
async def test_safe_defer_handles_http_40060_already_acknowledged() -> None:
    interaction = _FakeInteraction(is_done=False)
    interaction.response.defer.side_effect = _make_http_exception(
        status=400,
        code=ALREADY_ACKNOWLEDGED_CODE,
        message="Interaction has already been acknowledged",
    )

    result = await safe_defer(interaction, thinking=True)

    assert result is True


@pytest.mark.asyncio
async def test_safe_defer_handles_not_found_10062_unknown_interaction() -> None:
    interaction = _FakeInteraction(is_done=False)
    interaction.response.defer.side_effect = _make_not_found_exception(
        code=UNKNOWN_INTERACTION_CODE,
        message="Unknown interaction",
    )

    result = await safe_defer(interaction, thinking=True)

    assert result is False


@pytest.mark.asyncio
async def test_safe_defer_handles_http_10062_without_not_found_subclass() -> None:
    interaction = _FakeInteraction(is_done=False)
    interaction.response.defer.side_effect = _make_http_exception(
        status=404,
        code=UNKNOWN_INTERACTION_CODE,
        message="Unknown interaction",
    )

    result = await safe_defer(interaction, thinking=True)

    assert result is False


@pytest.mark.asyncio
async def test_safe_defer_reraises_unrelated_http_exception() -> None:
    interaction = _FakeInteraction(is_done=False)
    interaction.response.defer.side_effect = _make_http_exception(
        status=403,
        code=50001,
        message="Missing Access",
    )

    with pytest.raises(discord.HTTPException) as exc_info:
        await safe_defer(interaction, thinking=True)

    assert exc_info.value.code == 50001


def test_error_classification_helpers() -> None:
    exc_10062 = _make_not_found_exception(10062)
    exc_40060 = _make_http_exception(400, 40060, "Already acknowledged")
    exc_other = _make_http_exception(500, 50000, "Internal error")
    exc_responded = discord.InteractionResponded(MagicMock())
    exc_runtime = RuntimeError("standard error")

    assert is_unknown_interaction_error(exc_10062) is True
    assert is_unknown_interaction_error(exc_40060) is False
    assert is_unknown_interaction_error(exc_other) is False
    assert is_unknown_interaction_error(exc_runtime) is False

    assert is_already_acknowledged_error(exc_40060) is True
    assert is_already_acknowledged_error(exc_responded) is True
    assert is_already_acknowledged_error(exc_10062) is False
    assert is_already_acknowledged_error(exc_other) is False
    assert is_already_acknowledged_error(exc_runtime) is False


@pytest.mark.asyncio
async def test_on_app_command_error_handles_unknown_interaction() -> None:
    interaction = _FakeInteraction(is_done=False)
    command = MagicMock()
    command.name = "test-cmd"
    error = app_commands.CommandInvokeError(command, _make_not_found_exception(10062))

    await on_app_command_error(interaction, error)

    # Should not attempt sending responses to expired interaction
    interaction.response.send_message.assert_not_called()
    interaction.followup.send.assert_not_called()


@pytest.mark.asyncio
async def test_on_app_command_error_handles_already_acknowledged() -> None:
    interaction = _FakeInteraction(is_done=True)
    command = MagicMock()
    command.name = "test-cmd"
    error = app_commands.CommandInvokeError(
        command,
        _make_http_exception(400, 40060, "Already acknowledged"),
    )

    await on_app_command_error(interaction, error)

    interaction.response.send_message.assert_not_called()
    interaction.followup.send.assert_not_called()


@pytest.mark.asyncio
async def test_on_app_command_error_handles_unexpected_error() -> None:
    interaction = _FakeInteraction(is_done=False)
    command = MagicMock()
    command.name = "test-cmd"
    error = app_commands.CommandInvokeError(command, RuntimeError("Real bug"))

    await on_app_command_error(interaction, error)

    interaction.response.send_message.assert_awaited_once_with(
        "An unexpected error occurred while processing this command.",
        ephemeral=True,
    )
