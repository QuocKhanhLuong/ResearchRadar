"""Safe Discord interaction acknowledgement and lifecycle helpers."""

from __future__ import annotations

import logging
from typing import Any

import discord
from discord import app_commands

logger = logging.getLogger(__name__)

# Discord API error codes
UNKNOWN_INTERACTION_CODE = 10062
ALREADY_ACKNOWLEDGED_CODE = 40060


def is_unknown_interaction_error(exc: BaseException) -> bool:
    """Return True if exception represents an expired or unknown Discord interaction."""
    if isinstance(exc, discord.HTTPException):
        return exc.code == UNKNOWN_INTERACTION_CODE
    return False


def is_already_acknowledged_error(exc: BaseException) -> bool:
    """Return True if exception represents a duplicate or already acknowledged interaction."""
    if isinstance(exc, discord.InteractionResponded):
        return True
    if isinstance(exc, discord.HTTPException):
        return exc.code == ALREADY_ACKNOWLEDGED_CODE
    return False


async def safe_defer(
    interaction: discord.Interaction[Any],
    *,
    thinking: bool = True,
    ephemeral: bool = False,
) -> bool:
    """Safely defer a Discord slash command interaction.

    Returns:
        True if the command execution may continue (either newly deferred or already acknowledged).
        False if the interaction expired or is unknown (code 10062), signaling the caller to abort.

    Raises:
        discord.HTTPException: For any unrelated Discord HTTP errors.
    """
    is_done = getattr(interaction.response, "is_done", None)
    if callable(is_done) and is_done():
        logger.debug(
            "Interaction %s is already acknowledged (is_done=True); skipping defer.",
            getattr(interaction, "id", "unknown"),
        )
        return True

    try:
        if ephemeral:
            await interaction.response.defer(thinking=thinking, ephemeral=True)
        else:
            await interaction.response.defer(thinking=thinking)
        return True
    except discord.InteractionResponded:
        logger.debug(
            "Interaction %s was already responded to (InteractionResponded).",
            getattr(interaction, "id", "unknown"),
        )
        return True
    except discord.HTTPException as exc:
        if is_already_acknowledged_error(exc):
            logger.debug(
                "Interaction %s was already acknowledged by Discord (code %s).",
                getattr(interaction, "id", "unknown"),
                exc.code,
            )
            return True
        if is_unknown_interaction_error(exc):
            logger.warning(
                "Discord interaction %s expired or unknown (code 10062); "
                "aborting command execution.",
                getattr(interaction, "id", "unknown"),
            )
            return False
        # Do NOT swallow unrelated Discord HTTP errors
        raise


async def on_app_command_error(
    interaction: discord.Interaction[Any],
    error: app_commands.AppCommandError,
) -> None:
    """Global CommandTree error handler providing clean logging without traceback noise."""
    original = getattr(error, "original", error)
    command_name = interaction.command.name if interaction.command else "unknown"

    if is_unknown_interaction_error(original):
        logger.warning(
            "Slash command '/%s' aborted: interaction %s expired or unknown (code 10062).",
            command_name,
            getattr(interaction, "id", "unknown"),
        )
        return

    if is_already_acknowledged_error(original):
        logger.debug(
            "Slash command '/%s': interaction %s already acknowledged.",
            command_name,
            getattr(interaction, "id", "unknown"),
        )
        return

    if isinstance(error, app_commands.CommandNotFound):
        logger.warning("Application command not found: %s", error)
        return

    # Unexpected errors must still be logged with full traceback
    logger.exception(
        "Unhandled error executing slash command '/%s': %s",
        command_name,
        original,
        exc_info=error,
    )

    try:
        is_done = getattr(interaction.response, "is_done", None)
        if callable(is_done) and is_done():
            await interaction.followup.send(
                "An unexpected error occurred while processing this command.",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                "An unexpected error occurred while processing this command.",
                ephemeral=True,
            )
    except Exception:
        # Avoid crashing in the error handler if sending the error message also fails
        pass
