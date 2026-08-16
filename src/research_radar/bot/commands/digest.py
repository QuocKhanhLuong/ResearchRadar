"""Thin Discord slash-command adapter for persisted research digests."""

from __future__ import annotations

import logging
from typing import Protocol

import discord
from discord import app_commands

from research_radar.bot.interactions import safe_defer

logger = logging.getLogger(__name__)

MAX_DISCORD_CONTENT_CHARS = 2_000
_NO_MENTIONS = discord.AllowedMentions.none()


class DigestRenderable(Protocol):
    """A compact, display-ready digest returned by the injected service."""

    def render_text(self) -> str:
        """Return a human-readable digest derived from persisted research data."""


class DigestCommandService(Protocol):
    """The narrow digest-service surface required by Discord."""

    async def build_on_demand(self) -> DigestRenderable:
        """Build a digest from persisted research data without a new search."""


def register_digest_command(
    tree: app_commands.CommandTree[discord.Client],
    digest_service: DigestCommandService,
) -> None:
    """Register an on-demand ``/digest`` command over an injected service."""

    async def digest(interaction: discord.Interaction) -> None:
        if not await safe_defer(interaction, thinking=True):
            return
        try:
            result = await digest_service.build_on_demand()
            content = result.render_text()
            if not isinstance(content, str):
                raise TypeError("Digest renderer returned non-text content.")
        except ValueError as error:
            logger.info("Digest request was rejected: %s", error)
            await _edit(interaction, "I couldn't build that digest. Please try again later.")
            return
        except Exception:
            logger.exception("On-demand digest generation failed.")
            await _edit(
                interaction,
                "The research digest is temporarily unavailable. Please try again later.",
            )
            return

        await _edit(
            interaction,
            content.strip() or "No digest content is available for the selected period.",
        )

    tree.add_command(
        app_commands.Command(
            name="digest",
            description="Show a digest of recently stored research.",
            callback=digest,
        )
    )


async def _edit(interaction: discord.Interaction, content: str) -> None:
    await interaction.edit_original_response(
        content=_truncate(content, MAX_DISCORD_CONTENT_CHARS),
        allowed_mentions=_NO_MENTIONS,
    )


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: max(1, limit - 1)].rstrip()}…"
