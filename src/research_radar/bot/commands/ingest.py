"""Thin slash-command adapter for research-memory ingestion (/ingest)."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

import discord
from discord import app_commands

from research_radar.bot.embeds import discovery_warning_text
from research_radar.bot.interactions import safe_defer
from research_radar.errors import ProviderUnavailableError

logger = logging.getLogger(__name__)

PROVIDERS_UNAVAILABLE_MESSAGE = (
    "Paper sources are temporarily unavailable. Please try again later."
)

_NO_MENTIONS = discord.AllowedMentions.none()
_MAX_EMBED_FIELD_CHARS = 1024


class IngestCommandRegistrationService(Protocol):
    """Minimum async surface required to register the Discord ingest command."""

    async def ingest_research_topic(
        self,
        query: str,
        *,
        limit: int = 20,
        project_id: str | None = None,
        auto_read: int = 0,
    ) -> object: ...


class IngestResultView(Protocol):
    """Structural view of the result fields this adapter renders."""

    @property
    def run_id(self) -> str: ...

    @property
    def discovered_count(self) -> int: ...

    @property
    def canonical_count(self) -> int: ...

    @property
    def provider_counts(self) -> Mapping[str, int]: ...

    @property
    def warnings(self) -> Sequence[str]: ...


def _ingest_summary_embed(
    query: str,
    view: IngestResultView,
) -> discord.Embed:
    """Render a compact one-screen summary of a completed ingestion run."""

    embed = discord.Embed(
        title="Ingest complete",
        description=f"**Query:** {query}",
    )
    embed.add_field(name="Canonical papers stored", value=str(view.canonical_count), inline=True)
    embed.add_field(name="Raw records discovered", value=str(view.discovered_count), inline=True)
    provider_lines = "\n".join(
        f"• {provider}: {count}" for provider, count in sorted(view.provider_counts.items())
    )
    embed.add_field(
        name="Provider results",
        value=provider_lines or "None",
        inline=False,
    )
    embed.set_footer(text=f"Run ID: {view.run_id}")
    return embed


def register_ingest_command(
    tree: app_commands.CommandTree[discord.Client],
    ingestion_service: IngestCommandRegistrationService,
) -> None:
    """Register ``/ingest`` as a presentation-only adapter over the ingestion service."""

    async def ingest_cmd(
        interaction: discord.Interaction,
        query: app_commands.Range[str, 1, 300],
        count: app_commands.Range[int, 1, 50] = 20,
        project: str | None = None,
    ) -> None:
        if not await safe_defer(interaction, thinking=True):
            return
        try:
            # auto_read is intentionally never set from Discord; LLM reads must
            # never be triggered implicitly by a slash command.
            result = await ingestion_service.ingest_research_topic(
                query, limit=count, project_id=project
            )
        except ValueError as error:
            await interaction.edit_original_response(
                content=str(error), allowed_mentions=_NO_MENTIONS
            )
            return
        except ProviderUnavailableError:
            logger.exception("Paper ingestion failed for a provider-unavailable request.")
            await interaction.edit_original_response(
                content=PROVIDERS_UNAVAILABLE_MESSAGE, allowed_mentions=_NO_MENTIONS
            )
            return

        view = cast(IngestResultView, result)
        await interaction.edit_original_response(
            content=discovery_warning_text(view),
            embed=_ingest_summary_embed(query, view),
            allowed_mentions=_NO_MENTIONS,
        )

    tree.add_command(
        app_commands.Command(
            name="ingest",
            description="Discover and store papers for a research query.",
            callback=ingest_cmd,
        )
    )
