"""Thin slash-command adapter for paper discovery."""

from __future__ import annotations

import logging

import discord
from discord import app_commands

from research_radar.bot.embeds import discovery_warning_text, paper_search_embeds
from research_radar.errors import ProviderUnavailableError
from research_radar.research.service import ResearchService

logger = logging.getLogger(__name__)


def register_paper_command(
    tree: app_commands.CommandTree[discord.Client], research_service: ResearchService
) -> None:
    """Register `/paper` as a presentation-only adapter over ``ResearchService``."""

    async def paper(
        interaction: discord.Interaction,
        query: app_commands.Range[str, 1, 300],
        count: app_commands.Range[int, 1, 10] = 5,
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            result = await research_service.search(query, count)
        except ValueError as error:
            await interaction.edit_original_response(content=str(error))
            return
        except ProviderUnavailableError:
            logger.exception("Paper search failed for a provider-unavailable request.")
            await interaction.edit_original_response(
                content="Paper sources are temporarily unavailable. Please try again later."
            )
            return
        if not result.papers:
            await interaction.edit_original_response(
                content="No papers were retrieved from the configured sources for that query."
            )
            return
        embeds = paper_search_embeds(result)
        await interaction.edit_original_response(
            content=discovery_warning_text(result),
            embeds=embeds,
        )

    tree.add_command(
        app_commands.Command(
            name="paper",
            description="Find papers across configured scholarly sources.",
            callback=paper,
        )
    )
