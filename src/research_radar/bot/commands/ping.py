"""Health-check slash command."""

from __future__ import annotations

import discord
from discord import app_commands

PING_RESPONSE = "ResearchRadar is online."


async def ping(interaction: discord.Interaction) -> None:
    """Confirm that the Discord application is responsive."""

    await interaction.response.send_message(PING_RESPONSE)


def register_ping_command(tree: app_commands.CommandTree[discord.Client]) -> None:
    """Add the health-check command to an application's command tree."""

    tree.add_command(
        app_commands.Command(
            name="ping",
            description="Check whether ResearchRadar is online.",
            callback=ping,
        )
    )
