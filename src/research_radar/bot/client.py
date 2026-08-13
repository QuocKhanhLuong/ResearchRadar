"""Discord client factory and lifecycle management.

This module intentionally owns only the Discord boundary. Research services are
constructed elsewhere and may register shutdown hooks with the bot when needed.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable

import discord
from discord import app_commands

from research_radar.bot.commands.ping import register_ping_command
from research_radar.config import Settings

logger = logging.getLogger(__name__)

ShutdownHook = Callable[[], Awaitable[None]]


class ResearchRadarBot(discord.Client):
    """Minimal slash-command Discord client for the single-user application."""

    def __init__(
        self,
        settings: Settings,
        *,
        shutdown_hooks: Iterable[ShutdownHook] = (),
    ) -> None:
        super().__init__(intents=_application_intents())
        self.settings = settings
        self.tree = app_commands.CommandTree(self)
        self._shutdown_hooks = list(shutdown_hooks)
        self._owned_resources_closed = False

        register_ping_command(self.tree)

    async def setup_hook(self) -> None:
        """Synchronize slash commands before connecting to the gateway."""

        await self.sync_application_commands()

    async def sync_application_commands(self) -> list[app_commands.AppCommand]:
        """Sync commands globally or to the configured development guild."""

        guild_id = self.settings.discord_guild_id
        if guild_id is None:
            commands = await self.tree.sync()
            logger.info("Synchronized %d global Discord application command(s).", len(commands))
            return commands

        guild = discord.Object(id=guild_id)
        self.tree.copy_global_to(guild=guild)
        commands = await self.tree.sync(guild=guild)
        logger.info(
            "Synchronized %d Discord application command(s) to development guild %s.",
            len(commands),
            guild_id,
        )
        return commands

    def add_shutdown_hook(self, hook: ShutdownHook) -> None:
        """Register an async cleanup action for a resource owned by the bot process."""

        if self._owned_resources_closed:
            raise RuntimeError("Cannot register a shutdown hook after bot resources are closed.")
        self._shutdown_hooks.append(hook)

    async def close_owned_resources(self) -> None:
        """Run registered cleanup hooks once, without needing a Discord gateway connection."""

        if self._owned_resources_closed:
            return

        self._owned_resources_closed = True
        for hook in reversed(self._shutdown_hooks):
            try:
                await hook()
            except Exception:  # pragma: no cover - defensive isolation for shutdown paths
                logger.exception("Discord shutdown hook failed.")

    async def close(self) -> None:
        """Close owned resources before allowing discord.py to close its client resources."""

        try:
            await self.close_owned_resources()
        finally:
            await super().close()

    async def on_ready(self) -> None:
        """Log a concise ready signal after Discord completes its connection."""

        logger.info("ResearchRadar Discord bot is ready as %s.", self.user)


def create_bot(
    settings: Settings,
    *,
    shutdown_hooks: Iterable[ShutdownHook] = (),
) -> ResearchRadarBot:
    """Construct a bot without requiring a token or a live Discord connection."""

    return ResearchRadarBot(settings, shutdown_hooks=shutdown_hooks)


def run_bot(
    settings: Settings,
    *,
    shutdown_hooks: Iterable[ShutdownHook] = (),
) -> None:
    """Launch the Discord client after explicitly validating its required token."""

    bot = create_bot(settings, shutdown_hooks=shutdown_hooks)
    bot.run(settings.require_discord_token(), log_handler=None)


def _application_intents() -> discord.Intents:
    """Return the minimal gateway intents needed for application-command operation."""

    intents = discord.Intents.none()
    intents.guilds = True
    return intents
