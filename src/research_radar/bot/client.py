"""Discord client factory and lifecycle management.

This module intentionally owns only the Discord boundary. Research services are
constructed elsewhere and may register shutdown hooks with the bot when needed.

Gateway intents deliberately exclude the privileged Message Content intent:
Discord documents an exception that still delivers message content without it
for messages that mention the app and for direct messages addressed to the app,
which is exactly the chat-on-mention surface this bot uses. Requesting Message
Content would grant far more than this feature needs, so it stays off.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import discord
from discord import app_commands

from research_radar.bot.commands.ask import AskCommandRegistrationService, register_ask_command
from research_radar.bot.commands.digest import register_digest_command
from research_radar.bot.commands.gap import register_gap_commands
from research_radar.bot.commands.ingest import (
    IngestCommandRegistrationService,
    register_ingest_command,
)
from research_radar.bot.commands.paper import register_paper_command
from research_radar.bot.commands.ping import register_ping_command
from research_radar.bot.commands.project import ProjectCommandService, register_project_commands
from research_radar.bot.commands.read import register_read_command
from research_radar.bot.commands.watch import register_watch_commands
from research_radar.bot.interactions import on_app_command_error
from research_radar.bot.mention import MentionPolicy
from research_radar.config import Settings
from research_radar.research.service import ResearchService

logger = logging.getLogger(__name__)

ShutdownHook = Callable[[], Awaitable[None]]
StartupHook = Callable[[], Awaitable[None]]


class WatchCommandRegistrationService(Protocol):
    """Minimum async surface required to register the Discord watch commands."""

    async def add_topic(self, name: str, query: str) -> object: ...

    async def list_topics(self) -> list[object]: ...

    async def remove_topic(self, topic_id_or_name: str) -> bool: ...


class ReaderCommandRegistrationService(Protocol):
    """Minimum async surface required to register the Discord reader command."""

    async def read_url(self, url: str) -> object: ...


class DigestCommandRegistrationService(Protocol):
    """Minimum async surface required to register the Discord digest command."""

    async def build_on_demand(self) -> object: ...


class GapCommandRegistrationService(Protocol):
    """Minimum surface required to register the Discord gap commands."""

    async def analyze_gaps(self, topic: str, count: int = 1) -> object: ...

    def get_candidate_detail(self, candidate_id: str) -> tuple[object, list[object]]: ...


class ChatServiceProtocol(Protocol):
    """Minimum async surface required from the research chat service."""

    async def chat(self, request: object) -> object: ...


@dataclass(frozen=True, slots=True)
class MentionChatRequest:
    """ChatRequest-shaped payload handed to the chat service for one mention.

    Mirrors the field names of ``research_radar.chat.models.ChatRequest``
    without importing that module, which is owned by another worker.
    """

    text: str
    discord_user_id: str | None = None
    channel_id: str | None = None
    message_id: str | None = None
    project_hint: str | None = None


_MENTION_USAGE_HINT = "Mention me with your question in the same message and I will answer."
_CHAT_ERROR_REPLY = "Sorry, something went wrong while answering. Please try again shortly."
_EMPTY_RESPONSE_REPLY = "I could not produce an answer for that. Please try rephrasing."
_FAILURE_LOG_MESSAGE = "chat backend did not complete; replying with a safe error"
_CHUNK_LIMIT = 1900
_NO_MENTIONS = discord.AllowedMentions.none()


class ResearchRadarBot(discord.Client):
    """Minimal slash-command Discord client for the single-user application."""

    def __init__(
        self,
        settings: Settings,
        *,
        startup_hooks: Iterable[StartupHook] = (),
        shutdown_hooks: Iterable[ShutdownHook] = (),
        research_service: ResearchService | None = None,
        watch_service: WatchCommandRegistrationService | None = None,
        reader_service: ReaderCommandRegistrationService | None = None,
        digest_service: DigestCommandRegistrationService | None = None,
        gap_service: GapCommandRegistrationService | None = None,
        project_service: ProjectCommandService | None = None,
        ask_service: AskCommandRegistrationService | None = None,
        ingestion_service: IngestCommandRegistrationService | None = None,
        chat_service: ChatServiceProtocol | None = None,
    ) -> None:
        super().__init__(intents=_application_intents(settings))
        self.settings = settings
        self.tree = app_commands.CommandTree(self)
        self.tree.on_error = on_app_command_error
        self._startup_hooks = list(startup_hooks)
        self._shutdown_hooks = list(shutdown_hooks)
        self._owned_resources_closed = False
        self._chat_service = chat_service
        self._mention_policy = MentionPolicy(settings)

        register_ping_command(self.tree)
        if research_service is not None:
            register_paper_command(self.tree, research_service)
        if watch_service is not None:
            register_watch_commands(self.tree, watch_service)
        if reader_service is not None:
            register_read_command(self.tree, reader_service)
        if digest_service is not None:
            register_digest_command(self.tree, digest_service)
        if gap_service is not None:
            register_gap_commands(self.tree, gap_service)
        if project_service is not None:
            register_project_commands(self.tree, project_service)
        if ask_service is not None:
            register_ask_command(self.tree, ask_service)
        if ingestion_service is not None:
            register_ingest_command(self.tree, ingestion_service)
        if chat_service is not None:
            self.on_message = self._on_message  # type: ignore[method-assign]

    async def _on_message(self, message: discord.Message) -> None:
        """Admit one mention-driven message and relay accepted text to the chat service.

        This handler contains no research logic: admit, hint on empty text,
        call the chat service once inside a typing indicator, then chunk and
        send the reply. Failures are logged sanitized (exception type plus a
        fixed message) and answered with one short safe reply.
        """

        if self.user is None:
            return
        admission = self._mention_policy.admit(message, bot_user_id=self.user.id)
        if not admission.accepted:
            return
        if admission.text == "":
            await message.reply(content=_MENTION_USAGE_HINT, allowed_mentions=_NO_MENTIONS)
            return

        request = MentionChatRequest(
            text=admission.text,
            discord_user_id=str(message.author.id),
            channel_id=str(message.channel.id),
            message_id=str(message.id),
        )
        try:
            async with message.channel.typing():
                response = await self._chat_service.chat(request)
        except Exception as error:
            logger.warning("Chat turn failed (%s): %s", type(error).__name__, _FAILURE_LOG_MESSAGE)
            await message.reply(content=_CHAT_ERROR_REPLY, allowed_mentions=_NO_MENTIONS)
            return

        chunks = _chunk_message_text(str(getattr(response, "text", "") or ""))
        if not chunks:
            await message.reply(content=_EMPTY_RESPONSE_REPLY, allowed_mentions=_NO_MENTIONS)
            return
        for chunk in chunks:
            await message.channel.send(content=chunk, allowed_mentions=_NO_MENTIONS)

    async def setup_hook(self) -> None:
        """Synchronize slash commands before connecting to the gateway."""

        await self.sync_application_commands()
        for hook in self._startup_hooks:
            await hook()

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
    startup_hooks: Iterable[StartupHook] = (),
    shutdown_hooks: Iterable[ShutdownHook] = (),
    research_service: ResearchService | None = None,
    watch_service: WatchCommandRegistrationService | None = None,
    reader_service: ReaderCommandRegistrationService | None = None,
    digest_service: DigestCommandRegistrationService | None = None,
    gap_service: GapCommandRegistrationService | None = None,
    project_service: ProjectCommandService | None = None,
    ask_service: AskCommandRegistrationService | None = None,
    ingestion_service: IngestCommandRegistrationService | None = None,
    chat_service: ChatServiceProtocol | None = None,
) -> ResearchRadarBot:
    """Construct a bot without requiring a token or a live Discord connection."""

    return ResearchRadarBot(
        settings,
        startup_hooks=startup_hooks,
        shutdown_hooks=shutdown_hooks,
        research_service=research_service,
        watch_service=watch_service,
        reader_service=reader_service,
        digest_service=digest_service,
        gap_service=gap_service,
        project_service=project_service,
        ask_service=ask_service,
        ingestion_service=ingestion_service,
        chat_service=chat_service,
    )


def run_bot(
    settings: Settings,
    *,
    startup_hooks: Iterable[StartupHook] = (),
    shutdown_hooks: Iterable[ShutdownHook] = (),
    research_service: ResearchService | None = None,
    watch_service: WatchCommandRegistrationService | None = None,
    reader_service: ReaderCommandRegistrationService | None = None,
    digest_service: DigestCommandRegistrationService | None = None,
    gap_service: GapCommandRegistrationService | None = None,
    project_service: ProjectCommandService | None = None,
    ask_service: AskCommandRegistrationService | None = None,
    ingestion_service: IngestCommandRegistrationService | None = None,
    chat_service: ChatServiceProtocol | None = None,
) -> None:
    """Launch the Discord client after explicitly validating its required token."""

    bot = create_bot(
        settings,
        startup_hooks=startup_hooks,
        shutdown_hooks=shutdown_hooks,
        research_service=research_service,
        watch_service=watch_service,
        reader_service=reader_service,
        digest_service=digest_service,
        gap_service=gap_service,
        project_service=project_service,
        ask_service=ask_service,
        ingestion_service=ingestion_service,
        chat_service=chat_service,
    )
    bot.run(settings.require_discord_token(), log_handler=None)


def _chunk_message_text(text: str, limit: int = _CHUNK_LIMIT) -> list[str]:
    """Split long text into ordered chunks of at most ``limit`` characters.

    Paragraph and newline boundaries are preferred; text without any boundary
    is hard-cut. Empty input yields no chunks.
    """

    normalized = text.strip()
    if not normalized:
        return []
    chunks: list[str] = []
    remaining = normalized
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut <= 0:
            cut = window.rfind("\n")
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _application_intents(settings: Settings) -> discord.Intents:
    """Return the minimal gateway intents for commands plus chat-on-mention.

    Message Content stays off (see the module docstring); ``dm_messages`` is
    requested only when the DM chat setting allows direct-message chats.
    """

    intents = discord.Intents.none()
    intents.guilds = True
    intents.guild_messages = True
    if getattr(settings, "discord_dm_chat", True):
        intents.dm_messages = True
    return intents
