"""Owner-scoped slash commands for inspecting personal user memory.

/memory-status reports sanitized backend health and /memory-search runs a
bounded fact search; every rendered value comes from an explicit whitelist
(fact text, memory class, temporal validity for facts; backend/enabled/
healthy/persistence-path/episode-count for status). When
settings.discord_owner_user_id is configured both commands answer only that
user, refusing everyone else ephemerally before any backend call.

/memory-forget is deliberately NOT implemented: docs/spikes/graphiti_compat.md
confirms no safe targeted-deletion API in graphiti-core 0.29.x.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

import discord
from discord import app_commands

from research_radar.bot.interactions import safe_defer
from research_radar.memory.models import MemoryFact, MemoryStatus

try:
    from research_radar.memory.secrets import redact_secrets
except ImportError:  # pragma: no cover - W4 lands in a parallel worktree

    def redact_secrets(text: str) -> str:
        """Identity fallback until research_radar.memory.secrets ships."""
        return text


logger = logging.getLogger(__name__)

OWNER_REFUSAL_MESSAGE = "Personal memory commands are restricted to the bot owner."
MEMORY_UNAVAILABLE_MESSAGE = "Personal memory is currently unavailable."
MEMORY_DISABLED_MESSAGE = "Personal memory is disabled."

MAX_FACT_TEXT_CHARS = 300
MAX_FACT_LINE_CHARS = 600
MAX_STATUS_PATH_CHARS = 200
MAX_QUERY_TITLE_CHARS = 80
MAX_SEARCH_DESCRIPTION_CHARS = 3400


class MemoryCommandStore(Protocol):
    """Narrow async surface the personal-memory commands need."""

    @property
    def backend_name(self) -> str:
        """Return the stable backend identifier."""

    @property
    def enabled(self) -> bool:
        """Return whether the backend is configured to serve user memory."""

    async def search(self, query: str, *, limit: int = 8) -> list[MemoryFact]:
        """Return bounded advisory facts relevant to the query."""

    async def status(self) -> MemoryStatus:
        """Return sanitized backend health; never credentials or content."""


class MemoryCommandSettings(Protocol):
    """Settings fields the personal-memory commands read."""

    discord_owner_user_id: int | None
    user_memory_max_results: int


def _truncate(value: str, limit: int) -> str:
    """Clip a string to the limit, marking the cut with an ellipsis."""

    if len(value) <= limit:
        return value
    return f"{value[: max(1, limit - 1)].rstrip()}…"


def format_fact_validity(fact: MemoryFact) -> str:
    """Render the temporal-validity span of one fact, or "" when unbounded."""

    parts: list[str] = []
    if fact.valid_at is not None:
        parts.append(f"valid from {fact.valid_at:%Y-%m-%d}")
    if fact.invalid_at is not None:
        parts.append(f"until {fact.invalid_at:%Y-%m-%d}")
    return "; ".join(parts)


def format_fact_line(fact: MemoryFact) -> str:
    """Render one fact as a whitelisted single line safe for display.

    Only the fact text (secret-redacted, truncated), its memory class, and its
    temporal validity appear; source, score, and any backend metadata are
    dropped here so callers cannot leak them by accident.
    """

    label = fact.memory_class.value if fact.memory_class is not None else "unclassified"
    validity = format_fact_validity(fact)
    suffix = f" ({validity})" if validity else ""
    text = _truncate(redact_secrets(fact.fact), MAX_FACT_TEXT_CHARS)
    return _truncate(f"• [{label}] {text}{suffix}", MAX_FACT_LINE_CHARS)


def format_status_lines(status: MemoryStatus) -> list[str]:
    """Render the whitelisted status fields as ``label: value`` lines."""

    lines = [
        f"backend: {status.backend}",
        f"enabled: {'yes' if status.enabled else 'no'}",
        f"healthy: {'yes' if status.healthy else 'no'}",
    ]
    if status.persistence_path is not None:
        lines.append(f"path: {_truncate(status.persistence_path, MAX_STATUS_PATH_CHARS)}")
    if status.episode_count is not None:
        lines.append(f"episodes: {status.episode_count}")
    return lines


def render_memory_status_embed(status: MemoryStatus) -> discord.Embed:
    """Render sanitized backend health; never detail text or credentials."""

    if not status.enabled:
        description = MEMORY_DISABLED_MESSAGE
        color = discord.Colour.greyple()
    elif not status.healthy:
        description = MEMORY_UNAVAILABLE_MESSAGE
        color = discord.Colour.orange()
    else:
        description = "Personal memory is available."
        color = discord.Colour.green()
    embed = discord.Embed(title="Personal memory status", description=description, color=color)
    for line in format_status_lines(status):
        label, _, value = line.partition(": ")
        name = label.capitalize()
        if name == "Path":
            name = "Persistence path"
        embed.add_field(name=name, value=value, inline=True)
    return embed


def render_memory_search_embed(query: str, facts: Sequence[MemoryFact]) -> discord.Embed:
    """Render matching facts under a hard total-length cap for Discord."""

    title = f"Personal memory search: {_truncate(query, MAX_QUERY_TITLE_CHARS)}"
    embed = discord.Embed(title=title)
    if not facts:
        embed.description = "No matching facts."
        return embed
    lines: list[str] = []
    used = 0
    omitted = 0
    for fact in facts:
        line = format_fact_line(fact)
        if used + len(line) + 1 > MAX_SEARCH_DESCRIPTION_CHARS:
            omitted += 1
            continue
        lines.append(line)
        used += len(line) + 1
    if omitted:
        lines.append(f"(+{omitted} more omitted)")
    embed.description = "\n".join(lines)
    embed.set_footer(text="Advisory personal context — not scientific evidence")
    return embed


def _is_owner(interaction: discord.Interaction, settings: MemoryCommandSettings) -> bool:
    """Return True unless an owner is configured and the caller differs."""

    owner_id = getattr(settings, "discord_owner_user_id", None)
    return owner_id is None or interaction.user.id == owner_id


def register_memory_commands(
    tree: app_commands.CommandTree[discord.Client],
    memory_service: MemoryCommandStore,
    settings: MemoryCommandSettings,
) -> None:
    """Register /memory-status and /memory-search on the command tree."""

    @tree.command(
        name="memory-status",
        description="Show personal user-memory backend health.",
    )
    async def memory_status_cmd(interaction: discord.Interaction) -> None:
        if not _is_owner(interaction, settings):
            await interaction.response.send_message(OWNER_REFUSAL_MESSAGE, ephemeral=True)
            return
        if not await safe_defer(interaction, thinking=True, ephemeral=True):
            return
        try:
            status = await memory_service.status()
        except Exception:
            logger.exception("Unhandled error during /memory-status execution.")
            await interaction.followup.send(content=MEMORY_UNAVAILABLE_MESSAGE, ephemeral=True)
            return
        await interaction.followup.send(embed=render_memory_status_embed(status), ephemeral=True)

    @tree.command(
        name="memory-search",
        description="Search stored personal-memory facts (owner only).",
    )
    @app_commands.describe(query="Text to look for among stored personal-memory facts")
    async def memory_search_cmd(
        interaction: discord.Interaction,
        query: app_commands.Range[str, 1, 200],
    ) -> None:
        if not _is_owner(interaction, settings):
            await interaction.response.send_message(OWNER_REFUSAL_MESSAGE, ephemeral=True)
            return
        if not await safe_defer(interaction, thinking=True, ephemeral=True):
            return
        limit = int(getattr(settings, "user_memory_max_results", 8))
        try:
            facts = await memory_service.search(query, limit=limit)
        except Exception:
            logger.exception("Unhandled error during /memory-search execution.")
            await interaction.followup.send(content=MEMORY_UNAVAILABLE_MESSAGE, ephemeral=True)
            return
        await interaction.followup.send(
            embed=render_memory_search_embed(query, facts), ephemeral=True
        )
