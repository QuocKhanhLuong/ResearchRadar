"""Thin Discord slash-command adapters for the single-user watchlist."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

import discord
from discord import app_commands

from research_radar.bot.interactions import safe_defer

logger = logging.getLogger(__name__)

MAX_DISCORD_CONTENT_CHARS = 2_000
MAX_TOPIC_NAME_CHARS = 120
MAX_TOPIC_QUERY_CHARS = 300
_NO_MENTIONS = discord.AllowedMentions.none()


class WatchCommandService(Protocol):
    """The minimal asynchronous watch-service surface used by Discord."""

    async def add_topic(self, name: str, query: str) -> object:
        """Create a saved watch topic."""

    async def list_topics(self) -> Sequence[object]:
        """Return saved watch topics."""

    async def remove_topic(self, topic_id_or_name: str) -> bool:
        """Remove a topic by stable id or name."""


def register_watch_commands(
    tree: app_commands.CommandTree[discord.Client],
    watch_service: WatchCommandService,
) -> None:
    """Register the presentation-only ``/watch`` command group.

    The injected service owns validation and persistence.  This module only
    acknowledges Discord interactions and turns service results into compact,
    safe messages.
    """

    group = app_commands.Group(
        name="watch",
        description="Manage ResearchRadar paper-monitoring topics.",
    )

    async def add(
        interaction: discord.Interaction,
        name: app_commands.Range[str, 1, 120],
        query: app_commands.Range[str, 1, 300],
    ) -> None:
        if not await safe_defer(interaction, thinking=True):
            return
        try:
            topic = await watch_service.add_topic(name, query)
        except ValueError as error:
            logger.info("Watch topic creation was rejected: %s", error)
            await _edit(interaction, content=f"I couldn't add that watch topic: {error}")
            return
        except Exception:
            logger.exception("Watch topic creation failed.")
            await _edit(
                interaction,
                content="The watchlist is temporarily unavailable. Please try again later.",
            )
            return

        topic_name = _topic_value(topic, "name", name)
        topic_query = _topic_value(topic, "query", query)
        await _edit(
            interaction,
            content=(
                f"Now watching **{_truncate(topic_name, MAX_TOPIC_NAME_CHARS)}**\n"
                f"Query: {_truncate(topic_query, MAX_TOPIC_QUERY_CHARS)}"
            ),
        )

    async def list_(interaction: discord.Interaction) -> None:
        if not await safe_defer(interaction, thinking=True):
            return
        try:
            topics = await watch_service.list_topics()
        except Exception:
            logger.exception("Watchlist retrieval failed.")
            await _edit(
                interaction,
                content="The watchlist is temporarily unavailable. Please try again later.",
            )
            return

        await _edit(interaction, content=_render_topic_list(topics))

    async def remove(
        interaction: discord.Interaction,
        topic: app_commands.Range[str, 1, 255],
    ) -> None:
        if not await safe_defer(interaction, thinking=True):
            return
        try:
            removed = await watch_service.remove_topic(topic)
        except ValueError as error:
            logger.info("Watch topic removal was rejected: %s", error)
            await _edit(interaction, content=f"I couldn't remove that watch topic: {error}")
            return
        except Exception:
            logger.exception("Watch topic removal failed.")
            await _edit(
                interaction,
                content="The watchlist is temporarily unavailable. Please try again later.",
            )
            return

        if not removed:
            await _edit(interaction, content="No matching watch topic was found.")
            return
        await _edit(interaction, content="Removed the watch topic.")

    group.add_command(
        app_commands.Command(
            name="add",
            description="Add a paper-monitoring topic.",
            callback=add,
        )
    )
    group.add_command(
        app_commands.Command(
            name="list",
            description="Show saved paper-monitoring topics.",
            callback=list_,
        )
    )
    group.add_command(
        app_commands.Command(
            name="remove",
            description="Remove a topic by its name or id.",
            callback=remove,
        )
    )
    tree.add_command(group)


async def _edit(interaction: discord.Interaction, *, content: str) -> None:
    """Safely replace a deferred response without allowing output mentions."""

    await interaction.edit_original_response(
        content=_truncate(content, MAX_DISCORD_CONTENT_CHARS),
        allowed_mentions=_NO_MENTIONS,
    )


def _render_topic_list(topics: Sequence[object]) -> str:
    """Render bounded, duck-typed watch-topic records for a Discord message."""

    if not topics:
        return "Your watchlist is empty. Use `/watch add` to create a topic."

    lines = ["**Research watchlist**"]
    for index, topic in enumerate(topics, start=1):
        name = _truncate(_topic_value(topic, "name", "Unnamed topic"), MAX_TOPIC_NAME_CHARS)
        query = _truncate(_topic_value(topic, "query", ""), MAX_TOPIC_QUERY_CHARS)
        status = "enabled" if bool(getattr(topic, "enabled", True)) else "paused"
        topic_id = _truncate(_topic_value(topic, "id", ""), 80)
        last_scan = _format_last_scan(getattr(topic, "last_scan_at", None))
        lines.extend(
            (
                f"{index}. **{name}** ({status})",
                f"   Query: {query or 'Not available'}",
                f"   Last scan: {last_scan}",
                f"   ID: `{topic_id or 'not available'}`",
            )
        )
        rendered = "\n".join(lines)
        if len(rendered) > MAX_DISCORD_CONTENT_CHARS:
            return _truncate(rendered, MAX_DISCORD_CONTENT_CHARS)
    return "\n".join(lines)


def _topic_value(topic: object, name: str, fallback: str) -> str:
    """Read a compact string field from a service result without coupling types."""

    value = getattr(topic, name, fallback)
    return value if isinstance(value, str) and value.strip() else fallback


def _format_last_scan(value: object) -> str:
    if value is None:
        return "Not scanned yet"
    if isinstance(value, datetime):
        return value.isoformat(timespec="minutes")
    return "Unknown"


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: max(1, limit - 1)].rstrip()}…"
