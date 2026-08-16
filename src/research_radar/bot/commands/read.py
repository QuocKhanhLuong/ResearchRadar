"""Thin Discord slash-command adapter for structured paper reading."""

from __future__ import annotations

import logging
from typing import Protocol

import discord
from discord import app_commands

from research_radar.bot.interactions import safe_defer
from research_radar.errors import (
    LLMResponseError,
    LLMUnavailableError,
    PaperNotFoundError,
    PaperParseError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

MAX_EMBED_TITLE_CHARS = 256
MAX_EMBED_DESCRIPTION_CHARS = 300
MAX_EMBED_FIELD_CHARS = 700
MAX_DISCORD_URL_CHARS = 2_000
_NO_MENTIONS = discord.AllowedMentions.none()


class ReaderCommandService(Protocol):
    """The narrow asynchronous reader surface required by the Discord adapter."""

    async def read_url(self, url: str) -> object:
        """Read, analyze, and persist a paper available at a direct PDF URL."""


def register_read_command(
    tree: app_commands.CommandTree[discord.Client],
    reader_service: ReaderCommandService,
) -> None:
    """Register ``/read`` without coupling Discord to reader implementation types."""

    async def read(
        interaction: discord.Interaction,
        url: app_commands.Range[str, 1, 2_000],
    ) -> None:
        if not await safe_defer(interaction, thinking=True):
            return
        try:
            result = await reader_service.read_url(url)
        except PaperNotFoundError:
            await _error(
                interaction,
                "I couldn't resolve that paper. Please provide a direct public PDF URL.",
            )
            return
        except PaperParseError:
            await _error(
                interaction,
                "I couldn't extract readable text from that PDF. "
                "Please try another public PDF URL.",
            )
            return
        except LLMUnavailableError:
            await _error(
                interaction,
                "Paper text was extracted, but structured analysis is unavailable. "
                "Configure an LLM provider and try again.",
            )
            return
        except LLMResponseError:
            logger.exception("Paper reader received an invalid structured LLM response.")
            await _error(
                interaction,
                "Paper analysis returned an invalid response. Please try again later.",
            )
            return
        except ProviderUnavailableError:
            await _error(
                interaction,
                "The paper source is temporarily unavailable. Please try again later.",
            )
            return
        except ValueError:
            await _error(
                interaction,
                "That paper URL could not be processed. Please provide a public PDF URL.",
            )
            return
        except Exception:
            logger.exception("Paper reading workflow failed.")
            await _error(
                interaction,
                "I couldn't read that paper right now. Please try again later.",
            )
            return

        await interaction.edit_original_response(
            content=None,
            embed=_read_result_embed(result),
            allowed_mentions=_NO_MENTIONS,
        )

    tree.add_command(
        app_commands.Command(
            name="read",
            description="Extract and analyze a direct public PDF URL.",
            callback=read,
        )
    )


async def _error(interaction: discord.Interaction, content: str) -> None:
    """Replace a deferred response with a safe, human-readable failure."""

    await interaction.edit_original_response(content=content, allowed_mentions=_NO_MENTIONS)


def _read_result_embed(result: object) -> discord.Embed:
    """Render a bounded PaperCard-shaped result without importing reader/storage types."""

    paper = getattr(result, "paper", None)
    card = getattr(result, "card", None)
    title = _string_value(paper, "title", "Paper analysis")
    link = _safe_url(_paper_link(paper))
    embed = discord.Embed(
        title=_truncate(f"Read: {title}", MAX_EMBED_TITLE_CHARS),
        description=_truncate(
            "Structured analysis from extracted paper text. "
            "Evidence locations are shown when available.",
            MAX_EMBED_DESCRIPTION_CHARS,
        ),
        url=link,
    )
    embed.add_field(
        name="Problem",
        value=_field_value(getattr(card, "problem", None)),
        inline=False,
    )
    embed.add_field(
        name="Main contributions",
        value=_field_value(getattr(card, "contributions", None)),
        inline=False,
    )
    embed.add_field(
        name="Methods",
        value=_field_value(getattr(card, "methods", None)),
        inline=False,
    )
    embed.add_field(
        name="Datasets",
        value=_field_value(getattr(card, "datasets", None)),
        inline=False,
    )
    embed.add_field(
        name="Main claims",
        value=_claim_field_value(getattr(card, "main_claims", None)),
        inline=False,
    )
    embed.add_field(
        name="Limitations",
        value=_field_value(getattr(card, "limitations", None)),
        inline=False,
    )
    embed.add_field(
        name="Future work",
        value=_field_value(getattr(card, "future_work", None)),
        inline=False,
    )
    return embed


def _paper_link(paper: object) -> str | None:
    canonical_link = getattr(paper, "canonical_link", None)
    if isinstance(canonical_link, str):
        return canonical_link
    url = getattr(paper, "url", None)
    return url if isinstance(url, str) else None


def _safe_url(value: str | None) -> str | None:
    if value and len(value) <= MAX_DISCORD_URL_CHARS and value.startswith(("https://", "http://")):
        return value
    return None


def _field_value(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return _truncate(value.strip(), MAX_EMBED_FIELD_CHARS)
    if isinstance(value, (list, tuple)):
        entries = [str(item).strip() for item in value if str(item).strip()]
        if entries:
            return _truncate("\n".join(f"• {entry}" for entry in entries), MAX_EMBED_FIELD_CHARS)
    return "Not reported."


def _claim_field_value(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        return "Not reported."

    claims: list[str] = []
    for item in value:
        claim = _string_value(item, "claim", "")
        if not claim:
            continue
        section = _string_value(item, "source_section", "")
        suffix = f" ({section})" if section else ""
        claims.append(f"• {claim}{suffix}")
    return _truncate("\n".join(claims), MAX_EMBED_FIELD_CHARS) if claims else "Not reported."


def _string_value(value: object, attribute: str, fallback: str) -> str:
    candidate = getattr(value, attribute, fallback)
    return candidate.strip() if isinstance(candidate, str) and candidate.strip() else fallback


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: max(1, limit - 1)].rstrip()}…"
