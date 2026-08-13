"""Pure Discord presentation helpers for normalized research results."""

from __future__ import annotations

import discord

from research_radar.research.service import SearchResult

MAX_EMBEDS_PER_RESPONSE = 10
MAX_EMBED_TITLE_CHARS = 256
MAX_EMBED_DESCRIPTION_CHARS = 300
MAX_AUTHOR_TEXT_CHARS = 150
MAX_DISCORD_URL_CHARS = 2_000


def paper_search_embeds(result: SearchResult) -> list[discord.Embed]:
    """Render bounded normalized results within Discord's aggregate embed budget."""

    embeds: list[discord.Embed] = []
    for index, ranked in enumerate(result.papers[:MAX_EMBEDS_PER_RESPONSE], start=1):
        paper = ranked.paper
        year = paper.publication_year if paper.publication_year is not None else "Unknown"
        authors = _truncate(
            ", ".join(paper.authors) if paper.authors else "Unknown",
            MAX_AUTHOR_TEXT_CHARS,
        )
        link = _safe_discord_url(paper.canonical_link)
        lines = [
            f"**Authors:** {authors}",
            f"**Year:** {year}",
            f"**Venue:** {_truncate(paper.venue or 'Unknown', 80)}",
        ]
        if paper.citation_count is not None:
            lines.append(f"**Citations:** {paper.citation_count}")
        if link:
            lines.append(f"[Open paper]({link})")
        title_prefix = f"{index}. "
        title = _truncate(paper.title, MAX_EMBED_TITLE_CHARS - len(title_prefix))
        embeds.append(
            discord.Embed(
                title=f"{title_prefix}{title}",
                description=_truncate("\n".join(lines), MAX_EMBED_DESCRIPTION_CHARS),
                url=link,
            )
        )
    return embeds


def discovery_warning_text(result: SearchResult) -> str | None:
    """Join safe partial-source warnings for a compact Discord follow-up."""

    return " ".join(result.warnings) if result.warnings else None


def _safe_discord_url(value: str | None) -> str | None:
    """Avoid invalid oversized URLs originating from an external provider."""

    if value and len(value) <= MAX_DISCORD_URL_CHARS and value.startswith(("https://", "http://")):
        return value
    return None


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: max(1, limit - 1)].rstrip()}…"
