"""Pure Discord presentation helpers for normalized research results."""

from __future__ import annotations

import discord

from research_radar.research.service import SearchResult


def paper_search_embeds(result: SearchResult) -> list[discord.Embed]:
    """Render bounded normalized discovery results without touching providers."""

    embeds: list[discord.Embed] = []
    for index, ranked in enumerate(result.papers, start=1):
        paper = ranked.paper
        year = paper.publication_year if paper.publication_year is not None else "Unknown"
        lines = [
            f"**Authors:** {', '.join(paper.authors) if paper.authors else 'Unknown'}",
            f"**Year:** {year}",
            f"**Venue:** {paper.venue or 'Unknown'}",
        ]
        if paper.citation_count is not None:
            lines.append(f"**Citations:** {paper.citation_count}")
        if paper.canonical_link:
            lines.append(f"[Open paper]({paper.canonical_link})")
        embeds.append(
            discord.Embed(
                title=f"{index}. {paper.title}",
                description="\n".join(lines),
                url=paper.canonical_link,
            )
        )
    return embeds


def discovery_warning_text(result: SearchResult) -> str | None:
    """Join safe partial-source warnings for a compact Discord follow-up."""

    return " ".join(result.warnings) if result.warnings else None
