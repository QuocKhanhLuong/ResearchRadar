"""Discord slash command handler for ResearchRadar memory Q&A (/ask V1)."""

from __future__ import annotations

from typing import Protocol

import discord
from discord import app_commands

from research_radar.research.ask import AskResponse


class AskCommandRegistrationService(Protocol):
    """Narrow interface required by Discord /ask command handler."""

    async def ask(
        self,
        question: str,
        *,
        project_id_or_name: str | None = None,
        max_evidence: int = 10,
    ) -> AskResponse: ...


def render_ask_embed(question: str, response: AskResponse) -> discord.Embed:
    """Render an /ask answer as a Discord Embed."""

    color = discord.Color.green() if response.is_sufficient_evidence else discord.Color.orange()
    embed = discord.Embed(
        title=f"❓ Q: {question[:200]}",
        description=response.answer[:4000],
        color=color,
    )

    if response.referenced_paper_ids:
        paper_text = ", ".join(f"`{pid}`" for pid in response.referenced_paper_ids[:10])
        embed.add_field(name="Referenced Papers", value=paper_text, inline=False)

    if response.referenced_gap_ids:
        gap_text = ", ".join(f"`{gid}`" for gid in response.referenced_gap_ids[:10])
        embed.add_field(name="Referenced Gaps", value=gap_text, inline=False)

    embed.set_footer(text="ResearchRadar Memory V1 • Bounded Evidence Q&A")
    return embed


def register_ask_command(
    tree: app_commands.CommandTree[discord.Client],
    service: AskCommandRegistrationService,
) -> None:
    """Register the /ask slash command with Discord CommandTree."""

    @tree.command(
        name="ask",
        description="Ask a research question bounded strictly to stored ResearchRadar memory.",
    )
    @app_commands.describe(
        question="Research question to ask",
        project="Optional project name or ID to scope context",
    )
    async def ask_cmd(
        interaction: discord.Interaction,
        question: str,
        project: str | None = None,
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            res = await service.ask(question, project_id_or_name=project)
            embed = render_ask_embed(question, res)
            await interaction.followup.send(embed=embed)
        except Exception as exc:
            await interaction.followup.send(content=f"Error executing /ask: {exc}")
