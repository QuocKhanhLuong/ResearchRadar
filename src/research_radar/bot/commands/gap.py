"""Thin Discord slash-command adapter for candidate research gaps."""

from __future__ import annotations

import logging
from typing import Literal, Protocol

import discord
from discord import app_commands

from research_radar.bot.interactions import safe_defer
from research_radar.gap.service import GapAnalysisResult
from research_radar.models.gap import CandidateGap, CriticReview

logger = logging.getLogger(__name__)

MAX_DISCORD_CONTENT_CHARS = 2_000
_NO_MENTIONS = discord.AllowedMentions.none()


class GapCommandService(Protocol):
    """The narrow gap-service surface required by Discord handlers."""

    async def analyze_gaps(
        self,
        topic: str,
        count: int = 1,
        gap_type: Literal[
            "explicit", "coverage", "evaluation", "contradiction", "method_transfer"
        ] = "explicit",
    ) -> GapAnalysisResult: ...

    def get_candidate_detail(
        self, candidate_id: str
    ) -> tuple[CandidateGap | None, list[CriticReview]]: ...


def render_gap_embed(candidate: CandidateGap, review: CriticReview | None = None) -> discord.Embed:
    """Render a candidate research gap as a rich, structured Discord embed."""

    status_colors = {
        "preserved": discord.Color.green(),
        "downgraded": discord.Color.gold(),
        "rejected": discord.Color.red(),
        "candidate": discord.Color.blue(),
    }
    color = status_colors.get(candidate.review_status, discord.Color.blue())

    type_title = candidate.gap_type.upper()
    embed = discord.Embed(
        title=f"🔬 Candidate Research Gap [{type_title}]",
        description=f"**Title**: {candidate.title}",
        color=color,
    )

    embed.add_field(
        name="Candidate Research Question",
        value=candidate.research_question,
        inline=False,
    )

    evidence_summary = (
        f"{len(candidate.supporting_papers)} paper(s) | "
        f"{candidate.evidence_count} attributable statement(s)"
    )
    embed.add_field(
        name="Evidence Signal",
        value=evidence_summary,
        inline=True,
    )

    embed.add_field(
        name="Critic Status",
        value=f"**{candidate.review_status.upper()}**",
        inline=True,
    )

    if candidate.confidence is not None:
        embed.add_field(
            name="Confidence",
            value=f"{candidate.confidence:.2f}",
            inline=True,
        )

    if review and review.rationale:
        embed.add_field(
            name="Critic Rationale",
            value=review.rationale,
            inline=False,
        )

    embed.add_field(
        name="Search Scope",
        value=candidate.search_scope,
        inline=False,
    )

    if candidate.caveats:
        caveat_lines = "\n".join(f"• {c}" for c in candidate.caveats[:5])
        embed.add_field(
            name="Caveats",
            value=caveat_lines,
            inline=False,
        )

    if candidate.provenance.supporting_evidence:
        papers_summary: list[str] = []
        for ref in candidate.provenance.supporting_evidence[:5]:
            papers_summary.append(
                f"• **{ref.paper_title}**: *\"{ref.supporting_text}\"*"
            )
        embed.add_field(
            name="Supporting Evidence",
            value="\n".join(papers_summary)[:1024],
            inline=False,
        )

    if candidate.provenance.conflicting_evidence:
        conflict_summary: list[str] = []
        for ref in candidate.provenance.conflicting_evidence[:5]:
            conflict_summary.append(
                f"• **Claim B ({ref.paper_title})**: *\"{ref.supporting_text}\"*"
            )
        embed.add_field(
            name="Conflicting Evidence",
            value="\n".join(conflict_summary)[:1024],
            inline=False,
        )

    embed.set_footer(text=f"Candidate ID: {candidate.id} • ResearchRadar V2E")
    return embed


def register_gap_commands(
    tree: app_commands.CommandTree[discord.Client],
    gap_service: GapCommandService,
) -> None:
    """Register ``/gap`` and ``/gap-show`` slash commands."""

    async def gap_cmd(
        interaction: discord.Interaction,
        topic: str,
        count: int = 1,
        type: Literal[
            "explicit", "coverage", "evaluation", "contradiction", "method_transfer"
        ] = "explicit",
    ) -> None:
        if not await safe_defer(interaction, thinking=True):
            return
        try:
            result = await gap_service.analyze_gaps(topic, count=count, gap_type=type)
            if result.is_insufficient_evidence:
                msg = result.message or "Insufficient evidence."
                await interaction.edit_original_response(
                    content=_truncate(msg, MAX_DISCORD_CONTENT_CHARS),
                    allowed_mentions=_NO_MENTIONS,
                )
                return

            if not result.candidates:
                await interaction.edit_original_response(
                    content="No candidate research gaps were found.",
                    allowed_mentions=_NO_MENTIONS,
                )
                return

            first_candidate = result.candidates[0]
            first_review = result.reviews[0] if result.reviews else None
            embed = render_gap_embed(first_candidate, first_review)

            await interaction.edit_original_response(
                embed=embed,
                allowed_mentions=_NO_MENTIONS,
            )

        except ValueError as error:
            logger.info("Gap command rejected: %s", error)
            await interaction.edit_original_response(
                content=f"Invalid request: {error}",
                allowed_mentions=_NO_MENTIONS,
            )
        except Exception:
            logger.exception("Gap command execution failed.")
            await interaction.edit_original_response(
                content="The gap engine is temporarily unavailable. Please try again later.",
                allowed_mentions=_NO_MENTIONS,
            )

    async def gap_show_cmd(
        interaction: discord.Interaction, candidate_id: str
    ) -> None:
        if not await safe_defer(interaction, thinking=True):
            return
        try:
            candidate, reviews = gap_service.get_candidate_detail(candidate_id.strip())
            if candidate is None:
                await interaction.edit_original_response(
                    content=f"Candidate gap with ID '{candidate_id}' was not found.",
                    allowed_mentions=_NO_MENTIONS,
                )
                return

            latest_review = reviews[-1] if reviews else None
            embed = render_gap_embed(candidate, latest_review)
            await interaction.edit_original_response(
                embed=embed,
                allowed_mentions=_NO_MENTIONS,
            )

        except Exception:
            logger.exception("Gap show command execution failed.")
            await interaction.edit_original_response(
                content="Failed to retrieve candidate gap detail.",
                allowed_mentions=_NO_MENTIONS,
            )

    tree.add_command(
        app_commands.Command(
            name="gap",
            description="Mine evidence-backed candidate research gaps for a topic.",
            callback=gap_cmd,
        )
    )

    tree.add_command(
        app_commands.Command(
            name="gap-show",
            description="Inspect details and audit trail for a candidate gap ID.",
            callback=gap_show_cmd,
        )
    )


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: max(1, limit - 1)].rstrip()}…"
