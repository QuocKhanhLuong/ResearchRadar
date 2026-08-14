"""Discord slash command handlers for ResearchRadar project memory management."""

from __future__ import annotations

from typing import Protocol

import discord
from discord import app_commands

from research_radar.models.project import Project, ProjectGapLink, ProjectPaperLink


class ProjectCommandService(Protocol):
    """The narrow repository/service surface required by Discord project handlers."""

    def create_project(
        self,
        name: str,
        *,
        description: str | None = None,
        goal: str | None = None,
        keywords: list[str] | None = None,
        constraints: list[str] | None = None,
        hypotheses: list[str] | None = None,
        rejected_ideas: list[str] | None = None,
    ) -> Project: ...

    def get_project(self, project_id_or_name: str) -> Project | None: ...

    def list_projects(self) -> list[Project]: ...

    def add_paper_to_project(
        self,
        project_id: str,
        paper_id: str,
        *,
        relation: str = "relevant",
        note: str | None = None,
    ) -> ProjectPaperLink: ...

    def list_project_papers(self, project_id: str) -> list[ProjectPaperLink]: ...

    def add_gap_to_project(
        self,
        project_id: str,
        candidate_id: str,
        *,
        status: str = "active",
    ) -> ProjectGapLink: ...

    def list_project_gaps(self, project_id: str) -> list[ProjectGapLink]: ...


def render_project_embed(
    project: Project,
    papers: list[ProjectPaperLink] | None = None,
    gaps: list[ProjectGapLink] | None = None,
) -> discord.Embed:
    """Render a research project as a Discord Embed."""

    embed = discord.Embed(
        title=f"📁 Project: {project.name}",
        description=project.description or "No description provided.",
        color=discord.Color.teal(),
    )

    if project.goal:
        embed.add_field(name="Goal", value=project.goal, inline=False)

    if project.keywords:
        embed.add_field(name="Keywords", value=", ".join(project.keywords), inline=True)

    if project.constraints:
        embed.add_field(
            name="Constraints",
            value="\n".join(f"• {c}" for c in project.constraints),
            inline=False,
        )

    if project.hypotheses:
        embed.add_field(
            name="Hypotheses",
            value="\n".join(f"• {h}" for h in project.hypotheses),
            inline=False,
        )

    if project.rejected_ideas:
        embed.add_field(
            name="Rejected Ideas",
            value="\n".join(f"• {r}" for r in project.rejected_ideas),
            inline=False,
        )

    if papers:
        paper_lines = [f"• `{p.paper_id}` ({p.relation})" for p in papers[:10]]
        embed.add_field(name="Linked Papers", value="\n".join(paper_lines), inline=False)

    if gaps:
        gap_lines = [f"• `{g.candidate_id}` ({g.status})" for g in gaps[:10]]
        embed.add_field(name="Linked Gaps", value="\n".join(gap_lines), inline=False)

    embed.set_footer(text=f"Project ID: {project.id} • ResearchRadar Memory V1")
    return embed


def register_project_commands(
    tree: app_commands.CommandTree[discord.Client],
    service: ProjectCommandService,
) -> None:
    """Register project slash commands with the Discord command tree."""

    @tree.command(name="project-create", description="Create a new research project scope.")
    @app_commands.describe(
        name="Name of the research project",
        goal="Primary objective or goal",
        keywords="Comma-separated keywords (e.g. MRI,reconstruction,lesion)",
    )
    async def project_create_cmd(
        interaction: discord.Interaction,
        name: str,
        goal: str | None = None,
        keywords: str | None = None,
    ) -> None:
        await interaction.response.defer(thinking=True)
        kw_list = [k.strip() for k in keywords.split(",")] if keywords else []
        try:
            proj = service.create_project(name=name, goal=goal, keywords=kw_list)
            embed = render_project_embed(proj)
            await interaction.followup.send(
                content=f"Created project '{proj.name}'!", embed=embed
            )
        except Exception:
            await interaction.followup.send(
                content="Could not create project. Please check the provided name and arguments."
            )

    @tree.command(name="project-list", description="List all research projects.")
    async def project_list_cmd(interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)
        projects = service.list_projects()
        if not projects:
            await interaction.followup.send(content="No research projects found.")
            return

        embed = discord.Embed(
            title="📁 Research Projects",
            description=f"Total {len(projects)} stored project(s).",
            color=discord.Color.teal(),
        )
        for proj in projects[:25]:
            val = f"ID: `{proj.id}`"
            if proj.goal:
                val += f" | Goal: {proj.goal[:60]}"
            embed.add_field(name=proj.name, value=val, inline=False)

        await interaction.followup.send(embed=embed)

    @tree.command(name="project-show", description="Show details of a research project.")
    @app_commands.describe(project="Name or ID of the research project")
    async def project_show_cmd(interaction: discord.Interaction, project: str) -> None:
        await interaction.response.defer(thinking=True)
        proj = service.get_project(project)
        if proj is None:
            await interaction.followup.send(content=f"Project '{project}' not found.")
            return

        papers = service.list_project_papers(proj.id)
        gaps = service.list_project_gaps(proj.id)
        embed = render_project_embed(proj, papers=papers, gaps=gaps)
        await interaction.followup.send(embed=embed)

    @tree.command(name="project-add-paper", description="Link a paper to a research project.")
    @app_commands.describe(
        project="Name or ID of the research project",
        paper_id="ID of the stored paper",
        relation="Relation (e.g. seed, relevant, supporting, conflicting)",
    )
    async def project_add_paper_cmd(
        interaction: discord.Interaction,
        project: str,
        paper_id: str,
        relation: str = "relevant",
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            link = service.add_paper_to_project(
                project_id=project, paper_id=paper_id, relation=relation
            )
            await interaction.followup.send(
                content=f"Linked paper `{link.paper_id}` to project as '{link.relation}'."
            )
        except Exception:
            await interaction.followup.send(
                content="Could not link paper to project. Please verify project and paper IDs."
            )

    @tree.command(name="project-add-gap", description="Link a candidate gap to a research project.")
    @app_commands.describe(
        project="Name or ID of the research project",
        gap_id="ID of the candidate gap",
        status="Status (e.g. active, interesting, rejected, resolved)",
    )
    async def project_add_gap_cmd(
        interaction: discord.Interaction,
        project: str,
        gap_id: str,
        status: str = "active",
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            link = service.add_gap_to_project(
                project_id=project, candidate_id=gap_id, status=status
            )
            await interaction.followup.send(
                content=(
                    f"Linked candidate gap `{link.candidate_id}` to project "
                    f"with status '{link.status}'."
                )
            )
        except Exception:
            await interaction.followup.send(
                content="Could not link gap to project. Please verify project and gap IDs."
            )
