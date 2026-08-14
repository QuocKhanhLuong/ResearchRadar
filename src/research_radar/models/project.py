"""Domain models for ResearchRadar project memory."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Project(BaseModel):
    """A research project grouping goals, constraints, hypotheses, papers, and gaps."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str | None = None
    goal: str | None = None
    keywords: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    rejected_ideas: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ProjectPaperLink(BaseModel):
    """Relationship link between a project and a canonical paper."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    project_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    relation: str = Field(default="relevant")
    note: str | None = None
    created_at: datetime


class ProjectGapLink(BaseModel):
    """Relationship link between a project and a CandidateGap."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    project_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    status: str = Field(default="active")
    created_at: datetime
