"""Validated structured research knowledge extracted from a paper."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceClaim(BaseModel):
    """A claim plus the text/location that supports it when known."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim: str = Field(min_length=1)
    source_section: str | None = None
    supporting_text: str | None = None


class StructuredEvidence(BaseModel):
    """An extracted evidence item with explicit status and provenance."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    value: str = Field(min_length=1)
    status: Literal["observed", "explicitly_absent", "unknown"] = "observed"
    source_section: str | None = None
    supporting_text: str | None = None


class PaperCard(BaseModel):
    """A compact, evidence-aware representation of one analyzed paper."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    paper_id: str = Field(min_length=1)
    problem: str | None = None
    motivation: str | None = None
    contributions: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    tasks: list[StructuredEvidence] = Field(default_factory=list)
    modalities: list[StructuredEvidence] = Field(default_factory=list)
    evaluation_conditions: list[StructuredEvidence] = Field(default_factory=list)
    main_claims: list[EvidenceClaim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)
    failure_cases: list[str] = Field(default_factory=list)
