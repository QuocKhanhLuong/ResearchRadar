"""Extracted, bounded paper text ready for deterministic or LLM analysis."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PaperDocument(BaseModel):
    """Text extracted from one PDF, with heuristically detected sections."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str
    sections: dict[str, str] = Field(default_factory=dict)
    full_text: str
    source_url: str | None = None
    extraction_warning: str | None = None

    @property
    def section_names(self) -> set[str]:
        """Return normalized section labels for evidence validation."""

        return {name.casefold() for name in self.sections}
