"""Normalized scholarly-paper model shared by every provider."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Paper(BaseModel):
    """A provider-neutral scholarly-paper record.

    `source` identifies the provider that first produced this record. Additional
    identifiers preserve provenance without exposing an upstream response shape.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    citation_count: int | None = Field(default=None, ge=0)
    source: str = Field(min_length=1)
    external_ids: dict[str, str] = Field(default_factory=dict)

    @field_validator("authors", mode="before")
    @classmethod
    def normalize_authors(cls, value: object) -> list[str]:
        """Remove empty author names while preserving their source order."""

        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("authors must be a list of names")
        return [str(author).strip() for author in value if str(author).strip()]

    @field_validator("external_ids", mode="before")
    @classmethod
    def normalize_external_ids(cls, value: object) -> dict[str, str]:
        """Keep only non-empty string identifier pairs from provider adapters."""

        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("external_ids must be a mapping")
        return {
            str(key).strip().lower(): str(identifier).strip()
            for key, identifier in value.items()
            if str(key).strip() and str(identifier).strip()
        }

    @property
    def canonical_link(self) -> str | None:
        """Prefer a DOI link, falling back to the normalized provider URL."""

        if self.doi:
            return f"https://doi.org/{self.doi.removeprefix('https://doi.org/')}"
        return self.url
