"""The single embedding and semantic-index abstraction for the application.

Pinecone is a DERIVED index. Canonical research evidence lives in SQLite and
every semantic hit must be resolved back into SQLite before it is used.

There is exactly one ``SemanticIndex`` protocol and exactly one
``EmbeddingProvider`` protocol. Do not add ``VectorStore``, ``SemanticStore`` or
``EmbeddingIndex`` variants alongside them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from research_radar.errors import ResearchRadarError

EntityType = Literal["paper", "papercard"]

PAPER_SCHEMA_VERSION = "paper-v1"
PAPERCARD_SCHEMA_VERSION = "papercard-v1"

SCHEMA_VERSIONS: dict[str, str] = {
    "paper": PAPER_SCHEMA_VERSION,
    "papercard": PAPERCARD_SCHEMA_VERSION,
}


class SemanticIndexError(ResearchRadarError):
    """Raised when the derived semantic index cannot serve a request."""


class EmbeddingError(ResearchRadarError):
    """Raised when text cannot be embedded."""


class SemanticRecord(BaseModel):
    """One derived vector plus the compact metadata allowed in the index."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(min_length=1)
    entity_type: EntityType
    paper_id: str = Field(min_length=1)
    vector: list[float] = Field(min_length=1)
    publication_year: int | None = None
    embedding_schema_version: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)

    def metadata(self) -> dict[str, str | int]:
        """Return the compact metadata payload written to the index."""

        payload: dict[str, str | int] = {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "paper_id": self.paper_id,
            "embedding_schema_version": self.embedding_schema_version,
            "embedding_model": self.embedding_model,
        }
        if self.publication_year is not None:
            payload["publication_year"] = self.publication_year
        return payload


class SemanticHit(BaseModel):
    """One candidate returned by the index. Never treated as evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: str = Field(min_length=1)
    entity_type: EntityType
    paper_id: str = Field(min_length=1)
    score: float


class SemanticIndexStatus(BaseModel):
    """Compact, non-sensitive health summary for the semantic index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str
    available: bool
    detail: str | None = None


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turn bounded text into deterministic dense vectors."""

    model_id: str

    @property
    def dimension(self) -> int:
        """Return the vector dimension produced by this provider."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, preserving input order."""


@runtime_checkable
class SemanticIndex(Protocol):
    """A derived approximate-nearest-neighbour index over research entities."""

    backend: str

    @property
    def available(self) -> bool:
        """Return whether semantic retrieval can currently be attempted."""

    def upsert(self, records: Sequence[SemanticRecord]) -> int:
        """Idempotently write records and return how many were accepted."""

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 10,
        entity_type: EntityType | None = None,
    ) -> list[SemanticHit]:
        """Return bounded candidates, or an empty list when unavailable."""

    def delete(self, entity_ids: Sequence[str]) -> int:
        """Remove entities from the derived index and return the count."""

    def status(self) -> SemanticIndexStatus:
        """Return a compact availability summary without leaking credentials."""


def entity_vector_id(entity_type: EntityType, paper_id: str) -> str:
    """Return the deterministic vector ID, which makes upserts idempotent."""

    return f"{entity_type}:{paper_id}"
