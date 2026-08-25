"""The single artifact-storage abstraction for the whole application.

There is exactly one artifact store protocol. Do not introduce a parallel
document-cache or blob-store abstraction alongside it.
"""

from __future__ import annotations

import hashlib
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from research_radar.errors import ResearchRadarError

ArtifactType = Literal["pdf", "text", "sections"]

ARTIFACT_MIME_TYPES: dict[str, str] = {
    "pdf": "application/pdf",
    "text": "text/plain; charset=utf-8",
    "sections": "application/json",
}

ARTIFACT_SUFFIXES: dict[str, str] = {
    "pdf": ".pdf",
    "text": ".txt",
    "sections": ".sections.json",
}


class ArtifactStoreError(ResearchRadarError):
    """Raised when an artifact cannot be stored or resolved safely."""


class ArtifactRef(BaseModel):
    """A stable, content-addressed handle to one stored artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    artifact_type: ArtifactType
    backend: str = "local"
    object_key: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    byte_size: int = Field(ge=0)
    source_url: str | None = None


@runtime_checkable
class ArtifactStore(Protocol):
    """Store and resolve immutable, content-addressed document artifacts."""

    backend: str

    def put(
        self,
        *,
        paper_id: str,
        content: bytes,
        artifact_type: ArtifactType,
        sha256: str | None = None,
        source_url: str | None = None,
    ) -> ArtifactRef:
        """Store ``content`` idempotently and return its stable reference.

        ``sha256`` addresses the artifact. It defaults to the digest of
        ``content``, but a derived artifact (extracted text, section JSON) is
        addressed by the digest of the *source* document so every artifact for
        one document version shares a key.
        """

    def exists(self, *, paper_id: str, sha256: str, artifact_type: ArtifactType) -> bool:
        """Return whether an artifact is already stored for this identity."""

    def read(self, *, paper_id: str, sha256: str, artifact_type: ArtifactType) -> bytes:
        """Return stored bytes, raising ``ArtifactStoreError`` when absent."""

    def resolve(
        self, *, paper_id: str, sha256: str, artifact_type: ArtifactType
    ) -> ArtifactRef | None:
        """Return the stored reference, or ``None`` when it does not exist."""


def sha256_hex(content: bytes) -> str:
    """Return the lowercase hex SHA256 digest used to address every artifact."""

    return hashlib.sha256(content).hexdigest()
