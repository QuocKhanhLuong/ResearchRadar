"""Local filesystem backend for content-addressed document artifacts.

Layout::

    <artifact_root>/papers/<paper_id>/<sha256><suffix>

IMPLEMENTATION OWNER: worker W1. Fill in the method bodies below. Do not change
the public signatures, and do not add a second artifact abstraction.
"""

from __future__ import annotations

from pathlib import Path

from research_radar.artifacts.base import (
    ArtifactRef,
    ArtifactStoreError,
    ArtifactType,
)


class LocalArtifactStore:
    """Store artifacts on local disk, atomically and idempotently."""

    backend = "local"

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).expanduser().resolve()

    @property
    def root(self) -> Path:
        """Return the resolved artifact root directory."""

        return self._root

    def put(
        self,
        *,
        paper_id: str,
        content: bytes,
        artifact_type: ArtifactType,
        sha256: str | None = None,
        source_url: str | None = None,
    ) -> ArtifactRef:
        """Store ``content`` idempotently and return its stable reference."""

        raise NotImplementedError("W1 owns this implementation")

    def exists(self, *, paper_id: str, sha256: str, artifact_type: ArtifactType) -> bool:
        """Return whether an artifact is already stored for this identity."""

        raise NotImplementedError("W1 owns this implementation")

    def read(self, *, paper_id: str, sha256: str, artifact_type: ArtifactType) -> bytes:
        """Return stored bytes, raising ``ArtifactStoreError`` when absent."""

        raise NotImplementedError("W1 owns this implementation")

    def resolve(
        self, *, paper_id: str, sha256: str, artifact_type: ArtifactType
    ) -> ArtifactRef | None:
        """Return the stored reference, or ``None`` when it does not exist."""

        raise NotImplementedError("W1 owns this implementation")


__all__ = ["ArtifactStoreError", "LocalArtifactStore"]
