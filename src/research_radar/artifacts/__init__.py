"""Content-addressed document artifact storage.

SQLite stays the canonical structured store; this package owns the bytes.
"""

from research_radar.artifacts.base import (
    ARTIFACT_MIME_TYPES,
    ArtifactRef,
    ArtifactStore,
    ArtifactStoreError,
    ArtifactType,
    sha256_hex,
)
from research_radar.artifacts.local import LocalArtifactStore

__all__ = [
    "ARTIFACT_MIME_TYPES",
    "ArtifactRef",
    "ArtifactStore",
    "ArtifactStoreError",
    "ArtifactType",
    "LocalArtifactStore",
    "sha256_hex",
]
