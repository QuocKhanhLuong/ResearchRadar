"""Local filesystem backend for content-addressed document artifacts.

Layout::

    <artifact_root>/papers/<paper_id>/<sha256><suffix>

Writes are atomic (temp file + ``os.replace``) and idempotent: storing the
same bytes twice never duplicates or rewrites the file.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from research_radar.artifacts.base import (
    ARTIFACT_MIME_TYPES,
    ARTIFACT_SUFFIXES,
    ArtifactRef,
    ArtifactStoreError,
    ArtifactType,
    sha256_hex,
)

_PAPER_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

__all__ = ["ArtifactStoreError", "LocalArtifactStore"]


def _validate_paper_id(paper_id: str) -> None:
    """Reject paper ids that could escape the artifact root directory."""

    if (
        not _PAPER_ID_RE.fullmatch(paper_id)
        or ".." in paper_id
        or "/" in paper_id
        or "\\" in paper_id
        or paper_id.startswith(".")
    ):
        raise ArtifactStoreError(f"unsafe paper_id: {paper_id!r}")


def _validate_sha256(sha256: str) -> str:
    """Require a 64-character lowercase hex SHA256 digest."""

    if not _SHA256_RE.fullmatch(sha256):
        raise ArtifactStoreError(f"invalid sha256 digest: {sha256!r}")

    return sha256


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

        _validate_paper_id(paper_id)
        digest = sha256_hex(content) if sha256 is None else _validate_sha256(sha256)
        final = self._object_path(paper_id, digest, artifact_type)
        ref = ArtifactRef(
            paper_id=paper_id,
            sha256=digest,
            artifact_type=artifact_type,
            backend=self.backend,
            object_key=self._object_key(final),
            mime_type=ARTIFACT_MIME_TYPES[artifact_type],
            byte_size=len(content),
            source_url=source_url,
        )
        if final.is_file() and final.stat().st_size == len(content):
            return ref
        final.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(final, content)

        return ref

    def exists(self, *, paper_id: str, sha256: str, artifact_type: ArtifactType) -> bool:
        """Return whether an artifact is already stored for this identity."""

        try:
            _validate_paper_id(paper_id)
            _validate_sha256(sha256)
        except ArtifactStoreError:
            return False
        return self._object_path(paper_id, sha256, artifact_type).is_file()

    def read(self, *, paper_id: str, sha256: str, artifact_type: ArtifactType) -> bytes:
        """Return stored bytes, raising ``ArtifactStoreError`` when absent."""

        _validate_paper_id(paper_id)
        _validate_sha256(sha256)
        path = self._object_path(paper_id, sha256, artifact_type)
        try:
            return path.read_bytes()
        except OSError:
            raise ArtifactStoreError(
                f"{artifact_type} artifact sha256:{sha256[:12]} not found"
            ) from None

    def resolve(
        self, *, paper_id: str, sha256: str, artifact_type: ArtifactType
    ) -> ArtifactRef | None:
        """Return the stored reference, or ``None`` when it does not exist."""

        try:
            _validate_paper_id(paper_id)
            _validate_sha256(sha256)
        except ArtifactStoreError:
            return None
        path = self._object_path(paper_id, sha256, artifact_type)
        if not path.is_file():
            return None
        return ArtifactRef(
            paper_id=paper_id,
            sha256=sha256,
            artifact_type=artifact_type,
            backend=self.backend,
            object_key=self._object_key(path),
            mime_type=ARTIFACT_MIME_TYPES[artifact_type],
            byte_size=path.stat().st_size,
            source_url=None,
        )

    def _object_path(self, paper_id: str, sha256: str, artifact_type: ArtifactType) -> Path:
        """Return the final on-disk path for one artifact."""

        suffix = ARTIFACT_SUFFIXES[artifact_type]

        return self._root / "papers" / paper_id / f"{sha256}{suffix}"

    def _object_key(self, path: Path) -> str:
        """Return the root-relative, forward-slash key for an on-disk path."""

        return path.relative_to(self._root).as_posix()

    def _atomic_write(self, final: Path, content: bytes) -> None:
        """Write ``content`` to ``final`` atomically via a temp file + rename."""

        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=final.parent,
                prefix=f".{final.name}.",
                suffix=".part",
                delete=False,
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                temp_name = handle.name
            os.replace(temp_name, final)
            temp_name = None
        finally:
            if temp_name is not None:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
