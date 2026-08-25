"""Content-addressed PDF/text/section caching over the shared ArtifactStore.

Parsed documents are keyed by the SHA256 of the source PDF bytes so a repeat
read of an unchanged document never re-parses and never duplicates artifacts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from research_radar.artifacts.base import (
    ArtifactStore,
    ArtifactStoreError,
    ArtifactType,
    sha256_hex,
)
from research_radar.models import PaperDocument
from research_radar.storage.ingestion_repository import IngestionRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CachedDocument:
    """A rebuilt PaperDocument plus the content digest addressing its artifacts."""

    document: PaperDocument
    sha256: str
    source_url: str
    from_cache: bool


class DocumentCache:
    """Content-addressed PDF/text/section cache over an ArtifactStore."""

    def __init__(
        self,
        *,
        store: ArtifactStore,
        ingestion_repository: IngestionRepository,
    ) -> None:
        self._store = store
        self._ingestion_repository = ingestion_repository

    def load(self, *, paper_id: str, sha256: str) -> CachedDocument | None:
        """Return a PaperDocument rebuilt from stored artifacts, or ``None``.

        Both the sections JSON artifact and the plain-text artifact must exist
        for the same ``(paper_id, sha256)`` identity. A corrupt or unparseable
        artifact is reported as a cache miss instead of raising.
        """

        try:
            sections_bytes = self._store.read(
                paper_id=paper_id, sha256=sha256, artifact_type="sections"
            )
            text_bytes = self._store.read(
                paper_id=paper_id, sha256=sha256, artifact_type="text"
            )
        except ArtifactStoreError:
            return None

        try:
            payload = _decode_sections_payload(sections_bytes)
            document = PaperDocument(
                title=payload["title"],
                sections=dict(payload["sections"]),
                full_text=text_bytes.decode("utf-8"),
                source_url=payload.get("source_url"),
                extraction_warning=payload.get("extraction_warning"),
            )
        except (UnicodeDecodeError, ValueError, TypeError, KeyError) as exc:
            logger.warning("Skipping unreadable cached document artifacts: %s", exc)
            return None

        return CachedDocument(
            document=document,
            sha256=sha256,
            source_url=document.source_url or "",
            from_cache=True,
        )

    def store(
        self,
        *,
        paper_id: str,
        content: bytes,
        document: PaperDocument,
        source_url: str | None,
    ) -> str:
        """Store pdf, text, and section artifacts addressed by the source digest.

        Every derived artifact shares the SHA256 of ``content`` as its key, so
        storing identical bytes twice is a no-op at both the store and the
        recorded-reference level. Returns the content digest.
        """

        sha = sha256_hex(content)
        sections_payload = {
            "title": document.title,
            "sections": document.sections,
            "source_url": source_url,
            "extraction_warning": document.extraction_warning,
        }
        artifacts: tuple[tuple[ArtifactType, bytes], ...] = (
            ("pdf", content),
            ("text", document.full_text.encode("utf-8")),
            (
                "sections",
                json.dumps(sections_payload, sort_keys=True, ensure_ascii=False).encode("utf-8"),
            ),
        )
        for artifact_type, payload in artifacts:
            ref = self._store.put(
                paper_id=paper_id,
                content=payload,
                artifact_type=artifact_type,
                sha256=sha,
                source_url=source_url,
            )
            self._ingestion_repository.record_artifact(ref)
        return sha


def _decode_sections_payload(sections_bytes: bytes) -> dict[str, Any]:
    """Decode and structurally validate one stored sections JSON payload."""

    payload = json.loads(sections_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("sections payload must be a JSON object")
    if not isinstance(payload.get("title"), str):
        raise TypeError("sections payload title must be a string")
    if not isinstance(payload.get("sections"), dict):
        raise TypeError("sections payload sections must be an object")
    return payload
