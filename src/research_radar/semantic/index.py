"""Concrete semantic-index backends: a disabled no-op, an in-memory fake, and Pinecone.

All backends satisfy the ``SemanticIndex`` protocol from
``research_radar.semantic.base``. The index is a DERIVED cache: it stores
vectors plus the compact ``SemanticRecord.metadata()`` payload and nothing
else. Canonical evidence lives in SQLite.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

from research_radar.semantic.base import (
    EntityType,
    SemanticHit,
    SemanticIndexError,
    SemanticIndexStatus,
    SemanticRecord,
)

logger = logging.getLogger(__name__)

_MAX_PINECONE_BATCH = 100
_ENTITY_TYPES: tuple[str, ...] = ("paper", "papercard")


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity, treating zero-norm vectors as unrelated."""
    if len(left) != len(right):
        raise SemanticIndexError(
            f"Query vector length {len(left)} does not match indexed length {len(right)}."
        )
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _response_field(source: Any, name: str, default: Any = None) -> Any:
    """Read a field from either an object-style or dict-style SDK response."""
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


class DisabledSemanticIndex:
    """A total no-op index used when semantic retrieval is disabled."""

    backend = "disabled"

    @property
    def available(self) -> bool:
        """Always report the disabled backend as unavailable."""

        return False

    def upsert(self, records: Sequence[SemanticRecord]) -> int:
        """Accept and discard records without storing anything."""

        return 0

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 10,
        entity_type: EntityType | None = None,
    ) -> list[SemanticHit]:
        """Always return no candidates."""

        return []

    def delete(self, entity_ids: Sequence[str]) -> int:
        """Report zero removals."""

        return 0

    def status(self) -> SemanticIndexStatus:
        """Report the disabled backend."""

        return SemanticIndexStatus(
            backend="disabled",
            available=False,
            detail="Semantic retrieval is disabled.",
        )


class FakeSemanticIndex:
    """Deterministic in-memory index for tests; never touches a network."""

    backend = "fake"

    def __init__(self, *, available: bool = True) -> None:
        """Create an empty fake index with the requested availability."""

        self._available = available
        self._records: dict[str, SemanticRecord] = {}

    @property
    def available(self) -> bool:
        """Report the simulated availability."""

        return self._available

    def set_available(self, value: bool) -> None:
        """Flip the simulated availability, e.g. to model an outage mid-run."""

        self._available = value

    def upsert(self, records: Sequence[SemanticRecord]) -> int:
        """Store records keyed by entity ID, overwriting duplicates idempotently."""

        accepted = 0
        for record in records:
            self._records[record.entity_id] = record
            accepted += 1
        return accepted

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 10,
        entity_type: EntityType | None = None,
    ) -> list[SemanticHit]:
        """Return the nearest stored records ranked by cosine similarity."""

        if not self._available:
            return []
        for record in self._records.values():
            if len(record.vector) != len(vector):
                raise SemanticIndexError(
                    f"Query vector length {len(vector)} does not match indexed "
                    f"length {len(record.vector)}."
                )
        scored: list[tuple[float, str]] = []
        for entity_id, record in self._records.items():
            if entity_type is not None and record.entity_type != entity_type:
                continue
            scored.append((_cosine_similarity(vector, record.vector), entity_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        limit = max(top_k, 0)
        return [
            SemanticHit(
                entity_id=entity_id,
                entity_type=self._records[entity_id].entity_type,
                paper_id=self._records[entity_id].paper_id,
                score=score,
            )
            for score, entity_id in scored[:limit]
        ]

    def delete(self, entity_ids: Sequence[str]) -> int:
        """Remove entities and return how many were actually present."""

        removed = 0
        for entity_id in entity_ids:
            if self._records.pop(entity_id, None) is not None:
                removed += 1
        return removed

    def status(self) -> SemanticIndexStatus:
        """Report the fake backend and its simulated availability."""

        detail = (
            "In-memory fake semantic index."
            if self._available
            else "In-memory fake semantic index is simulating an outage."
        )
        return SemanticIndexStatus(
            backend="fake",
            available=self._available,
            detail=detail,
        )


class PineconeSemanticIndex:
    """Pinecone-backed derived index that degrades gracefully during outages."""

    backend = "pinecone"

    def __init__(
        self,
        *,
        api_key: str,
        index_name: str,
        namespace: str = "research-radar",
        client: object | None = None,
    ) -> None:
        """Validate credentials and optionally accept an injected test client."""

        if not api_key:
            raise ValueError("api_key must be a non-empty string.")
        if not index_name:
            raise ValueError("index_name must be a non-empty string.")
        self._api_key = api_key
        self._index_name = index_name
        self._namespace = namespace
        self._client_source: Any = client
        self._index_handle: Any = None
        self._available = True

    @property
    def available(self) -> bool:
        """Report whether semantic retrieval can currently be attempted."""

        return self._available

    def _note_outage(self, operation: str, exc: Exception) -> None:
        """Mark the index unavailable and warn once without leaking secrets."""

        self._available = False
        logger.warning(
            "Pinecone %s failed (%s); continuing without semantic retrieval.",
            operation,
            type(exc).__name__,
        )

    def _resolve_index(self) -> Any | None:
        """Return the index handle, or None when setup failed transiently."""

        try:
            return self._ensure_index()
        except Exception as exc:
            # A missing pinecone package raises SemanticIndexError here. It is
            # treated as an outage like any other setup failure so that every
            # method degrades to a no-op consistently, including for callers
            # that use the index directly rather than through HybridRetriever.
            self._note_outage("setup", exc)
            return None

    def _ensure_index(self) -> Any:
        """Lazily create the Pinecone client and index handle on first use."""

        if self._index_handle is not None:
            return self._index_handle
        if self._client_source is None:
            try:
                from pinecone import Pinecone
            except ImportError as exc:
                raise SemanticIndexError(
                    "The Pinecone backend requires the 'pinecone' extra."
                ) from exc
            self._client_source = Pinecone(api_key=self._api_key)
        self._index_handle = self._client_source.Index(self._index_name)
        return self._index_handle

    def upsert(self, records: Sequence[SemanticRecord]) -> int:
        """Idempotently write compact records in batches of at most 100."""

        payloads = [
            {
                "id": record.entity_id,
                "values": record.vector,
                "metadata": record.metadata(),
            }
            for record in records
        ]
        if not payloads:
            return 0
        index = self._resolve_index()
        if index is None:
            return 0
        try:
            for start in range(0, len(payloads), _MAX_PINECONE_BATCH):
                batch = payloads[start : start + _MAX_PINECONE_BATCH]
                index.upsert(vectors=batch, namespace=self._namespace)
        except Exception as exc:
            self._note_outage("upsert", exc)
            return 0
        return len(payloads)

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 10,
        entity_type: EntityType | None = None,
    ) -> list[SemanticHit]:
        """Return bounded hits, or an empty list when the backend is down."""

        index = self._resolve_index()
        if index is None:
            return []
        query_filter = {"entity_type": entity_type} if entity_type else None
        try:
            response = index.query(
                vector=list(vector),
                top_k=top_k,
                namespace=self._namespace,
                include_metadata=True,
                filter=query_filter,
            )
        except Exception as exc:
            self._note_outage("query", exc)
            return []
        hits: list[SemanticHit] = []
        for match in _response_field(response, "matches") or []:
            hit = self._hit_from_match(match)
            if hit is not None:
                hits.append(hit)
        return hits[: max(top_k, 0)]

    def delete(self, entity_ids: Sequence[str]) -> int:
        """Forward the IDs to Pinecone and return how many were requested."""

        ids = list(entity_ids)
        if not ids:
            return 0
        index = self._resolve_index()
        if index is None:
            return 0
        try:
            index.delete(ids=ids, namespace=self._namespace)
        except Exception as exc:
            self._note_outage("delete", exc)
            return 0
        return len(ids)

    def status(self) -> SemanticIndexStatus:
        """Report availability without leaking credentials or endpoints."""

        if self._available:
            return SemanticIndexStatus(
                backend="pinecone",
                available=True,
                detail="Pinecone semantic index is ready.",
            )
        return SemanticIndexStatus(
            backend="pinecone",
            available=False,
            detail="Pinecone semantic index is unavailable; lexical retrieval continues.",
        )

    def _hit_from_match(self, match: Any) -> SemanticHit | None:
        """Map one Pinecone match to a hit, skipping malformed derived records."""

        metadata = _response_field(match, "metadata") or {}
        paper_id = metadata.get("paper_id")
        raw_entity_type = metadata.get("entity_type")
        if not paper_id or not raw_entity_type or raw_entity_type not in _ENTITY_TYPES:
            return None
        entity_id = str(metadata.get("entity_id") or _response_field(match, "id", "") or "")
        if not entity_id:
            return None
        return SemanticHit(
            entity_id=entity_id,
            entity_type=raw_entity_type,
            paper_id=str(paper_id),
            score=float(_response_field(match, "score", 0.0)),
        )
