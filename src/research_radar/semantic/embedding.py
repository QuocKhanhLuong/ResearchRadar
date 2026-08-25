"""Embedding providers and deterministic vector-text builders.

Only bounded, preprocessed text is ever embedded: paper title plus abstract,
or the compact PaperCard summary. Full documents are never embedded.
"""

from __future__ import annotations

import hashlib
import math
import threading
import unicodedata
from collections.abc import Sequence
from typing import Any, Protocol

from research_radar.models.paper_card import PaperCard
from research_radar.semantic.base import SCHEMA_VERSIONS, EmbeddingError, EmbeddingProvider


class _PaperLike(Protocol):
    """Structural type for anything with a title and optional abstract."""

    @property
    def title(self) -> str: ...

    @property
    def abstract(self) -> str | None: ...


def prepare_text(text: str, *, max_chars: int = 2000) -> str:
    """NFKC-normalize, collapse whitespace, strip, and bound to ``max_chars``."""

    normalized = unicodedata.normalize("NFKC", text)
    collapsed = " ".join(normalized.split()).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    truncated = collapsed[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space].rstrip()
    return truncated.rstrip()


def paper_vector_text(paper: _PaperLike) -> str:
    """Return the bounded embedding text: title plus abstract, nothing more."""

    return prepare_text(f"{paper.title}\n{paper.abstract or ''}")


def paper_card_vector_text(card: PaperCard) -> str:
    """Return the deterministic labeled summary of a PaperCard for embedding."""

    parts: list[str] = []
    if card.problem:
        parts.append(f"Problem: {card.problem}")
    if card.contributions:
        parts.append(f"Contributions: {'; '.join(card.contributions)}")
    if card.methods:
        parts.append(f"Methods: {'; '.join(card.methods)}")
    if card.tasks:
        parts.append(f"Tasks: {'; '.join(ev.value for ev in card.tasks)}")
    if card.modalities:
        parts.append(f"Modalities: {'; '.join(ev.value for ev in card.modalities)}")
    if card.datasets:
        parts.append(f"Datasets: {'; '.join(card.datasets)}")
    if card.metrics:
        parts.append(f"Metrics: {'; '.join(card.metrics)}")
    if card.main_claims:
        parts.append(f"Claims: {'; '.join(claim.claim for claim in card.main_claims)}")
    if card.limitations:
        parts.append(f"Limitations: {'; '.join(card.limitations)}")
    return prepare_text("\n".join(parts))


def embedding_fingerprint(
    provider: EmbeddingProvider, *, entity_type: str
) -> tuple[str, str, int]:
    """Return ``(schema_version, model_id, dimension)`` for reindex detection."""

    try:
        schema_version = SCHEMA_VERSIONS[entity_type]
    except KeyError as exc:
        raise EmbeddingError(f"Unknown entity_type: {entity_type!r}") from exc
    return (schema_version, provider.model_id, provider.dimension)


class FakeEmbeddingProvider:
    """Deterministic dependency-free embedding provider for unit tests."""

    def __init__(self, *, dimension: int = 8, model_id: str = "fake-embedding-v1") -> None:
        if dimension < 1:
            raise EmbeddingError("Embedding dimension must be at least 1.")
        self.model_id = model_id
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the fixed vector dimension produced by this provider."""

        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Map each text deterministically to a unit-norm pseudo-random vector."""

        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(prepare_text(text).encode("utf-8")).digest()
            raw = [
                ((digest[i % len(digest)] / 255.0) * 2) - 1 for i in range(self._dimension)
            ]
            norm = math.sqrt(sum(component * component for component in raw))
            if norm == 0.0:
                vectors.append([0.0] * self._dimension)
            else:
                vectors.append([component / norm for component in raw])
        return vectors


class LocalEmbeddingProvider:
    """Sentence-transformers backed provider with lazy, offline-safe loading."""

    def __init__(
        self,
        *,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_input_chars: int = 2000,
        batch_size: int = 16,
    ) -> None:
        self.model_id = model_id
        self._max_input_chars = max_input_chars
        self._batch_size = batch_size
        self._model: Any | None = None
        self._load_lock = threading.Lock()

    def _ensure_model(self) -> Any:
        """Load the sentence-transformers model once, lazily and offline-safe."""

        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                msg = (
                    "The local embedding backend requires the "
                    "'sentence-transformers' extra."
                )
                raise EmbeddingError(msg) from exc
            try:
                self._model = SentenceTransformer(self.model_id)
            except Exception as exc:
                raise EmbeddingError(
                    f"The local embedding model '{self.model_id}' failed to load."
                ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        """Return the loaded model's sentence embedding dimension."""

        dimension = self._ensure_model().get_sentence_embedding_dimension()
        if not dimension:
            raise EmbeddingError(
                f"The local embedding model '{self.model_id}' reported no dimension."
            )
        return int(dimension)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode preprocessed texts in batches, preserving input order."""

        prepared = [prepare_text(text, max_chars=self._max_input_chars) for text in texts]
        if not prepared:
            return []
        model = self._ensure_model()
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(prepared), self._batch_size):
                batch = prepared[start : start + self._batch_size]
                embeddings = model.encode(
                    batch,
                    batch_size=self._batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                vectors.extend(row.tolist() for row in embeddings)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Local embedding encoding failed.") from exc
        return vectors
