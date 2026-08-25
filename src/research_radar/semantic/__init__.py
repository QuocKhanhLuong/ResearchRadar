"""Derived semantic retrieval: local embeddings and an optional vector index."""

from research_radar.semantic.base import (
    PAPER_SCHEMA_VERSION,
    PAPERCARD_SCHEMA_VERSION,
    SCHEMA_VERSIONS,
    EmbeddingError,
    EmbeddingProvider,
    EntityType,
    SemanticHit,
    SemanticIndex,
    SemanticIndexError,
    SemanticIndexStatus,
    SemanticRecord,
    entity_vector_id,
)

__all__ = [
    "PAPERCARD_SCHEMA_VERSION",
    "PAPER_SCHEMA_VERSION",
    "SCHEMA_VERSIONS",
    "EmbeddingError",
    "EmbeddingProvider",
    "EntityType",
    "SemanticHit",
    "SemanticIndex",
    "SemanticIndexError",
    "SemanticIndexStatus",
    "SemanticRecord",
    "entity_vector_id",
]
