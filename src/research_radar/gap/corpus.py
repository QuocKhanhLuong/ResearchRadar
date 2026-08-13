"""Deterministic topic-scoped corpus selection over stored research memory."""

from __future__ import annotations

from typing import Protocol

from research_radar.storage.repositories import ScopedCorpusResult


class ScopedCorpusRepository(Protocol):
    """Minimum repository interface required for scoped corpus selection."""

    def get_scoped_corpus(self, topic: str, limit: int = 50) -> ScopedCorpusResult: ...


class ScopedCorpusService:
    """Select stored PaperCards and metadata relevant to a specific topic query."""

    def __init__(self, repository: ScopedCorpusRepository) -> None:
        self._repository = repository

    def select_corpus(self, topic: str, limit: int = 50) -> ScopedCorpusResult:
        """Deterministically scope the local research memory for a topic."""

        clean_topic = " ".join(topic.split())
        if not clean_topic:
            raise ValueError("Topic query cannot be empty.")
        return self._repository.get_scoped_corpus(clean_topic, limit=limit)
