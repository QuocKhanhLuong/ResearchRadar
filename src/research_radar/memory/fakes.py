"""Deterministic in-memory UserMemoryStore for tests and offline smoke runs.

Shipped in ``src`` so smoke scripts can import it, but it is test
infrastructure: it never touches a network and its ranking is deliberately
simple (case-insensitive token overlap) so assertions stay stable.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from research_radar.memory.models import (
    MemoryClass,
    MemoryFact,
    MemoryStatus,
    UserMemoryContext,
)

_LOGGER = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

_OUTAGE_DETAIL = "simulated outage"


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One accepted episode write, kept so tests can assert what was stored."""

    content: str
    source_description: str
    reference_time: datetime
    memory_class: MemoryClass | None


def _utcnow() -> datetime:
    """Return the current aware UTC time."""

    return datetime.now(UTC)


def _as_utc(moment: datetime) -> datetime:
    """Normalize a naive datetime to UTC; aware datetimes pass through."""

    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment


def _tokens(text: str) -> frozenset[str]:
    """Lowercase a text and split it into alphanumeric tokens."""

    return frozenset(_TOKEN_PATTERN.findall(text.lower()))


class FakeUserMemoryStore:
    """Deterministic in-memory UserMemoryStore for tests and smoke scripts.

    Accepts seeded facts, records every accepted episode, and can be built in
    ``fail=True`` mode that simulates a total backend outage: every method then
    degrades exactly like the real adapters must, and nothing raises.
    """

    def __init__(
        self,
        facts: Iterable[MemoryFact] = (),
        *,
        fail: bool = False,
    ) -> None:
        """Seed initial facts and optionally simulate a total backend outage."""

        self._fail = fail
        self._facts: list[MemoryFact] = list(facts)
        self._episodes: list[EpisodeRecord] = []
        self._warned = False

    @property
    def backend_name(self) -> str:
        """Report the fake backend name."""

        return "fake"

    @property
    def enabled(self) -> bool:
        """Report the store as enabled even while simulating an outage."""

        return True

    @property
    def episodes(self) -> tuple[EpisodeRecord, ...]:
        """Return a read-only snapshot of every accepted episode."""

        return tuple(self._episodes)

    def _warn_once(self, message: str) -> None:
        """Log one sanitized warning per simulated failure class."""

        if not self._warned:
            self._warned = True
            _LOGGER.warning("%s", message)

    async def add_episode(
        self,
        content: str,
        *,
        source_description: str = "discord-chat",
        reference_time: datetime | None = None,
        memory_class: MemoryClass | None = None,
    ) -> bool:
        """Record the episode and expose it as a searchable fact."""

        if self._fail:
            self._warn_once("fake user memory outage: episode not stored")
            return False
        if not content.strip():
            return False
        moment = _as_utc(reference_time) if reference_time is not None else _utcnow()
        self._episodes.append(
            EpisodeRecord(
                content=content,
                source_description=source_description,
                reference_time=moment,
                memory_class=memory_class,
            )
        )
        self._facts.append(
            MemoryFact(
                fact=content,
                memory_class=memory_class,
                valid_at=moment,
                source=source_description,
            )
        )
        return True

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        include_historical: bool = False,
    ) -> list[MemoryFact]:
        """Rank facts by case-insensitive token overlap with the query.

        Facts whose ``invalid_at`` lies in the past are excluded unless
        ``include_historical`` is True. Ties keep insertion order.
        """

        if self._fail:
            self._warn_once("fake user memory outage: search unavailable")
            return []
        bound = max(limit, 0)
        if bound == 0:
            return []
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        cutoff = _utcnow()
        scored: list[tuple[int, int, MemoryFact]] = []
        for index, fact in enumerate(self._facts):
            if (
                not include_historical
                and fact.invalid_at is not None
                and _as_utc(fact.invalid_at) <= cutoff
            ):
                continue
            overlap = len(query_tokens & _tokens(fact.fact))
            if overlap == 0:
                continue
            scored.append((overlap, index, fact))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [replace(fact, score=float(score)) for score, _, fact in scored[:bound]]

    async def get_context(self, query: str, *, limit: int = 8) -> UserMemoryContext:
        """Return the searched facts as bounded advisory context."""

        if self._fail:
            self._warn_once("fake user memory outage: context unavailable")
            return UserMemoryContext(backend=self.backend_name, degraded=True, facts=())
        facts = tuple(await self.search(query, limit=limit))
        return UserMemoryContext(backend=self.backend_name, degraded=False, facts=facts)

    async def status(self) -> MemoryStatus:
        """Report sanitized health; the detail is always a short fixed string."""

        if self._fail:
            self._warn_once("fake user memory outage: status unhealthy")
            return MemoryStatus(
                backend=self.backend_name,
                enabled=True,
                healthy=False,
                detail=_OUTAGE_DETAIL,
                persistence_path=None,
                episode_count=None,
            )
        return MemoryStatus(
            backend=self.backend_name,
            enabled=True,
            healthy=True,
            detail="",
            persistence_path=None,
            episode_count=len(self._episodes),
        )

    async def close(self) -> None:
        """Hold no resources; closing is a no-op even during an outage."""
