"""Provider-neutral scholarly discovery contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from research_radar.models.paper import Paper


@runtime_checkable
class PaperProvider(Protocol):
    """A scholarly source that returns normalized papers for a text query."""

    name: str

    async def search(self, query: str, limit: int = 10) -> list[Paper]:
        """Return at most `limit` normalized papers or raise a provider error."""


def clamp_provider_limit(limit: int, *, maximum: int = 25) -> int:
    """Defensively bound requests independently of a user-facing service limit."""

    return max(1, min(limit, maximum))
