"""The always-off UserMemoryStore used when user memory is disabled."""

from __future__ import annotations

from datetime import datetime

from research_radar.memory.models import (
    MemoryClass,
    MemoryFact,
    MemoryStatus,
    UserMemoryContext,
)


class DisabledUserMemoryStore:
    """No-op backend: never enabled, never raises, never stores anything."""

    @property
    def backend_name(self) -> str:
        """Report the disabled backend name."""

        return "disabled"

    @property
    def enabled(self) -> bool:
        """Always report the store as not enabled."""

        return False

    async def add_episode(
        self,
        content: str,
        *,
        source_description: str = "discord-chat",
        reference_time: datetime | None = None,
        memory_class: MemoryClass | None = None,
    ) -> bool:
        """Accept and discard the episode; report it as not stored."""

        return False

    async def search(self, query: str, *, limit: int = 8) -> list[MemoryFact]:
        """Always return no facts."""

        return []

    async def get_context(self, query: str, *, limit: int = 8) -> UserMemoryContext:
        """Return an empty, non-degraded context for the disabled backend."""

        return UserMemoryContext(backend=self.backend_name, degraded=False, facts=())

    async def status(self) -> MemoryStatus:
        """Report the disabled backend as intentionally off, not unhealthy."""

        return MemoryStatus(
            backend=self.backend_name,
            enabled=False,
            healthy=True,
            detail="",
            persistence_path=None,
            episode_count=None,
        )

    async def close(self) -> None:
        """Hold no resources; closing is a no-op."""
