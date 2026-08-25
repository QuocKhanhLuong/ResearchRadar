"""The single personal-user-memory boundary for the application.

User memory is ADVISORY context about the single user. It is never a source of
scientific evidence and must never outrank explicit SQLite project state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from research_radar.memory.models import (
    MemoryClass,
    MemoryFact,
    MemoryStatus,
    UserMemoryContext,
)


@runtime_checkable
class UserMemoryStore(Protocol):
    """Async personal-memory boundary. Never a source of scientific evidence."""

    @property
    def backend_name(self) -> str:
        """Return the stable backend identifier used in statuses and contexts."""

    @property
    def enabled(self) -> bool:
        """Return whether the backend is configured to serve user memory."""

    async def add_episode(
        self,
        content: str,
        *,
        source_description: str = "discord-chat",
        reference_time: datetime | None = None,
        memory_class: MemoryClass | None = None,
    ) -> bool:
        """Persist one durable user episode. Returns False when not stored."""

    async def search(self, query: str, *, limit: int = 8) -> list[MemoryFact]:
        """Return bounded advisory facts relevant to the query."""

    async def get_context(self, query: str, *, limit: int = 8) -> UserMemoryContext:
        """Return the bounded advisory context handed to prompt assembly."""

    async def status(self) -> MemoryStatus:
        """Return sanitized backend health; never credentials or content."""

    async def close(self) -> None:
        """Release backend resources; idempotent and never raising."""
