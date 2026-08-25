"""Advisory user-memory boundary: models, protocol, and safe implementations."""

from __future__ import annotations

from research_radar.memory.base import UserMemoryStore
from research_radar.memory.disabled import DisabledUserMemoryStore
from research_radar.memory.fakes import EpisodeRecord, FakeUserMemoryStore
from research_radar.memory.models import (
    MemoryClass,
    MemoryFact,
    MemoryStatus,
    UserMemoryContext,
)

__all__ = [
    "DisabledUserMemoryStore",
    "EpisodeRecord",
    "FakeUserMemoryStore",
    "MemoryClass",
    "MemoryFact",
    "MemoryStatus",
    "UserMemoryContext",
    "UserMemoryStore",
]
