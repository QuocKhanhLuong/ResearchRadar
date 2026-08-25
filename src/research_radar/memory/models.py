"""Value types for the advisory user-memory subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class MemoryClass(StrEnum):
    """Coarse classification of one durable user-context fact."""

    PREFERENCE = "preference"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    RESEARCH_INTEREST = "research_interest"
    PROJECT_DECISION = "project_decision"
    REJECTED_IDEA = "rejected_idea"
    TOOL_PREFERENCE = "tool_preference"
    WORKFLOW_PREFERENCE = "workflow_preference"
    RESEARCH_DIRECTION = "research_direction"
    TEMPORAL_PLAN = "temporal_plan"


@dataclass(frozen=True, slots=True)
class MemoryFact:
    """One advisory user-context fact recovered from the memory backend."""

    fact: str
    memory_class: MemoryClass | None = None
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    source: str = "graphiti"
    score: float | None = None


@dataclass(frozen=True, slots=True)
class UserMemoryContext:
    """Bounded, advisory user context handed to prompt assembly."""

    facts: tuple[MemoryFact, ...] = ()
    backend: str = "disabled"
    degraded: bool = False

    @property
    def available(self) -> bool:
        """Return whether any usable fact survived retrieval."""

        return bool(self.facts)


@dataclass(frozen=True, slots=True)
class MemoryStatus:
    """Safe-to-display backend health. Must never contain credentials."""

    backend: str
    enabled: bool
    healthy: bool
    detail: str = ""
    persistence_path: str | None = None
    episode_count: int | None = None
