"""Unit tests for the memory value types."""

from __future__ import annotations

import dataclasses

import pytest

from research_radar.memory.models import (
    MemoryClass,
    MemoryFact,
    MemoryStatus,
    UserMemoryContext,
)


def test_memory_class_values_are_stable_strings() -> None:
    """Every member keeps its contract value and behaves as a plain string."""

    expected = {
        "PREFERENCE": "preference",
        "GOAL": "goal",
        "CONSTRAINT": "constraint",
        "RESEARCH_INTEREST": "research_interest",
        "PROJECT_DECISION": "project_decision",
        "REJECTED_IDEA": "rejected_idea",
        "TOOL_PREFERENCE": "tool_preference",
        "WORKFLOW_PREFERENCE": "workflow_preference",
        "RESEARCH_DIRECTION": "research_direction",
        "TEMPORAL_PLAN": "temporal_plan",
    }
    assert {member.name: member.value for member in MemoryClass} == expected
    assert isinstance(MemoryClass.PREFERENCE, str)
    assert MemoryClass.PREFERENCE == "preference"


def test_memory_fact_defaults() -> None:
    """A fact needs only its text; every other field has a safe default."""

    fact = MemoryFact(fact="Prefers plotly for figures")
    assert fact.memory_class is None
    assert fact.valid_at is None
    assert fact.invalid_at is None
    assert fact.source == "graphiti"
    assert fact.score is None


def test_memory_fact_is_frozen() -> None:
    """Facts are immutable value objects."""

    fact = MemoryFact(fact="Works on protein folding")
    with pytest.raises(dataclasses.FrozenInstanceError):
        fact.fact = "mutated"  # type: ignore[misc]


def test_memory_context_defaults_to_disabled_and_empty() -> None:
    """The default context is the disabled backend with no facts."""

    context = UserMemoryContext()
    assert context.facts == ()
    assert context.backend == "disabled"
    assert context.degraded is False
    assert context.available is False


def test_user_memory_context_available() -> None:
    """available flips to True exactly when at least one fact survived."""

    empty = UserMemoryContext(backend="fake", degraded=True, facts=())
    assert empty.available is False

    populated = UserMemoryContext(
        facts=(MemoryFact(fact="Uses uv", memory_class=MemoryClass.TOOL_PREFERENCE),),
        backend="graphiti",
    )
    assert populated.available is True


def test_memory_status_defaults() -> None:
    """Optional status fields default to None/empty, never to content."""

    status = MemoryStatus(backend="disabled", enabled=False, healthy=True)
    assert status.detail == ""
    assert status.persistence_path is None
    assert status.episode_count is None


def test_memory_status_has_no_secret_shaped_surface() -> None:
    """detail stays empty or a short fixed string and no field echoes secrets."""

    fields = set(MemoryStatus.__dataclass_fields__)
    assert fields <= {
        "backend",
        "enabled",
        "healthy",
        "detail",
        "persistence_path",
        "episode_count",
    }
    statuses = [
        MemoryStatus(backend="disabled", enabled=False, healthy=True),
        MemoryStatus(
            backend="graphiti",
            enabled=True,
            healthy=False,
            detail="unreachable",
            persistence_path="data/user_memory",
            episode_count=7,
        ),
    ]
    for status in statuses:
        assert len(status.detail) < 64
        for marker in ("sk-", "sk-ant-", "pcsk_", "Bearer ", "Authorization:", "password"):
            assert marker not in status.detail


def test_user_memory_context_is_frozen() -> None:
    """UserMemoryContext instances cannot be mutated."""

    context = UserMemoryContext()
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.degraded = True  # type: ignore[misc]


def test_memory_status_is_frozen() -> None:
    """MemoryStatus instances cannot be mutated."""

    status = MemoryStatus(backend="disabled", enabled=False, healthy=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        status.healthy = False  # type: ignore[misc]


def test_memory_models_have_slots() -> None:
    """All memory value types use slots for memory efficiency and safety."""

    for model_cls in (MemoryFact, UserMemoryContext, MemoryStatus):
        assert hasattr(model_cls, "__slots__")
        instance = (
            model_cls(fact="test")
            if model_cls is MemoryFact
            else model_cls(backend="test", enabled=True, healthy=True)
            if model_cls is MemoryStatus
            else model_cls()
        )
        assert not hasattr(instance, "__dict__")

