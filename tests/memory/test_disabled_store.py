"""Unit tests for the DisabledUserMemoryStore."""

from __future__ import annotations

from datetime import UTC, datetime

from research_radar.memory import (
    DisabledUserMemoryStore,
    MemoryClass,
    UserMemoryStore,
)

SECRET_SHAPED = "Authorization: Bearer sk-ant-not-a-real-key"


async def test_disabled_store_satisfies_protocol() -> None:
    """The store is a structural UserMemoryStore."""

    assert isinstance(DisabledUserMemoryStore(), UserMemoryStore)


def test_disabled_store_never_reports_enabled() -> None:
    """backend and enabled reflect the off state before any call."""

    store = DisabledUserMemoryStore()
    assert store.backend_name == "disabled"
    assert store.enabled is False


async def test_disabled_store_methods_never_raise_and_never_store() -> None:
    """Every method completes quietly with empty results."""

    store = DisabledUserMemoryStore()
    assert await store.add_episode(
        SECRET_SHAPED,
        source_description="discord-chat",
        reference_time=datetime.now(tz=UTC),
        memory_class=MemoryClass.PREFERENCE,
    ) is False
    assert await store.search("protein folding") == []
    context = await store.get_context("anything at all")
    assert context.backend == "disabled"
    assert context.degraded is False
    assert context.facts == ()
    assert context.available is False
    await store.close()


async def test_disabled_store_status_is_safe_and_off() -> None:
    """Status reports disabled, healthy-by-intent, and carries no secrets."""

    status = await DisabledUserMemoryStore().status()
    assert status.backend == "disabled"
    assert status.enabled is False
    assert status.healthy is True
    assert status.detail == "" or len(status.detail) < 64
    for marker in ("sk-", "Bearer ", "Authorization:", "password"):
        assert marker not in status.detail.lower()
