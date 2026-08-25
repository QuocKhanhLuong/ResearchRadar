"""Unit tests for the GraphitiUserMemoryStore adapter.

These tests run fully offline and never require graphiti-core to be installed:
the backend is driven through an injected fake client factory. The one real
integration test at the bottom skips itself when graphiti-core/kuzu are absent.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from research_radar.config import Settings
from research_radar.memory import UserMemoryStore
from research_radar.memory.graphiti_store import GraphitiUserMemoryStore

_TEST_CREDENTIAL = "sk-test-DO-NOT-LEAK-9876543210abcdef"
_TEST_EPISODE_TEXT = "EPISODE-MARKER-zqxwcv prefers plotly over matplotlib"


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    """Build isolated settings pointing at a temporary database location."""

    values: dict[str, object] = {
        "user_memory_backend": "graphiti",
        "user_memory_db_path": str(tmp_path / "memory" / "user_memory"),
        "user_memory_group_id": "primary-user",
        "llm_base_url": "http://llm.invalid/v1",
        "llm_api_key": _TEST_CREDENTIAL,
        "llm_model": "test-model",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@dataclass(frozen=True)
class FakeEdge:
    """Minimal stand-in for graphiti_core.edges.EntityEdge."""

    fact: str
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    expired_at: datetime | None = None
    score: float | None = None


class FakeGraphiti:
    """Fake Graphiti client recording calls and simulating failures."""

    def __init__(
        self,
        *,
        edges: list[FakeEdge] | None = None,
        fail_methods: set[str] | None = None,
    ) -> None:
        self.edges = list(edges or [])
        self.fail_methods = set(fail_methods or ())
        self.build_calls = 0
        self.close_calls = 0
        self.add_episode_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []

    async def build_indices_and_constraints(self) -> None:
        if "build" in self.fail_methods:
            raise RuntimeError("simulated build failure")
        self.build_calls += 1
        await asyncio.sleep(0)

    async def add_episode(self, **kwargs: Any) -> None:
        if "add_episode" in self.fail_methods:
            raise RuntimeError("simulated write failure")
        self.add_episode_calls.append(kwargs)

    async def search(self, query: str, **kwargs: Any) -> list[FakeEdge]:
        if "search" in self.fail_methods:
            raise RuntimeError("simulated search failure")
        self.search_calls.append({"query": query, **kwargs})
        num_results = int(kwargs.get("num_results", 10))
        return self.edges[:num_results]

    async def close(self) -> int:
        if "close" in self.fail_methods:
            raise RuntimeError("simulated close failure")
        self.close_calls += 1
        return self.close_calls


def make_store(
    tmp_path: Path,
    fake: FakeGraphiti,
    *,
    factory_calls: list[int] | None = None,
    **settings_overrides: object,
) -> GraphitiUserMemoryStore:
    """Build a store wired to a fake client via an injectable factory."""

    def factory() -> FakeGraphiti:
        if factory_calls is not None:
            factory_calls.append(1)
        return fake

    return GraphitiUserMemoryStore(
        make_settings(tmp_path, **settings_overrides),
        client_factory=factory,
    )


def warning_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return captured records at WARNING or above for the adapter logger."""

    return [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and record.name == "research_radar.memory.graphiti_store"
    ]


async def test_importing_the_package_and_module_stays_lazy() -> None:
    """Importing research_radar.memory must never pull in graphiti_core/kuzu."""

    if "graphiti_core" in sys.modules or "kuzu" in sys.modules:
        pytest.skip("graphiti_core/kuzu already imported by another test")
    sys.modules.pop("research_radar.memory.graphiti_store", None)
    importlib.import_module("research_radar.memory.graphiti_store")
    assert "graphiti_core" not in sys.modules
    assert "kuzu" not in sys.modules


async def test_store_satisfies_user_memory_store_protocol(tmp_path: Path) -> None:
    """The adapter is a structural UserMemoryStore."""

    store = make_store(tmp_path, FakeGraphiti())
    assert isinstance(store, UserMemoryStore)
    await store.close()


async def test_storage_directory_created_on_first_use_not_at_construction(
    tmp_path: Path,
) -> None:
    """No filesystem work happens until a method first touches the backend."""

    settings = make_settings(tmp_path)
    db_parent = Path(settings.user_memory_db_path).parent
    store = GraphitiUserMemoryStore(settings, client_factory=FakeGraphiti)
    assert not db_parent.exists()
    assert await store.search("anything at all") == []
    assert db_parent.exists()
    await store.close()


async def test_initialization_happens_exactly_once_under_concurrency(
    tmp_path: Path,
) -> None:
    """Concurrent chat turns share one lazy initialization."""

    fake = FakeGraphiti()
    factory_calls: list[int] = []
    store = make_store(tmp_path, fake, factory_calls=factory_calls)
    results = list(
        await asyncio.gather(*(store.search(f"query {n}") for n in range(8)))
    )
    assert all(result == [] for result in results)
    assert len(factory_calls) == 1
    assert fake.build_calls == 1
    await store.close()


async def test_missing_dependency_degrades_every_method_and_warns_once(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An ImportError behaves like a total outage with exactly one warning."""

    def broken_factory() -> Any:
        raise ImportError("No module named 'graphiti_core'")

    store = GraphitiUserMemoryStore(
        make_settings(tmp_path),
        client_factory=broken_factory,
    )
    with caplog.at_level(logging.WARNING):
        assert await store.add_episode(_TEST_EPISODE_TEXT) is False
        assert await store.search("plotly preferences") == []
        context = await store.get_context("plotly preferences")
        status = await store.status()
        second = await store.search("plotly preferences")
        assert second == []
    assert context.backend == "graphiti"
    assert context.degraded is True
    assert context.facts == ()
    assert status.backend == "graphiti"
    assert status.enabled is True
    assert status.healthy is False
    assert status.detail == "user memory backend unavailable"
    assert len(warning_records(caplog)) == 1
    assert "memory" in caplog.text
    await store.close()


async def test_initialization_failure_degrades_and_never_retries(tmp_path: Path) -> None:
    """A non-ImportError construction failure sticks until restart."""

    fake = FakeGraphiti(fail_methods={"build"})
    factory_calls: list[int] = []
    store = make_store(tmp_path, fake, factory_calls=factory_calls)
    assert await store.search("first attempt") == []
    assert await store.search("second attempt") == []
    status = await store.status()
    assert status.healthy is False
    assert len(factory_calls) == 1
    assert fake.build_calls == 0
    await store.close()


async def test_add_episode_failure_degrades_once_and_logs_once(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed write returns False once and never raises or spams."""

    fake = FakeGraphiti(fail_methods={"add_episode"})
    store = make_store(tmp_path, fake)
    with caplog.at_level(logging.WARNING):
        assert await store.add_episode(_TEST_EPISODE_TEXT) is False
        assert await store.add_episode("second EPISODE-MARKER-zqxwcv attempt") is False
    assert len(fake.add_episode_calls) == 0
    assert len(warning_records(caplog)) == 1
    await store.close()


async def test_search_failure_degrades_context_with_one_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing search yields empty facts and degraded context, warned once."""

    fake = FakeGraphiti(fail_methods={"search"})
    store = make_store(tmp_path, fake)
    with caplog.at_level(logging.WARNING):
        assert await store.search("protein folding") == []
        context = await store.get_context("protein folding")
        assert (await store.get_context("another question")).facts == ()
    assert context.degraded is True
    assert context.facts == ()
    assert len(warning_records(caplog)) == 1
    await store.close()


async def test_temporal_filtering_and_field_mapping(tmp_path: Path) -> None:
    """Superseded facts hide by default; history stays reachable; fields map."""

    now = datetime.now(tz=UTC)
    fake = FakeGraphiti(
        edges=[
            FakeEdge(fact="prefers local embeddings today", score=3.5),
            FakeEdge(
                fact="future invalidation still current",
                invalid_at=now + timedelta(days=30),
                score=2.0,
            ),
            FakeEdge(
                fact="August preference superseded",
                valid_at=now - timedelta(days=60),
                invalid_at=now - timedelta(days=1),
                score=9.0,
            ),
            FakeEdge(
                fact="expired but never invalidated",
                expired_at=now - timedelta(days=2),
            ),
            FakeEdge(
                fact="expiry scheduled later",
                expired_at=now + timedelta(days=10),
            ),
        ]
    )
    store = make_store(tmp_path, fake)
    default_facts = await store.search("embedding preferences", limit=10)
    assert [fact.fact for fact in default_facts] == [
        "prefers local embeddings today",
        "future invalidation still current",
        "expiry scheduled later",
    ]
    assert default_facts[0].score == 3.5
    historical_facts = await store.search(
        "embedding preferences", limit=10, include_historical=True
    )
    assert {fact.fact for fact in historical_facts} == {
        "prefers local embeddings today",
        "future invalidation still current",
        "August preference superseded",
        "expired but never invalidated",
        "expiry scheduled later",
    }
    august = next(
        fact for fact in historical_facts if fact.fact == "August preference superseded"
    )
    assert august.valid_at == now - timedelta(days=60)
    assert august.invalid_at == now - timedelta(days=1)
    assert august.source == "graphiti"
    assert await store.search("embedding preferences", limit=0) == []
    await store.close()


async def test_close_is_idempotent_and_safe_before_initialization(tmp_path: Path) -> None:
    """close() works pre-init and collapses repeated calls into one."""

    factory_calls: list[int] = []
    store = make_store(tmp_path, FakeGraphiti(), factory_calls=factory_calls)
    await store.close()
    await store.close()
    assert factory_calls == []


async def test_close_after_init_closes_underlying_client_once(tmp_path: Path) -> None:
    """The wrapped client is closed exactly once across repeated closes."""

    fake = FakeGraphiti()
    store = make_store(tmp_path, fake)
    assert await store.search("warm up the backend") == []
    await store.close()
    await store.close()
    assert fake.close_calls == 1
    assert await store.search("after close") == []


async def test_group_id_scopes_writes_and_reads(tmp_path: Path) -> None:
    """Every backend call is scoped to the configured single-user group."""

    fake = FakeGraphiti(edges=[FakeEdge(fact="scoped fact about retrieval eval")])
    store = make_store(tmp_path, fake)
    assert await store.add_episode(_TEST_EPISODE_TEXT) is True
    assert await store.search("retrieval evaluation") != []
    assert fake.add_episode_calls[0]["group_id"] == "primary-user"
    assert fake.search_calls[0]["group_ids"] == ["primary-user"]
    await store.close()


async def test_entity_types_carry_memory_class_on_writes(tmp_path: Path) -> None:
    """Writes with a class pass the full keyed table; untyped writes pass None."""

    from research_radar.memory import MemoryClass

    fake = FakeGraphiti()
    store = make_store(tmp_path, fake)
    await store.add_episode("I prefer plotly", memory_class=MemoryClass.TOOL_PREFERENCE)
    await store.add_episode("unclassified note")
    typed_keys = set(fake.add_episode_calls[0]["entity_types"])
    expected_keys = {f"memory_{cls.value}" for cls in MemoryClass}
    assert typed_keys == expected_keys
    assert fake.add_episode_calls[1]["entity_types"] is None
    await store.close()


async def test_blank_query_skips_backend_entirely(tmp_path: Path) -> None:
    """Empty queries touch neither the filesystem nor the fake backend."""

    fake = FakeGraphiti()
    factory_calls: list[int] = []
    store = make_store(tmp_path, fake, factory_calls=factory_calls)
    assert await store.search("") == []
    assert await store.search("   ") == []
    context = await store.get_context("")
    assert context.degraded is False
    assert context.available is False
    assert factory_calls == []


async def test_no_episode_text_or_credential_leaks_into_logs_or_status(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failures and successes alike keep content and credentials out of view."""

    fake = FakeGraphiti(edges=[FakeEdge(fact=_TEST_EPISODE_TEXT)], fail_methods={"search"})
    store = make_store(tmp_path, fake)
    with caplog.at_level(logging.WARNING):
        assert await store.add_episode(_TEST_EPISODE_TEXT) is True
        assert await store.search("anything") == []
        status = await store.status()
        assert status.healthy is False
    leak_sources = [caplog.text]
    leak_sources.append(
        " ".join(
            str(value)
            for value in (
                status.backend,
                status.enabled,
                status.healthy,
                status.detail,
                status.persistence_path,
                status.episode_count,
            )
        )
    )
    for source in leak_sources:
        assert _TEST_EPISODE_TEXT not in source
        assert _TEST_CREDENTIAL not in source
    assert _TEST_EPISODE_TEXT in str(fake.add_episode_calls[0]["episode_body"])
    await store.close()


async def test_default_client_factory_builds_real_backend_offline(tmp_path: Path) -> None:
    """With graphiti-core installed, real init succeeds offline without warnings."""

    environ_before = dict(os.environ)
    settings = make_settings(tmp_path)
    store = GraphitiUserMemoryStore(settings)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            status = await store.status()
    finally:
        await store.close()
    if "graphiti_core" not in sys.modules:
        pytest.skip("graphiti-core/kuzu are not installed")
    kuzu_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, DeprecationWarning) and "Kuzu" in str(warning.message)
    ]
    assert kuzu_warnings == []
    assert status.healthy is True
    assert status.detail == ""
    assert Path(settings.user_memory_db_path).exists()
    assert dict(os.environ) == environ_before


async def test_missing_llm_configuration_degrades_with_an_actionable_message(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """USER_MEMORY_BACKEND=graphiti without LLM settings must say so plainly.

    Graphiti needs an LLM for extraction and an embedder for search, so it
    cannot run against LLM_PROVIDER=mock. The operator has to be told that
    rather than being pointed at a generic backend failure, and no database
    file may be left behind by the doomed attempt.
    """

    for override in (
        {"llm_api_key": None},
        {"llm_base_url": None},
        {"llm_model": None},
    ):
        caplog.clear()
        settings = make_settings(tmp_path, **override)
        built = False

        def _never_called() -> object:  # pragma: no cover - must not run
            nonlocal built
            built = True
            raise AssertionError("client factory ran despite missing LLM settings")

        store = GraphitiUserMemoryStore(settings, client_factory=_never_called)
        with caplog.at_level(logging.WARNING):
            status = await store.status()

        assert built is False
        assert status.healthy is False
        messages = " ".join(record.getMessage() for record in warning_records(caplog))
        assert "LLM_BASE_URL" in messages
        assert "LLM_API_KEY" in messages
        assert _TEST_CREDENTIAL not in messages
        assert not settings.user_memory_db_path_resolved().parent.exists()

        assert await store.add_episode("I prefer duckdb") is False
        assert await store.search("preferences") == []
        await store.close()
