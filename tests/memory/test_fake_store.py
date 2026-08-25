"""Unit tests for the FakeUserMemoryStore."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from research_radar.memory import (
    EpisodeRecord,
    FakeUserMemoryStore,
    MemoryClass,
    MemoryFact,
    UserMemoryContext,
    UserMemoryStore,
)


def _fact(text: str, **overrides: object) -> MemoryFact:
    """Build a MemoryFact with optional field overrides."""

    return MemoryFact(fact=text, **overrides)  # type: ignore[arg-type]


async def test_fake_store_satisfies_protocol() -> None:
    """The fake is a structural UserMemoryStore."""

    assert isinstance(FakeUserMemoryStore(), UserMemoryStore)


async def test_search_ranks_by_case_insensitive_token_overlap() -> None:
    """More overlapping tokens rank first; matching is case-insensitive."""

    store = FakeUserMemoryStore(
        [
            _fact("Vector databases for semantic search"),
            _fact("PROTEIN folding simulations on GPUs"),
            _fact("Protein folding and vector databases together"),
        ]
    )
    facts = await store.search("protein folding")
    assert facts[0].fact == "PROTEIN folding simulations on GPUs"
    assert facts[0].score == 2.0
    assert facts[1].fact == "Protein folding and vector databases together"
    assert facts[1].score == 2.0  # same overlap; insertion order broke the tie
    assert len(facts) == 2


async def test_search_is_deterministic_across_calls() -> None:
    """Repeated identical queries return identical rankings."""

    store = FakeUserMemoryStore([_fact("prefers plotly"), _fact("prefers matplotlib")])
    first = await store.search("plotly")
    second = await store.search("plotly")
    assert [f.fact for f in first] == [f.fact for f in second]


async def test_search_respects_limit_and_returns_empty_on_no_overlap() -> None:
    """limit bounds results; unrelated queries match nothing."""

    store = FakeUserMemoryStore([_fact("a"), _fact("b"), _fact("c")])
    assert len(await store.search("a b c", limit=2)) == 2
    assert await store.search("zzz qqq") == []
    assert await store.search("") == []


async def test_add_episode_records_and_becomes_searchable() -> None:
    """Accepted writes land in the read-only episodes log and the index."""

    store = FakeUserMemoryStore()
    moment = datetime.now(tz=UTC)
    accepted = await store.add_episode(
        "I prefer plotly for figures",
        source_description="discord-chat",
        reference_time=moment,
        memory_class=MemoryClass.TOOL_PREFERENCE,
    )
    assert accepted is True
    assert len(store.episodes) == 1
    record = store.episodes[0]
    assert isinstance(record, EpisodeRecord)
    assert record.content == "I prefer plotly for figures"
    assert record.memory_class is MemoryClass.TOOL_PREFERENCE
    assert record.reference_time == moment
    facts = await store.search("plotly figures")
    assert len(facts) == 1
    assert facts[0].memory_class is MemoryClass.TOOL_PREFERENCE


async def test_blank_episode_is_not_accepted() -> None:
    """Whitespace-only content stores nothing."""

    store = FakeUserMemoryStore()
    assert await store.add_episode("   ") is False
    assert store.episodes == ()


async def test_episodes_property_is_read_only_snapshot() -> None:
    """The property cannot be rebound and returns an independent snapshot."""

    store = FakeUserMemoryStore()
    await store.add_episode("my goal is a postdoc at a good lab")
    snapshot = store.episodes
    assert isinstance(snapshot, tuple)
    with pytest.raises(AttributeError):
        store.episodes = ()  # type: ignore[misc]
    assert len(store.episodes) == 1


async def test_temporal_facts_exclude_past_invalid_at_by_default() -> None:
    """A fact invalidated in the past hides unless include_historical=True."""

    now = datetime.now(tz=UTC)
    store = FakeUserMemoryStore(
        [
            _fact("uses pytorch", invalid_at=now - timedelta(days=1)),
            _fact("uses jax", invalid_at=now + timedelta(days=365)),
            _fact("likes rust"),
        ]
    )
    default_hits = {f.fact for f in await store.search("uses")}
    assert default_hits == {"uses jax"}
    historical_hits = {
        f.fact for f in await store.search("uses", include_historical=True)
    }
    assert historical_hits == {"uses pytorch", "uses jax"}
    future_hits = {f.fact for f in await store.search("uses")}
    assert "uses jax" in future_hits


async def test_get_context_happy_path() -> None:
    """Healthy mode returns ranked facts, non-degraded and available."""

    store = FakeUserMemoryStore([_fact("studies gap detection in reviews")])
    context = await store.get_context("gap detection")
    assert isinstance(context, UserMemoryContext)
    assert context.backend == "fake"
    assert context.degraded is False
    assert context.available is True
    assert context.facts[0].fact == "studies gap detection in reviews"


async def test_outage_mode_simulates_total_backend_failure() -> None:
    """With fail=True nothing raises and everything degrades."""

    store = FakeUserMemoryStore([_fact("seeded fact about protein folding")], fail=True)
    assert await store.add_episode("should be rejected") is False
    assert store.episodes == ()
    assert await store.search("protein folding") == []
    context = await store.get_context("protein folding")
    assert context.backend == "fake"
    assert context.degraded is True
    assert context.facts == ()
    assert context.available is False
    await store.close()


async def test_outage_mode_status_is_unhealthy_with_fixed_detail() -> None:
    """The outage status reports enabled but unhealthy, never leaking content."""

    store = FakeUserMemoryStore(fail=True)
    status = await store.status()
    assert status.backend == "fake"
    assert status.enabled is True
    assert status.healthy is False
    assert status.detail == "simulated outage"
    assert "protein" not in status.detail.lower()


async def test_healthy_status_counts_episodes_without_secrets() -> None:
    """Healthy status carries the episode count and an empty detail."""

    store = FakeUserMemoryStore()
    await store.add_episode("I work on retrieval evaluation")
    status = await store.status()
    assert status.backend == "fake"
    assert status.enabled is True
    assert status.healthy is True
    assert status.detail == "" or len(status.detail) < 64
    assert status.episode_count == 1
