"""Tests for owner-restricted /memory-status and /memory-search commands."""

from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from research_radar.bot.commands.memory import (
    MAX_FACT_LINE_CHARS,
    MAX_SEARCH_DESCRIPTION_CHARS,
    OWNER_REFUSAL_MESSAGE,
    format_fact_line,
    register_memory_commands,
)
from research_radar.memory import (
    DisabledUserMemoryStore,
    FakeUserMemoryStore,
    MemoryClass,
    MemoryFact,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import inspect_user_memory  # noqa: E402

OWNER_ID = 123_456_789_012_345
STRANGER_ID = 999_888_777_666_555
CREDENTIAL_SHAPE = "sk-test-NOT-A-REAL-KEY-000000000"
GRAPHITI_STORE_MODULE = "research_radar.memory.graphiti_store"


@dataclass(frozen=True)
class FakeSettings:
    """Duck-typed Settings carrying exactly the fields the commands read."""

    discord_owner_user_id: int | None = None
    user_memory_max_results: int = 8


class FakeTree:
    """Captures command callbacks the way a real CommandTree would."""

    def __init__(self) -> None:
        self.commands: dict[str, Any] = {}

    def command(self, *, name: str, description: str | None = None):
        def decorator(callback: Any) -> Any:
            self.commands[name] = callback
            return callback

        return decorator


class FakeResponse:
    def __init__(self) -> None:
        self.send_message = AsyncMock()
        self.defer = AsyncMock()

    def is_done(self) -> bool:
        return False


class FakeInteraction:
    def __init__(self, user_id: int) -> None:
        self.user = SimpleNamespace(id=user_id)
        self.response = FakeResponse()
        self.followup = SimpleNamespace(send=AsyncMock())


class RecordingStore:
    """Wraps a store and records every backend access."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.status_calls = 0
        self.searches: list[tuple[str, int]] = []

    @property
    def backend_name(self) -> str:
        return self._inner.backend_name

    @property
    def enabled(self) -> bool:
        return self._inner.enabled

    async def search(self, query: str, *, limit: int = 8) -> list[MemoryFact]:
        self.searches.append((query, limit))
        return await self._inner.search(query, limit=limit)

    async def status(self) -> Any:
        self.status_calls += 1
        return await self._inner.status()


class CrashingStore:
    """Store whose every read raises with exception text containing a secret."""

    backend_name = "fake"
    enabled = True

    async def search(self, query: str, *, limit: int = 8) -> list[MemoryFact]:
        raise RuntimeError(f"search failed while dialing {CREDENTIAL_SHAPE}")

    async def status(self) -> Any:
        raise RuntimeError(f"connection failed for {CREDENTIAL_SHAPE}")


def registered_commands(store: Any, settings: FakeSettings | None = None) -> dict[str, Any]:
    tree = FakeTree()
    register_memory_commands(tree, store, settings or FakeSettings())
    return tree.commands


def embed_text(embed: discord.Embed) -> str:
    parts = [embed.title or "", embed.description or "", embed.footer.text or ""]
    for field in embed.fields:
        parts.append(field.name)
        parts.append(field.value or "")
    return "\n".join(parts)


def sent_text(interaction: FakeInteraction) -> str:
    parts: list[str] = []
    response = interaction.response.send_message.await_args_list
    followup = interaction.followup.send.await_args_list
    for call in (*response, *followup):
        embed = call.kwargs.get("embed")
        if embed is not None:
            parts.append(embed_text(embed))
        if call.kwargs.get("content"):
            parts.append(str(call.kwargs["content"]))
    return "\n".join(parts)


def credential_fact() -> MemoryFact:
    return MemoryFact(
        fact=f"my deploy key is {CREDENTIAL_SHAPE} rotate it monthly",
        memory_class=MemoryClass.TOOL_PREFERENCE,
        valid_at=datetime(2026, 8, 1, tzinfo=UTC),
        source="internal-graph-source",
        score=0.42,
    )


async def test_non_owner_gets_ephemeral_refusal_and_backend_is_never_queried() -> None:
    store = RecordingStore(FakeUserMemoryStore([credential_fact()]))
    commands = registered_commands(store, FakeSettings(discord_owner_user_id=OWNER_ID))

    status_interaction = FakeInteraction(STRANGER_ID)
    await commands["memory-status"](status_interaction)
    search_interaction = FakeInteraction(STRANGER_ID)
    await commands["memory-search"](search_interaction, query="deploy key")

    assert store.status_calls == 0
    assert store.searches == []
    status_interaction.response.send_message.assert_awaited_once_with(
        OWNER_REFUSAL_MESSAGE, ephemeral=True
    )
    search_interaction.response.send_message.assert_awaited_once_with(
        OWNER_REFUSAL_MESSAGE, ephemeral=True
    )
    status_interaction.response.defer.assert_not_awaited()
    search_interaction.response.defer.assert_not_awaited()


async def test_configured_owner_can_query_both_commands() -> None:
    settings = FakeSettings(discord_owner_user_id=OWNER_ID, user_memory_max_results=3)
    store = RecordingStore(FakeUserMemoryStore([credential_fact()]))
    commands = registered_commands(store, settings)

    status_interaction = FakeInteraction(OWNER_ID)
    await commands["memory-status"](status_interaction)
    assert store.status_calls == 1
    status_interaction.response.defer.assert_awaited_once_with(thinking=True, ephemeral=True)
    assert status_interaction.followup.send.call_args.kwargs["ephemeral"] is True

    search_interaction = FakeInteraction(OWNER_ID)
    await commands["memory-search"](search_interaction, query="deploy key")
    assert store.searches == [("deploy key", 3)]
    assert search_interaction.followup.send.call_args.kwargs["ephemeral"] is True


async def test_unconfigured_owner_id_allows_any_caller() -> None:
    store = RecordingStore(FakeUserMemoryStore())
    commands = registered_commands(store, FakeSettings())

    interaction = FakeInteraction(STRANGER_ID)
    await commands["memory-status"](interaction)

    assert store.status_calls == 1
    interaction.response.send_message.assert_not_awaited()


async def test_disabled_backend_reports_disabled_plainly() -> None:
    commands = registered_commands(DisabledUserMemoryStore(), FakeSettings())

    status_interaction = FakeInteraction(OWNER_ID)
    await commands["memory-status"](status_interaction)
    embed = status_interaction.followup.send.call_args.kwargs["embed"]
    fields = {field.name: field.value for field in embed.fields}

    assert "disabled" in embed.description.lower()
    assert fields["Enabled"] == "no"
    assert "Persistence path" not in fields
    assert "Episodes" not in fields

    search_interaction = FakeInteraction(OWNER_ID)
    await commands["memory-search"](search_interaction, query="anything")
    search_embed = search_interaction.followup.send.call_args.kwargs["embed"]
    assert search_embed.description == "No matching facts."


async def test_backend_outage_reports_plainly_without_traceback_or_detail() -> None:
    commands = registered_commands(FakeUserMemoryStore(fail=True), FakeSettings())

    status_interaction = FakeInteraction(OWNER_ID)
    await commands["memory-status"](status_interaction)
    payload = sent_text(status_interaction)

    assert "unavailable" in payload.lower()
    assert "Traceback" not in payload
    assert "simulated outage" not in payload

    search_interaction = FakeInteraction(OWNER_ID)
    await commands["memory-search"](search_interaction, query="anything")
    search_payload = sent_text(search_interaction)
    # An outage surfaces as an ordinary empty result: stores never raise.
    assert "no matching facts" in search_payload.lower()
    assert "Traceback" not in search_payload


async def test_raising_store_never_leaks_exception_text_to_discord() -> None:
    commands = registered_commands(CrashingStore(), FakeSettings())

    status_interaction = FakeInteraction(OWNER_ID)
    await commands["memory-status"](status_interaction)
    payload = sent_text(status_interaction)

    assert CREDENTIAL_SHAPE not in payload
    assert "Traceback" not in payload
    assert "connection failed" not in payload

    search_interaction = FakeInteraction(OWNER_ID)
    await commands["memory-search"](search_interaction, query="key")
    assert CREDENTIAL_SHAPE not in sent_text(search_interaction)


async def test_status_output_never_contains_stored_fact_content() -> None:
    commands = registered_commands(FakeUserMemoryStore([credential_fact()]), FakeSettings())

    interaction = FakeInteraction(OWNER_ID)
    await commands["memory-status"](interaction)
    payload = sent_text(interaction)

    assert CREDENTIAL_SHAPE not in payload
    assert "deploy key" not in payload


async def test_search_output_renders_only_whitelisted_fact_fields() -> None:
    fact = MemoryFact(
        fact="likes terse replies in direct messages",
        memory_class=MemoryClass.WORKFLOW_PREFERENCE,
        valid_at=datetime(2026, 8, 1, tzinfo=UTC),
        invalid_at=datetime(2026, 12, 1, tzinfo=UTC),
        source="internal-graph-source",
        score=0.42,
    )
    commands = registered_commands(FakeUserMemoryStore([fact]), FakeSettings())

    interaction = FakeInteraction(OWNER_ID)
    await commands["memory-search"](interaction, query="terse replies")
    text = sent_text(interaction)

    assert "[workflow_preference]" in text
    assert "valid from 2026-08-01; until 2026-12-01" in text
    for banned in ("internal-graph-source", "0.42", "score", "source", "uuid", "group_id"):
        assert banned not in text, banned


async def test_credential_shaped_fact_renders_as_an_ordinary_whitelisted_line_only() -> None:
    commands = registered_commands(FakeUserMemoryStore([credential_fact()]), FakeSettings())

    interaction = FakeInteraction(OWNER_ID)
    await commands["memory-search"](interaction, query="deploy key")
    embed = interaction.followup.send.call_args.kwargs["embed"]

    lines = [line for line in (embed.description or "").splitlines() if line]
    assert len(lines) == 1
    line_pattern = (
        r"^• \[[a-z_]+\] .+?"
        r"( \(valid from \d{4}-\d{2}-\d{2}(; until \d{4}-\d{2}-\d{2})?\))?$"
    )
    for line in lines:
        assert re.match(line_pattern, line), line
    assert "Persistence path" not in embed_text(embed)


def test_format_fact_line_truncates_and_drops_nonwhitelisted_fields() -> None:
    fact = MemoryFact(
        fact="y" * 900,
        memory_class=MemoryClass.GOAL,
        valid_at=datetime(2026, 8, 1, tzinfo=UTC),
        invalid_at=datetime(2026, 9, 1, tzinfo=UTC),
        source="hidden-source",
        score=0.9,
    )

    line = format_fact_line(fact)

    assert line.startswith("• [goal] ")
    assert "valid from 2026-08-01" in line
    assert "until 2026-09-01" in line
    assert len(line) <= MAX_FACT_LINE_CHARS
    assert "…" in line
    assert "hidden-source" not in line
    assert "0.9" not in line


async def test_long_facts_are_truncated_and_total_length_is_capped() -> None:
    facts = [
        MemoryFact(fact=f"zebra {index} " + "y" * 450, memory_class=MemoryClass.GOAL)
        for index in range(20)
    ]
    commands = registered_commands(
        FakeUserMemoryStore(facts), FakeSettings(user_memory_max_results=50)
    )

    interaction = FakeInteraction(OWNER_ID)
    await commands["memory-search"](interaction, query="zebra")
    embed = interaction.followup.send.call_args.kwargs["embed"]
    description = embed.description or ""

    assert len(description) <= MAX_SEARCH_DESCRIPTION_CHARS + 64
    assert all(len(line) <= MAX_FACT_LINE_CHARS for line in description.splitlines())
    assert "more omitted" in description


async def test_expired_interaction_aborts_before_any_backend_call() -> None:
    store = RecordingStore(FakeUserMemoryStore())
    commands = registered_commands(store, FakeSettings())
    http_response = MagicMock(status=404, reason="Not Found")
    interaction = FakeInteraction(OWNER_ID)
    interaction.response.defer = AsyncMock(
        side_effect=discord.NotFound(
            http_response, {"code": 10062, "message": "Unknown interaction"}
        )
    )

    await commands["memory-status"](interaction)

    assert store.status_calls == 0
    interaction.followup.send.assert_not_awaited()


def test_script_builds_disabled_store_and_inspection_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = SimpleNamespace(user_memory_backend="disabled")
    store = inspect_user_memory.build_store_from_settings(settings)

    exit_code = asyncio.run(inspect_user_memory.inspect_memory(store, query=None, limit=8))

    out = capsys.readouterr().out
    assert isinstance(store, DisabledUserMemoryStore)
    assert "user memory is disabled" in out
    assert exit_code == 0


def test_script_main_prints_disabled_notice_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        inspect_user_memory,
        "get_settings",
        lambda: SimpleNamespace(user_memory_backend="disabled", user_memory_max_results=5),
    )

    with pytest.raises(SystemExit) as excinfo:
        inspect_user_memory.main(["--query", "preferences"])

    assert excinfo.value.code == 0
    assert "user memory is disabled" in capsys.readouterr().out


def test_script_query_output_prints_only_whitelisted_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fact = MemoryFact(
        fact="likes quiet tooling",
        memory_class=MemoryClass.TOOL_PREFERENCE,
        valid_at=datetime(2026, 8, 1, tzinfo=UTC),
        source="internal-graph-source",
        score=0.7,
    )
    store = FakeUserMemoryStore([fact])

    exit_code = asyncio.run(inspect_user_memory.inspect_memory(store, query="quiet", limit=5))

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "likes quiet tooling" in out
    assert "[tool_preference]" in out
    assert "valid from 2026-08-01" in out
    for banned in ("internal-graph-source", "score", "0.7", "group_id", "object at 0x"):
        assert banned not in out, banned


def test_script_graphiti_import_error_degrades_to_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, GRAPHITI_STORE_MODULE, None)
    settings = SimpleNamespace(user_memory_backend="graphiti")

    store = inspect_user_memory.build_store_from_settings(settings)
    exit_code = asyncio.run(inspect_user_memory.inspect_memory(store, query=None, limit=8))

    out = capsys.readouterr().out
    assert "personal memory is unavailable" in out
    assert exit_code == 0


def test_script_graphiti_construction_failure_degrades_to_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(settings: object) -> object:
        raise RuntimeError("no driver available")

    fake_module = SimpleNamespace(GraphitiUserMemoryStore=_boom)
    monkeypatch.setitem(sys.modules, GRAPHITI_STORE_MODULE, fake_module)
    settings = SimpleNamespace(user_memory_backend="graphiti")

    store = inspect_user_memory.build_store_from_settings(settings)
    exit_code = asyncio.run(inspect_user_memory.inspect_memory(store, query=None, limit=8))

    out = capsys.readouterr().out
    assert "personal memory is unavailable" in out
    assert exit_code == 0
