"""Focused unit tests for thin Discord command adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from research_radar.bot.commands.digest import register_digest_command
from research_radar.bot.commands.read import register_read_command
from research_radar.bot.commands.watch import register_watch_commands
from research_radar.errors import LLMUnavailableError, PaperParseError


class _FakeTree:
    def __init__(self) -> None:
        self.commands: list[Any] = []

    def add_command(self, command: Any) -> None:
        self.commands.append(command)


class _FakeInteraction:
    def __init__(self) -> None:
        self.response = SimpleNamespace(defer=AsyncMock())
        self.edit_original_response = AsyncMock()


@dataclass(frozen=True)
class _Topic:
    id: str
    name: str
    query: str
    enabled: bool = True
    last_scan_at: datetime | None = None


class _WatchService:
    def __init__(self, topics: list[_Topic] | None = None) -> None:
        self.topics = topics or []
        self.add_calls: list[tuple[str, str]] = []
        self.remove_calls: list[str] = []

    async def add_topic(self, name: str, query: str) -> _Topic:
        self.add_calls.append((name, query))
        return _Topic("watch-1", name, query)

    async def list_topics(self) -> list[_Topic]:
        return self.topics

    async def remove_topic(self, topic_id_or_name: str) -> bool:
        self.remove_calls.append(topic_id_or_name)
        return topic_id_or_name == "watch-1"


class _ReaderService:
    def __init__(self, outcome: object | Exception) -> None:
        self.outcome = outcome
        self.urls: list[str] = []

    async def read_url(self, url: str) -> object:
        self.urls.append(url)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class _Digest:
    def __init__(self, content: str) -> None:
        self.content = content

    def render_text(self) -> str:
        return self.content


class _DigestService:
    def __init__(self, outcome: _Digest | Exception) -> None:
        self.outcome = outcome
        self.calls = 0

    async def build_on_demand(self) -> _Digest:
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _callbacks(tree: _FakeTree) -> dict[str, Any]:
    command = tree.commands[0]
    return {child.name: child.callback for child in command.commands}


@pytest.mark.asyncio
async def test_watch_add_defers_then_uses_injected_service() -> None:
    tree = _FakeTree()
    service = _WatchService()
    register_watch_commands(tree, service)
    interaction = _FakeInteraction()

    await _callbacks(tree)["add"](interaction, "Medical MRI", "sparse MRI reconstruction")

    interaction.response.defer.assert_awaited_once_with(thinking=True)
    assert service.add_calls == [("Medical MRI", "sparse MRI reconstruction")]
    assert "Now watching **Medical MRI**" in interaction.edit_original_response.call_args.kwargs[
        "content"
    ]


@pytest.mark.asyncio
async def test_watch_list_renders_saved_topics_without_domain_imports() -> None:
    tree = _FakeTree()
    service = _WatchService(
        [
            _Topic(
                "watch-1",
                "Medical MRI",
                "sparse MRI reconstruction",
                last_scan_at=datetime(2026, 8, 13, 8, 0, tzinfo=UTC),
            ),
            _Topic("watch-2", "Paused", "anomaly detection", enabled=False),
        ]
    )
    register_watch_commands(tree, service)
    interaction = _FakeInteraction()

    await _callbacks(tree)["list"](interaction)

    content = interaction.edit_original_response.call_args.kwargs["content"]
    assert "**Research watchlist**" in content
    assert "Medical MRI" in content
    assert "(paused)" in content
    assert "2026-08-13T08:00+00:00" in content


@pytest.mark.asyncio
async def test_watch_remove_reports_a_missing_topic() -> None:
    tree = _FakeTree()
    service = _WatchService()
    register_watch_commands(tree, service)
    interaction = _FakeInteraction()

    await _callbacks(tree)["remove"](interaction, "does-not-exist")

    assert service.remove_calls == ["does-not-exist"]
    assert interaction.edit_original_response.call_args.kwargs["content"] == (
        "No matching watch topic was found."
    )


@pytest.mark.asyncio
async def test_read_defers_and_renders_a_validated_card() -> None:
    tree = _FakeTree()
    result = SimpleNamespace(
        paper=SimpleNamespace(title="Bounded Paper", canonical_link="https://papers.example/paper.pdf"),
        card=SimpleNamespace(
            problem="Reliable reading",
            contributions=["Bounded extraction"],
            methods=["Heuristic parsing"],
            datasets=["Demo set"],
            main_claims=[
                SimpleNamespace(
                    claim="Extraction is predictable",
                    source_section="Results",
                )
            ],
            limitations=["Scanned PDFs are unsupported"],
            future_work=["Add OCR later"],
        ),
    )
    service = _ReaderService(result)
    register_read_command(tree, service)
    interaction = _FakeInteraction()

    await tree.commands[0].callback(interaction, "https://papers.example/paper.pdf")

    interaction.response.defer.assert_awaited_once_with(thinking=True)
    assert service.urls == ["https://papers.example/paper.pdf"]
    embed = interaction.edit_original_response.call_args.kwargs["embed"]
    assert embed.title == "Read: Bounded Paper"
    assert embed.url == "https://papers.example/paper.pdf"
    fields = {field.name: field.value for field in embed.fields}
    assert fields["Problem"] == "Reliable reading"
    assert "Extraction is predictable (Results)" in fields["Main claims"]


@pytest.mark.asyncio
async def test_read_maps_parse_and_unavailable_model_errors_to_concise_messages() -> None:
    tree = _FakeTree()
    service = _ReaderService(PaperParseError("technical parser detail"))
    register_read_command(tree, service)
    interaction = _FakeInteraction()

    await tree.commands[0].callback(interaction, "https://papers.example/bad.pdf")

    assert "couldn't extract readable text" in interaction.edit_original_response.call_args.kwargs[
        "content"
    ]

    tree = _FakeTree()
    service = _ReaderService(LLMUnavailableError("credential detail"))
    register_read_command(tree, service)
    interaction = _FakeInteraction()

    await tree.commands[0].callback(interaction, "https://papers.example/paper.pdf")

    content = interaction.edit_original_response.call_args.kwargs["content"]
    assert "structured analysis is unavailable" in content


@pytest.mark.asyncio
async def test_digest_defers_and_uses_the_shared_digest_renderer() -> None:
    tree = _FakeTree()
    service = _DigestService(_Digest("ResearchRadar Daily Digest\n\nNew papers discovered: 2"))
    register_digest_command(tree, service)
    interaction = _FakeInteraction()

    await tree.commands[0].callback(interaction)

    interaction.response.defer.assert_awaited_once_with(thinking=True)
    assert service.calls == 1
    assert interaction.edit_original_response.call_args.kwargs["content"] == (
        "ResearchRadar Daily Digest\n\nNew papers discovered: 2"
    )


@pytest.mark.asyncio
async def test_digest_hides_internal_service_failure_details() -> None:
    tree = _FakeTree()
    service = _DigestService(RuntimeError("database path and secret-like detail"))
    register_digest_command(tree, service)
    interaction = _FakeInteraction()

    await tree.commands[0].callback(interaction)

    content = interaction.edit_original_response.call_args.kwargs["content"]
    assert content == "The research digest is temporarily unavailable. Please try again later."
    assert "secret-like" not in content
