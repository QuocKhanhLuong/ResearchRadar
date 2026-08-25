"""Focused unit tests for the /ingest command adapter and the store inspector."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from research_radar.bot.commands.ingest import (
    PROVIDERS_UNAVAILABLE_MESSAGE,
    register_ingest_command,
)
from research_radar.errors import ProviderUnavailableError
from research_radar.models.paper import Paper
from research_radar.storage.database import create_database
from research_radar.storage.repositories import ResearchRepository

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inspect_research_store import inspect_store, table_counts  # noqa: E402


class _FakeTree:
    def __init__(self) -> None:
        self.commands: list[Any] = []

    def add_command(self, command: Any) -> None:
        self.commands.append(command)


class _FakeInteraction:
    def __init__(self) -> None:
        self.response = SimpleNamespace(defer=AsyncMock())
        self.edit_original_response = AsyncMock()


class _IngestService:
    def __init__(self, outcome: object | Exception) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    async def ingest_research_topic(
        self,
        query: str,
        *,
        limit: int = 20,
        project_id: str | None = None,
        auto_read: int = 0,
    ) -> object:
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "project_id": project_id,
                "auto_read": auto_read,
            }
        )
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _success_result(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "run_id": "run-123",
        "query": "sparse MRI reconstruction",
        "discovered_count": 7,
        "canonical_count": 5,
        "paper_ids": ["p-1", "p-2"],
        "warnings": [],
        "provider_counts": {"arxiv": 4, "openalex": 3},
        "read_paper_ids": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_register_ingest_adds_exactly_one_command_named_ingest() -> None:
    tree = _FakeTree()
    register_ingest_command(tree, _IngestService(_success_result()))

    assert len(tree.commands) == 1
    assert tree.commands[0].name == "ingest"


@pytest.mark.asyncio
async def test_successful_ingest_edits_response_and_forwards_count_as_limit() -> None:
    tree = _FakeTree()
    service = _IngestService(_success_result())
    register_ingest_command(tree, service)
    interaction = _FakeInteraction()

    await tree.commands[0].callback(
        interaction, query="sparse MRI reconstruction", count=5, project=None
    )

    interaction.response.defer.assert_awaited_once_with(thinking=True)
    assert interaction.edit_original_response.await_count == 1
    assert service.calls[0]["limit"] == 5
    assert service.calls[0]["project_id"] is None

    call = interaction.edit_original_response.call_args.kwargs
    embed = call["embed"]
    fields = {field.name: field.value for field in embed.fields}
    assert fields["Canonical papers stored"] == "5"
    assert fields["Raw records discovered"] == "7"
    assert "arxiv: 4" in fields["Provider results"]
    assert embed.footer.text is not None and "run-123" in embed.footer.text
    assert not any("p-1" in value for value in fields.values())


@pytest.mark.asyncio
async def test_command_never_passes_a_nonzero_auto_read() -> None:
    tree = _FakeTree()
    service = _IngestService(_success_result())
    register_ingest_command(tree, service)
    interaction = _FakeInteraction()

    await tree.commands[0].callback(interaction, query="any topic", count=50, project="proj")

    assert len(service.calls) == 1
    assert service.calls[0]["auto_read"] == 0


@pytest.mark.asyncio
async def test_value_error_from_service_produces_error_text_without_traceback() -> None:
    tree = _FakeTree()
    service = _IngestService(ValueError("Query must be between 1 and 300 characters."))
    register_ingest_command(tree, service)
    interaction = _FakeInteraction()

    await tree.commands[0].callback(interaction, query="", count=5, project=None)

    content = interaction.edit_original_response.call_args.kwargs["content"]
    assert content == "Query must be between 1 and 300 characters."
    assert "Traceback" not in content


@pytest.mark.asyncio
async def test_provider_unavailable_produces_friendly_unavailable_message() -> None:
    tree = _FakeTree()
    service = _IngestService(ProviderUnavailableError("semantic scholar timeout"))
    register_ingest_command(tree, service)
    interaction = _FakeInteraction()

    await tree.commands[0].callback(interaction, query="topic", count=5, project=None)

    content = interaction.edit_original_response.call_args.kwargs["content"]
    assert content == PROVIDERS_UNAVAILABLE_MESSAGE


@pytest.mark.asyncio
async def test_expired_interaction_aborts_before_calling_the_service() -> None:
    tree = _FakeTree()
    service = _IngestService(_success_result())
    register_ingest_command(tree, service)

    resp = MagicMock(status=404, reason="Not Found")
    interaction = SimpleNamespace(
        id=123,
        response=SimpleNamespace(
            defer=AsyncMock(
                side_effect=discord.NotFound(
                    resp, {"code": 10062, "message": "Unknown interaction"}
                )
            ),
            is_done=lambda: False,
        ),
        edit_original_response=AsyncMock(),
    )

    await tree.commands[0].callback(interaction, query="t", count=5, project=None)

    assert service.calls == []
    interaction.edit_original_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_result_warnings_appear_in_the_response_content() -> None:
    warning = "openalex was unreachable; results are partial"
    tree = _FakeTree()
    service = _IngestService(_success_result(warnings=[warning]))
    register_ingest_command(tree, service)
    interaction = _FakeInteraction()

    await tree.commands[0].callback(interaction, query="topic", count=5, project=None)

    content = interaction.edit_original_response.call_args.kwargs["content"]
    assert warning in content


@pytest.mark.asyncio
async def test_response_never_contains_traceback_or_secrets_from_exceptions() -> None:
    fake_key = "sk-test-SECRET-API-KEY-000"
    tree = _FakeTree()
    service = _IngestService(ProviderUnavailableError(f"auth failed for {fake_key}"))
    register_ingest_command(tree, service)
    interaction = _FakeInteraction()

    await tree.commands[0].callback(interaction, query="topic", count=5, project=None)

    content = interaction.edit_original_response.call_args.kwargs["content"]
    assert "Traceback" not in content
    assert fake_key not in content
    assert content == PROVIDERS_UNAVAILABLE_MESSAGE


def test_table_counts_are_zero_on_a_fresh_database(tmp_path: Path) -> None:
    db = create_database(f"sqlite:///{tmp_path / 'fresh.db'}")
    db.initialize_schema()

    counts = dict(table_counts(db))

    expected_labels = {
        "papers",
        "paper_sources",
        "paper_cards",
        "document_artifacts",
        "ingestion_runs",
        "provider_retrievals",
        "projects",
        "project_papers",
        "gap_candidates",
        "critic_reviews",
    }
    assert set(counts) == expected_labels
    assert all(value == 0 for value in counts.values())

    repository = ResearchRepository(db)
    stored_id = repository.upsert_merged_paper(
        Paper(
            id="demo-paper-1",
            title="Spectral Regularization for Robust MRI Reconstruction",
            publication_year=2024,
            doi="10.1234/demo.001",
            source="arxiv",
        )
    )

    counts_after = dict(table_counts(db))
    assert counts_after["papers"] == 1
    assert counts_after["paper_sources"] >= 1
    assert counts_after["ingestion_runs"] == 0
    assert len(stored_id) > 0
    db.dispose()


def test_unknown_paper_id_takes_not_found_path_with_exit_code_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = create_database(f"sqlite:///{tmp_path / 'empty.db'}")
    db.initialize_schema()

    exit_code = inspect_store(db, paper_id="no-such-paper")

    assert exit_code == 1
    assert "not found" in capsys.readouterr().out
    db.dispose()
