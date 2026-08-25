"""Tests for chat and personal-memory settings in `research_radar.config`."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from research_radar.config import Settings

NEW_SETTINGS_DEFAULTS: dict[str, object] = {
    "discord_owner_user_id": None,
    "discord_allowed_channel_ids": (),
    "discord_chat_on_mention": True,
    "discord_dm_chat": True,
    "user_memory_backend": "disabled",
    "user_memory_db_path": "data/user_memory",
    "user_memory_group_id": "primary-user",
    "user_memory_max_results": 8,
    "user_memory_capture": True,
    "chat_live_discovery_limit": 10,
}


def test_new_settings_have_working_defaults() -> None:
    settings = Settings(_env_file=None)

    for name, expected in NEW_SETTINGS_DEFAULTS.items():
        assert getattr(settings, name) == expected, name


def test_empty_env_values_do_not_crash_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in NEW_SETTINGS_DEFAULTS:
        monkeypatch.setenv(name.upper(), "")

    settings = Settings(_env_file=None)

    for name, expected in NEW_SETTINGS_DEFAULTS.items():
        assert getattr(settings, name) == expected, name


def test_discord_allowed_channel_ids_absent_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISCORD_ALLOWED_CHANNEL_IDS", raising=False)

    settings = Settings(_env_file=None)

    assert settings.discord_allowed_channel_ids == ()


def test_discord_allowed_channel_ids_parses_csv_with_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNEL_IDS", " 1, 2 ,3 ")

    settings = Settings(_env_file=None)

    assert settings.discord_allowed_channel_ids == (1, 2, 3)


def test_discord_allowed_channel_ids_rejects_non_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNEL_IDS", "abc")

    with pytest.raises(ValidationError, match="DISCORD_ALLOWED_CHANNEL_IDS"):
        Settings(_env_file=None)


def test_user_memory_backend_normalizes_case_and_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USER_MEMORY_BACKEND", "  Graphiti ")

    settings = Settings(_env_file=None)

    assert settings.user_memory_backend == "graphiti"


def test_user_memory_backend_rejects_unknown_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USER_MEMORY_BACKEND", "neo4j")

    with pytest.raises(ValidationError, match="USER_MEMORY_BACKEND must be one of"):
        Settings(_env_file=None)


def test_user_memory_db_path_resolved_is_absolute_and_creates_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "never-created" / "nested"
    monkeypatch.setenv("USER_MEMORY_DB_PATH", str(target))

    settings = Settings(_env_file=None)

    resolved = settings.user_memory_db_path_resolved()
    assert resolved.is_absolute()
    assert resolved == target.expanduser().resolve()
    assert not resolved.exists()
    assert not (tmp_path / "never-created").exists()
