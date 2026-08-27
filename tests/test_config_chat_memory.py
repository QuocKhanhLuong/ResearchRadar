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


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("", ()),
        ("   ", ()),
        (",", ()),
        (" , , ", ()),
        ("100,,200", (100, 200)),
        (" 100 , 200 , 300 ", (100, 200, 300)),
    ],
)
def test_discord_allowed_channel_ids_blank_and_sparse_csv(
    monkeypatch: pytest.MonkeyPatch, env_value: str, expected: tuple[int, ...]
) -> None:
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNEL_IDS", env_value)

    settings = Settings(_env_file=None)

    assert settings.discord_allowed_channel_ids == expected


def test_discord_allowed_channel_ids_direct_collection_input() -> None:
    settings_list = Settings(discord_allowed_channel_ids=[10, 20], _env_file=None)
    assert settings_list.discord_allowed_channel_ids == (10, 20)

    settings_tuple = Settings(discord_allowed_channel_ids=(30, 40), _env_file=None)
    assert settings_tuple.discord_allowed_channel_ids == (30, 40)

    settings_str_list = Settings(discord_allowed_channel_ids=["50", "60"], _env_file=None)
    assert settings_str_list.discord_allowed_channel_ids == (50, 60)

    with pytest.raises(ValidationError, match="DISCORD_ALLOWED_CHANNEL_IDS"):
        Settings(discord_allowed_channel_ids=["not_an_int"], _env_file=None)



def test_build_application_bot_constructs_offline_safely(tmp_path: Path) -> None:
    from research_radar.main import build_application_bot
    from research_radar.memory.disabled import DisabledUserMemoryStore

    settings = Settings(
        database_url=f"sqlite:///{tmp_path}/test.db",
        artifact_root=str(tmp_path / "artifacts"),
        user_memory_db_path=str(tmp_path / "memory"),
        _env_file=None,
    )

    bot = build_application_bot(settings)
    assert bot is not None
    assert isinstance(bot._chat_service._user_memory, DisabledUserMemoryStore)

