from pathlib import Path

import pytest

from research_radar.config import Settings
from research_radar.errors import ConfigurationError


def test_settings_have_safe_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.database_url == "sqlite:///data/research_radar.db"
    assert settings.llm_provider == "mock"
    assert settings.watch_scan_hours == 6
    assert settings.http_timeout_seconds == 20


def test_discord_token_is_required_only_when_requested() -> None:
    settings = Settings(discord_token=None, _env_file=None)

    try:
        settings.require_discord_token()
    except ConfigurationError as error:
        assert "DISCORD_TOKEN" in str(error)
    else:  # pragma: no cover - defensive assertion for clearer failures
        raise AssertionError("Expected missing Discord token to be rejected")


def test_settings_reject_an_unknown_timezone() -> None:
    with pytest.raises(ValueError, match="Unknown IANA timezone"):
        Settings(timezone="Not/A_Zone")


def test_env_example_covers_all_settings_fields() -> None:
    env_example_path = Path(__file__).resolve().parent.parent / ".env.example"
    assert env_example_path.exists(), ".env.example must exist at repo root"

    env_vars: set[str] = set()
    for line in env_example_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            var_name = line.split("=", 1)[0].strip()
            if var_name:
                env_vars.add(var_name.lower())

    settings_fields = set(Settings.model_fields.keys())
    missing_in_example = settings_fields - env_vars
    assert not missing_in_example, (
        f".env.example is missing configuration fields: {missing_in_example}"
    )


def test_loading_from_env_example_template_gives_working_defaults(
    tmp_path: Path,
) -> None:
    env_example_path = Path(__file__).resolve().parent.parent / ".env.example"
    temp_env = tmp_path / ".env"
    temp_env.write_text(env_example_path.read_text(encoding="utf-8"), encoding="utf-8")

    settings = Settings(_env_file=temp_env)

    assert settings.database_url == "sqlite:///data/research_radar.db"
    assert settings.artifact_root == "data/artifacts"
    assert settings.llm_provider == "mock"
    assert settings.embedding_provider == "disabled"
    assert settings.semantic_index == "disabled"
    assert settings.user_memory_backend == "disabled"
    assert settings.discord_token is None
    assert settings.discord_allowed_channel_ids == ()


def test_all_settings_empty_env_vars_produce_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for field_name in Settings.model_fields:
        monkeypatch.setenv(field_name.upper(), "")

    settings = Settings(_env_file=None)

    assert settings.database_url == "sqlite:///data/research_radar.db"
    assert settings.artifact_root == "data/artifacts"
    assert settings.llm_provider == "mock"
    assert settings.embedding_provider == "disabled"
    assert settings.semantic_index == "disabled"
    assert settings.user_memory_backend == "disabled"
    assert settings.discord_token is None
    assert settings.discord_allowed_channel_ids == ()


def test_embedding_provider_validation() -> None:
    assert Settings(embedding_provider="disabled", _env_file=None).embedding_provider == "disabled"
    assert Settings(embedding_provider="LOCAL", _env_file=None).embedding_provider == "local"

    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER must be one of"):
        Settings(embedding_provider="invalid_provider", _env_file=None)


def test_semantic_index_validation() -> None:
    assert Settings(semantic_index="disabled", _env_file=None).semantic_index == "disabled"
    assert Settings(semantic_index="PINECONE", _env_file=None).semantic_index == "pinecone"

    with pytest.raises(ValueError, match="SEMANTIC_INDEX must be one of"):
        Settings(semantic_index="invalid_index", _env_file=None)


def test_artifact_root_path_resolved(tmp_path: Path) -> None:
    target = tmp_path / "artifacts" / "nested"
    settings = Settings(artifact_root=str(target), _env_file=None)

    resolved = settings.artifact_root_path()
    assert resolved.is_absolute()
    assert resolved == target.expanduser().resolve()
    assert not resolved.exists()

