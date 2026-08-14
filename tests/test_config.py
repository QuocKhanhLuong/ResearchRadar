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
