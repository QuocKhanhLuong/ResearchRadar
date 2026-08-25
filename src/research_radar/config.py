"""Application configuration loaded from environment variables and `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from research_radar.errors import ConfigurationError


class Settings(BaseSettings):
    """Runtime settings with safe defaults for local development and tests."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    discord_token: SecretStr | None = None
    discord_guild_id: int | None = None
    discord_channel_id: int | None = None

    database_url: str = "sqlite:///data/research_radar.db"

    openalex_email: str | None = None
    openalex_api_key: SecretStr | None = None
    semantic_scholar_api_key: SecretStr | None = None

    watch_scan_hours: int = Field(default=6, ge=1, le=168)
    digest_hour: int = Field(default=8, ge=0, le=23)
    timezone: str = "Asia/Bangkok"

    llm_provider: str = "mock"
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    http_timeout_seconds: float = Field(default=20.0, gt=0, le=120)

    artifact_root: str = "data/artifacts"

    embedding_provider: str = "disabled"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    semantic_index: str = "disabled"
    pinecone_api_key: SecretStr | None = None
    pinecone_index: str | None = None
    pinecone_namespace: str = "research-radar"

    ingestion_metadata_limit: int = Field(default=50, ge=1, le=200)

    @field_validator("embedding_provider")
    @classmethod
    def validate_embedding_provider(cls, value: str) -> str:
        """Reject unknown embedding backends before any service is composed."""

        normalized = value.strip().casefold()
        if normalized not in {"disabled", "local"}:
            raise ValueError("EMBEDDING_PROVIDER must be one of: disabled, local")
        return normalized

    @field_validator("semantic_index")
    @classmethod
    def validate_semantic_index(cls, value: str) -> str:
        """Reject unknown semantic index backends before any service is composed."""

        normalized = value.strip().casefold()
        if normalized not in {"disabled", "pinecone"}:
            raise ValueError("SEMANTIC_INDEX must be one of: disabled, pinecone")
        return normalized

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        """Reject unknown IANA zones before APScheduler configuration begins."""

        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Unknown IANA timezone: {value}") from error
        return value

    def artifact_root_path(self) -> Path:
        """Return the artifact root as an absolute path without creating it."""

        return Path(self.artifact_root).expanduser().resolve()

    def require_discord_token(self) -> str:
        """Return the token only for the explicit bot-launch path."""

        if self.discord_token is None:
            raise ConfigurationError(
                "DISCORD_TOKEN is required to launch the Discord bot. "
                "Copy .env.example to .env and configure it first."
            )
        return self.discord_token.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Return process-wide settings after the environment is loaded once."""

    return Settings()
