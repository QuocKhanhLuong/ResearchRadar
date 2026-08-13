"""Vendor-neutral structured language-model contracts."""

from __future__ import annotations

from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from research_radar.errors import ResearchRadarError

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMMessage(BaseModel):
    """A minimal chat message independent of any model vendor SDK."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    def as_request_payload(self) -> dict[str, str]:
        """Return the common chat-completions representation."""

        return {"role": self.role, "content": self.content}


class LLMProvider(Protocol):
    """Generate a Pydantic-validated structured response from chat messages."""

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[ModelT],
    ) -> ModelT:
        """Generate and validate one structured result."""


class LLMResponseError(ResearchRadarError):
    """Raised when a reachable model endpoint returns unusable structured data."""
