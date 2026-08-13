"""Deterministic LLM test double that is unavailable by default."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from research_radar.errors import LLMUnavailableError
from research_radar.reader.llm.base import LLMMessage, ModelT


class MockLLMProvider:
    """Return an explicitly supplied fixture, never an invented analysis.

    Leaving ``response`` unset intentionally models the normal application
    default (``LLM_PROVIDER=mock``): callers receive a clear unavailable error
    instead of synthetic paper claims.
    """

    def __init__(self, response: BaseModel | dict[str, Any] | None = None) -> None:
        self._response = response
        self.requests: list[list[LLMMessage]] = []

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[ModelT],
    ) -> ModelT:
        self.requests.append(list(messages))
        if self._response is None:
            raise LLMUnavailableError(
                "No language model is configured. "
                "Configure a remote LLM provider to analyze papers."
            )

        payload: BaseModel | dict[str, Any]
        if isinstance(self._response, BaseModel):
            payload = self._response.model_dump(mode="json")
        else:
            payload = self._response
        return response_model.model_validate(payload)
