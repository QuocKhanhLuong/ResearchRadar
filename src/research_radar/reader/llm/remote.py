"""OpenAI-compatible remote structured-generation adapter.

The wire format is a widely implemented chat-completions convention, not an
endorsement of or dependency on any particular hosted model vendor.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import ValidationError

from research_radar.errors import LLMResponseError, LLMUnavailableError
from research_radar.reader.llm.base import LLMMessage, ModelT


class RemoteLLMProvider:
    """Call a configured OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        normalized_base_url = base_url.strip().rstrip("/")
        if not normalized_base_url:
            raise ValueError("base_url must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._endpoint = _chat_completions_endpoint(normalized_base_url)
        self._model = model.strip()
        self._api_key = api_key.strip() if api_key else None
        self._timeout = httpx.Timeout(timeout_seconds)
        self._client = client or httpx.AsyncClient(timeout=self._timeout)
        self._owns_client = client is None

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[ModelT],
    ) -> ModelT:
        """Request JSON-object output and validate it against ``response_model``."""

        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "messages": [message.as_request_payload() for message in messages],
            "response_format": {"type": "json_object"},
        }
        try:
            response = await self._client.post(
                self._endpoint,
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("The remote LLM request timed out.") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError("The remote LLM endpoint could not be reached.") from exc

        if not response.is_success:
            raise LLMUnavailableError(
                f"The remote LLM endpoint returned HTTP {response.status_code}."
            )

        try:
            raw_response = response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseError("The remote LLM returned invalid JSON.") from exc

        try:
            content = _extract_message_content(raw_response)
            structured_payload = _parse_structured_content(content)
            return response_model.model_validate(structured_payload)
        except (KeyError, TypeError, ValidationError, json.JSONDecodeError) as exc:
            raise LLMResponseError(
                "The remote LLM returned an invalid structured response."
            ) from exc

    async def aclose(self) -> None:
        """Close only the client created by this provider."""

        if self._owns_client:
            await self._client.aclose()


def _chat_completions_endpoint(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _extract_message_content(response: object) -> object:
    if not isinstance(response, Mapping):
        raise TypeError("response must be an object")
    choices = response["choices"]
    if not isinstance(choices, list) or not choices:
        raise TypeError("response choices must be a non-empty list")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise TypeError("response choice must be an object")
    message = first_choice["message"]
    if not isinstance(message, Mapping):
        raise TypeError("response message must be an object")
    return message["content"]


def _parse_structured_content(content: object) -> Mapping[str, Any]:
    if isinstance(content, Mapping):
        return content
    if not isinstance(content, str):
        raise TypeError("response message content must be a JSON object string")
    parsed = json.loads(content)
    if not isinstance(parsed, Mapping):
        raise TypeError("response message content must decode to an object")
    return parsed
