"""OpenAI-compatible remote structured-generation adapter.

The wire format is a widely implemented chat-completions convention, not an
endorsement of or dependency on any particular hosted model vendor.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import ValidationError

from research_radar.errors import LLMResponseError, LLMUnavailableError
from research_radar.reader.llm.base import LLMMessage, ModelT
from research_radar.reader.llm.telemetry import LLMOperation, UsageSink, parse_usage

logger = logging.getLogger(__name__)

_RESPONSE_FORMAT_FALLBACK_INSTRUCTION = (
    "Respond with a single valid JSON object and nothing else. Do not use markdown fences."
)


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
        usage_sink: UsageSink | None = None,
        provider_name: str = "remote",
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
        self._usage_sink = usage_sink
        self._provider_name = provider_name

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[ModelT],
        *,
        operation: LLMOperation = "other",
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
        response = await self._send(payload, headers)

        if (
            response.status_code in (400, 422)
            and _rejects_response_format(response)
        ):
            payload = {
                "model": self._model,
                "messages": [
                    *[message.as_request_payload() for message in messages],
                    {"role": "system", "content": _RESPONSE_FORMAT_FALLBACK_INSTRUCTION},
                ],
            }
            response = await self._send(payload, headers)

        if not response.is_success:
            raise _unavailable_error_for_status(response.status_code)

        try:
            raw_response = response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseError("The remote LLM returned invalid JSON.") from exc

        try:
            content = _extract_message_content(raw_response)
            structured_payload = _parse_structured_content(content)
            validated = response_model.model_validate(structured_payload)
        except (KeyError, TypeError, ValidationError, json.JSONDecodeError) as exc:
            raise LLMResponseError(
                "The remote LLM returned an invalid structured response."
            ) from exc

        self._record_usage(raw_response, operation=operation)
        return validated

    async def aclose(self) -> None:
        """Close only the client created by this provider."""

        if self._owns_client:
            await self._client.aclose()

    async def _send(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        """POST one chat-completions request, mapping transport failures."""

        try:
            return await self._client.post(
                self._endpoint,
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("The remote LLM request timed out.") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError("The remote LLM endpoint could not be reached.") from exc

    def _record_usage(self, raw_response: object, *, operation: LLMOperation) -> None:
        """Report one successful call to the usage sink without ever failing."""

        if self._usage_sink is None:
            return
        usage = parse_usage(
            raw_response,
            provider=self._provider_name,
            model=self._model,
            operation=operation,
        )
        if usage is None:
            return
        try:
            self._usage_sink.record(usage)
        except Exception:
            logger.warning("Failed to record LLM usage telemetry.", exc_info=True)


def _unavailable_error_for_status(status_code: int) -> LLMUnavailableError:
    """Map a non-success HTTP status to a typed availability failure."""

    if status_code in (401, 403):
        return LLMUnavailableError("The remote LLM rejected the configured credentials.")
    if status_code == 429:
        return LLMUnavailableError("The remote LLM endpoint is rate limited.")
    return LLMUnavailableError(f"The remote LLM endpoint returned HTTP {status_code}.")


def _rejects_response_format(response: httpx.Response) -> bool:
    """Detect whether the endpoint rejected the ``response_format`` parameter."""

    try:
        body = response.text
    except Exception:
        return False
    if not isinstance(body, str) or not body:
        return False
    return "response_format" in body.lower()


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
