"""Tests for the remote LLM provider and its usage telemetry."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from research_radar.errors import LLMResponseError, LLMUnavailableError
from research_radar.reader.llm import LLMMessage, RemoteLLMProvider
from research_radar.reader.llm.telemetry import InMemoryUsageSink, LLMUsage, parse_usage


class _Answer(BaseModel):
    answer: str
    confidence: int


class _RecordingTransport:
    """Queue canned responses while capturing every outgoing request."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.payloads: list[dict[str, Any]] = []
        self.urls: list[str] = []
        self.authorization: list[str | None] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.payloads.append(json.loads(request.content))
        self.urls.append(str(request.url))
        self.authorization.append(request.headers.get("authorization"))
        if not self._responses:
            raise AssertionError("unexpected extra HTTP request")
        return self._responses.pop(0)


def _success(content: Any, *, usage: dict[str, Any] | None = None) -> httpx.Response:
    body: dict[str, Any] = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


async def _generate(
    transport: _RecordingTransport,
    *,
    operation: str | None = None,
    usage_sink: InMemoryUsageSink | None = None,
) -> Any:
    """Run one generate_structured call against the recorded transport."""

    provider_kwargs: dict[str, Any] = {}
    if usage_sink is not None:
        provider_kwargs["usage_sink"] = usage_sink
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        provider = RemoteLLMProvider(
            base_url="https://models.example/v1",
            model="compact-model",
            api_key="secret-key",
            client=client,
            timeout_seconds=2,
            **provider_kwargs,
        )
        messages = [LLMMessage(role="user", content="Analyze")]
        call_kwargs = {} if operation is None else {"operation": operation}
        return await provider.generate_structured(messages, _Answer, **call_kwargs)


async def test_happy_path_sends_response_format_and_validates() -> None:
    transport = _RecordingTransport([_success('{"answer":"Grounded result","confidence":3}')])

    answer = await _generate(transport)

    assert answer == _Answer(answer="Grounded result", confidence=3)
    assert len(transport.payloads) == 1
    assert transport.payloads[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    "content",
    ['{"answer":"String form","confidence":1}', {"answer": "Object form", "confidence": 2}],
)
async def test_json_string_and_json_object_content_both_validate(content: Any) -> None:
    transport = _RecordingTransport([_success(content)])

    answer = await _generate(transport)

    assert answer.confidence in (1, 2)


async def test_rejected_response_format_retries_once_without_it() -> None:
    transport = _RecordingTransport(
        [
            httpx.Response(
                400,
                json={"error": {"message": "Unsupported parameter: response_format"}},
            ),
            _success({"answer": "Fallback ok", "confidence": 2}),
        ]
    )

    answer = await _generate(transport)

    assert answer == _Answer(answer="Fallback ok", confidence=2)
    assert len(transport.payloads) == 2
    assert transport.payloads[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in transport.payloads[1]
    last_message = transport.payloads[1]["messages"][-1]
    assert last_message["role"] == "system"
    assert "single valid JSON object" in last_message["content"]


async def test_unrelated_bad_request_is_not_retried() -> None:
    transport = _RecordingTransport(
        [httpx.Response(400, json={"error": {"message": "Malformed request body"}})]
    )

    with pytest.raises(LLMUnavailableError, match="HTTP 400"):
        await _generate(transport)

    assert len(transport.urls) == 1


@pytest.mark.parametrize("status", [401, 403])
async def test_auth_failures_raise_without_leaking_the_key(status: int) -> None:
    transport = _RecordingTransport([httpx.Response(status, text="denied")])

    with pytest.raises(LLMUnavailableError) as exc_info:
        await _generate(transport)

    assert str(exc_info.value) == "The remote LLM rejected the configured credentials."
    assert "secret-key" not in str(exc_info.value)
    assert len(transport.urls) == 1


async def test_rate_limit_raises_rate_limited_error() -> None:
    transport = _RecordingTransport([httpx.Response(429, text="slow down")])

    with pytest.raises(LLMUnavailableError, match="rate limited"):
        await _generate(transport)

    assert len(transport.urls) == 1


async def test_server_error_is_not_retried() -> None:
    transport = _RecordingTransport([httpx.Response(500, text="boom")])

    with pytest.raises(LLMUnavailableError, match="HTTP 500"):
        await _generate(transport)

    assert len(transport.urls) == 1


async def test_timeout_raises_and_makes_exactly_one_request() -> None:
    transport_calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        transport_calls.append(request)
        raise httpx.ReadTimeout("timed out", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = RemoteLLMProvider(
            base_url="https://models.example/v1",
            model="compact-model",
            api_key="secret-key",
            client=client,
            timeout_seconds=2,
        )
        with pytest.raises(LLMUnavailableError, match="timed out"):
            await provider.generate_structured(
                [LLMMessage(role="user", content="Analyze")], _Answer
            )

    assert len(transport_calls) == 1


async def test_pydantic_validation_failure_is_not_retried() -> None:
    transport = _RecordingTransport([_success('{"answer":"missing confidence"}')])

    with pytest.raises(LLMResponseError, match="invalid structured response"):
        await _generate(transport)

    assert len(transport.urls) == 1


async def test_non_json_body_raises_response_error() -> None:
    transport = _RecordingTransport([httpx.Response(200, text="<html>not json</html>")])

    with pytest.raises(LLMResponseError, match="invalid JSON"):
        await _generate(transport)


async def test_missing_choices_raises_response_error() -> None:
    transport = _RecordingTransport([httpx.Response(200, json={"object": "chat.completion"})])

    with pytest.raises(LLMResponseError, match="invalid structured response"):
        await _generate(transport)


async def test_successful_call_records_usage_telemetry() -> None:
    sink = InMemoryUsageSink()
    transport = _RecordingTransport(
        [
            _success(
                '{"answer":"Counted","confidence":4}',
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        ]
    )

    answer = await _generate(transport, operation="paper_card_extraction", usage_sink=sink)

    assert answer is not None
    entries = sink.entries
    assert len(entries) == 1
    entry = entries[0]
    assert (entry.input_tokens, entry.output_tokens, entry.total_tokens) == (10, 5, 15)
    assert entry.provider == "remote"
    assert entry.model == "compact-model"
    assert entry.operation == "paper_card_extraction"


async def test_response_without_usage_records_nothing() -> None:
    sink = InMemoryUsageSink()
    transport = _RecordingTransport([_success('{"answer":"No usage","confidence":1}')])

    await _generate(transport, usage_sink=sink)

    assert sink.entries == []


async def test_failing_usage_sink_does_not_break_the_request() -> None:
    class BrokenSink:
        def record(self, usage: LLMUsage) -> None:
            raise RuntimeError("sink exploded")

    transport = _RecordingTransport(
        [_success('{"answer":"Still fine","confidence":2}', usage={"total_tokens": 9})]
    )

    answer = await _generate(transport, usage_sink=BrokenSink())  # type: ignore[arg-type]

    assert answer == _Answer(answer="Still fine", confidence=2)


async def test_authorization_header_is_set_and_url_has_no_key() -> None:
    transport = _RecordingTransport([_success('{"answer":"Auth ok","confidence":1}')])

    await _generate(transport)

    assert transport.authorization == ["Bearer secret-key"]
    assert all("secret-key" not in url for url in transport.urls)


def test_in_memory_sink_evicts_oldest_and_totals_sum() -> None:
    sink = InMemoryUsageSink(max_entries=3)
    now = datetime.now(UTC)
    for index in range(5):
        sink.record(
            LLMUsage(
                provider="remote",
                model="compact-model",
                operation="other",
                input_tokens=index + 1,
                output_tokens=index * 2,
                total_tokens=index * 3,
                timestamp=now + timedelta(seconds=index),
            )
        )

    assert len(sink.entries) == 3
    assert [entry.input_tokens for entry in sink.entries] == [3, 4, 5]
    totals = sink.totals()
    assert totals == {"input_tokens": 12, "output_tokens": 18, "total_tokens": 27}


def test_parse_usage_accepts_anthropic_style_keys_and_garbage_safely() -> None:
    anthropic = parse_usage(
        {"usage": {"input_tokens": 7, "output_tokens": 2}},
        provider="remote",
        model="m",
        operation="ask_synthesis",
    )
    assert anthropic is not None
    assert (anthropic.input_tokens, anthropic.output_tokens) == (7, 2)

    assert parse_usage("not a mapping", provider="r", model="m", operation="other") is None
    assert parse_usage({}, provider="r", model="m", operation="other") is None
    assert parse_usage({"usage": "junk"}, provider="r", model="m", operation="other") is None
    garbage = parse_usage(
        {"usage": {"prompt_tokens": "many", "total_tokens": None}},
        provider="r",
        model="m",
        operation="other",
    )
    assert garbage is not None
    assert garbage.input_tokens is None and garbage.total_tokens is None
