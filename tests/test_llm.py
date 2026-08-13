from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from research_radar.errors import LLMUnavailableError
from research_radar.reader.llm import (
    LLMMessage,
    LLMResponseError,
    MockLLMProvider,
    RemoteLLMProvider,
)


class _Answer(BaseModel):
    answer: str
    confidence: int


@pytest.mark.asyncio
async def test_mock_llm_is_unavailable_without_an_explicit_fixture() -> None:
    provider = MockLLMProvider()

    with pytest.raises(LLMUnavailableError, match="No language model is configured"):
        await provider.generate_structured([LLMMessage(role="user", content="Analyze")], _Answer)


@pytest.mark.asyncio
async def test_mock_llm_returns_a_deterministic_validated_fixture() -> None:
    provider = MockLLMProvider({"answer": "Evidence is limited.", "confidence": 2})
    messages = [LLMMessage(role="user", content="Analyze")]

    answer = await provider.generate_structured(messages, _Answer)

    assert answer == _Answer(answer="Evidence is limited.", confidence=2)
    assert provider.requests == [messages]


@pytest.mark.asyncio
async def test_remote_llm_posts_to_openai_compatible_endpoint_and_validates_response() -> None:
    received: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["url"] = str(request.url)
        received["authorization"] = request.headers.get("authorization")
        received["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"answer":"Grounded result","confidence":3}'}}
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = RemoteLLMProvider(
            base_url="https://models.example/v1",
            model="compact-model",
            api_key="test-key",
            client=client,
            timeout_seconds=2,
        )
        answer = await provider.generate_structured(
            [LLMMessage(role="system", content="Return JSON.")], _Answer
        )

    assert answer.answer == "Grounded result"
    assert received["url"] == "https://models.example/v1/chat/completions"
    assert received["authorization"] == "Bearer test-key"
    assert received["payload"] == {
        "model": "compact-model",
        "messages": [{"role": "system", "content": "Return JSON."}],
        "response_format": {"type": "json_object"},
    }


@pytest.mark.asyncio
async def test_remote_llm_maps_endpoint_failures_and_bad_structured_results() -> None:
    def unavailable_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable_handler)) as client:
        provider = RemoteLLMProvider(
            base_url="https://models.example/v1",
            model="compact-model",
            client=client,
        )
        with pytest.raises(LLMUnavailableError, match="HTTP 503"):
            await provider.generate_structured(
                [LLMMessage(role="user", content="Analyze")], _Answer
            )

    def malformed_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(malformed_handler)) as client:
        provider = RemoteLLMProvider(
            base_url="https://models.example/v1/chat/completions",
            model="compact-model",
            client=client,
        )
        with pytest.raises(LLMResponseError, match="invalid structured response"):
            await provider.generate_structured(
                [LLMMessage(role="user", content="Analyze")], _Answer
            )
