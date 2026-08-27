"""Tests for ChatRouter deterministic routing and the optional LLM assist."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any

import pytest
from pydantic import BaseModel

from research_radar.chat import ChatMode, ChatRequest, RouteDecision
from research_radar.chat import router as router_module
from research_radar.chat.router import (
    ChatRouter,
    RoutingAssistResponse,
    normalize_search_query,
)


class ExplodingProvider:
    """LLMProvider double that fails the test if generate_structured runs."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(
        self, messages: list[BaseModel], response_model: type[BaseModel]
    ) -> BaseModel:
        self.calls += 1
        raise AssertionError("generate_structured must not be called")


class ScriptedProvider:
    """Deterministic LLMProvider double with a scripted payload or error."""

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        exc: Exception | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self._payload = payload
        self._exc = exc
        self._delay_seconds = delay_seconds
        self.calls = 0
        self.last_messages: list[Any] = []

    async def generate_structured(
        self, messages: list[Any], response_model: type[BaseModel]
    ) -> BaseModel:
        self.calls += 1
        self.last_messages = list(messages)
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        if self._exc is not None:
            raise self._exc
        assert self._payload is not None
        return response_model.model_validate(self._payload)


async def route_text(text: str, **request_kwargs: Any) -> RouteDecision:
    return await ChatRouter().route(ChatRequest(text=text, **request_kwargs))


# ---------------------------------------------------------------------------
# Phase-brief examples.
# ---------------------------------------------------------------------------


async def test_brief_greeting_routes_conversational() -> None:
    decision = await route_text("hello")

    assert decision.mode is ChatMode.CONVERSATIONAL
    assert decision.needs_stored_research is False
    assert decision.allows_live_discovery is False
    assert decision.needs_user_memory is False
    assert decision.search_query == ""


async def test_brief_memory_question_routes_personal_memory() -> None:
    decision = await route_text("what research areas do I seem to care about?")

    assert decision.mode is ChatMode.PERSONAL_MEMORY
    assert decision.needs_user_memory is True
    assert decision.allows_live_discovery is False
    assert decision.needs_stored_research is False


async def test_brief_research_request_routes_research_stored_with_live() -> None:
    decision = await route_text("find recent work about medical VLA reasoning")

    assert decision.mode is ChatMode.RESEARCH_STORED
    assert decision.allows_live_discovery is True
    assert decision.needs_stored_research is True
    assert decision.search_query == "medical VLA reasoning"


async def test_brief_comparison_routes_research_stored_with_live() -> None:
    decision = await route_text("compare world models and VLA for medical AI")

    assert decision.mode is ChatMode.RESEARCH_STORED
    assert decision.allows_live_discovery is True
    assert decision.search_query == "world models and VLA for medical AI"


async def test_brief_durable_statement_needs_user_memory_only() -> None:
    decision = await route_text("I don't want to pursue GAN-based directions")

    assert decision.mode is ChatMode.CONVERSATIONAL
    assert decision.needs_user_memory is True
    assert decision.needs_stored_research is False
    assert decision.allows_live_discovery is False
    assert decision.search_query == ""


async def test_project_hint_routes_project_research() -> None:
    decision = await route_text("what is the current status?", project_hint="medvla")

    assert decision.mode is ChatMode.PROJECT_RESEARCH
    assert decision.project_hint == "medvla"
    assert decision.needs_stored_research is True
    assert decision.allows_live_discovery is False


async def test_textual_project_reference_routes_project_research() -> None:
    decision = await route_text("where do things stand on my project medvla")

    assert decision.mode is ChatMode.PROJECT_RESEARCH
    assert decision.project_hint == "medvla"
    assert decision.search_query != ""


@pytest.mark.parametrize("text", ["", "   ", "\n\t  \n"])
async def test_empty_or_whitespace_text_is_conversational_without_research(
    text: str,
) -> None:
    decision = await route_text(text)

    assert decision.mode is ChatMode.CONVERSATIONAL
    assert decision.needs_user_memory is False
    assert decision.needs_stored_research is False
    assert decision.allows_live_discovery is False
    assert decision.search_query == ""


async def test_router_never_returns_research_live() -> None:
    texts = [
        "hello",
        "what are my interests?",
        "find papers on diffusion policy",
        "compare rlhf and dpo for alignment",
        "my goal is to survey medical VLA work",
        "tell me about world models for robotics",
        "",
    ]
    provider = ScriptedProvider(payload={"mode": "research", "topic": "world models"})
    router = ChatRouter(llm_provider=provider)

    for text in texts:
        decision = await router.route(ChatRequest(text=text))
        assert decision.mode is not ChatMode.RESEARCH_LIVE


# ---------------------------------------------------------------------------
# Deterministic rules are authoritative; clear cases never call the LLM.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "hey there!",
        "thanks bot",
        "what can you do?",
        "what research areas do I seem to care about?",
        "remember what I asked about last week",
        "I prefer smaller open-source models",
        "My goal is reproducible evaluations",
        "I don't want to pursue GAN-based directions",
        "find recent work about medical VLA reasoning",
        "compare world models and VLA for medical AI",
        "",
        "   ",
    ],
)
async def test_clear_cases_never_invoke_the_provider(text: str) -> None:
    provider = ExplodingProvider()
    router = ChatRouter(llm_provider=provider)

    decision = await router.route(ChatRequest(text=text))

    assert provider.calls == 0
    assert decision.mode is not ChatMode.RESEARCH_LIVE


async def test_greeting_with_raising_provider_makes_no_call() -> None:
    """Contract assertion: greetings must never reach the optional assist."""

    provider = ExplodingProvider()
    decision = await ChatRouter(llm_provider=provider).route(ChatRequest(text="hello"))

    assert provider.calls == 0
    assert decision.mode is ChatMode.CONVERSATIONAL


# ---------------------------------------------------------------------------
# Optional LLM assist: ambiguity condition, mapping, and silent fallbacks.
# ---------------------------------------------------------------------------


AMBIGUOUS_TEXT = "tell me about diffusion models for surgical robotics"


async def test_ambiguous_text_refined_to_research_by_assist() -> None:
    provider = ScriptedProvider(
        payload={"mode": "research", "topic": "diffusion models in surgical robotics"}
    )
    router = ChatRouter(llm_provider=provider)

    decision = await router.route(ChatRequest(text=AMBIGUOUS_TEXT))

    assert provider.calls == 1
    assert decision.mode is ChatMode.RESEARCH_STORED
    assert decision.allows_live_discovery is True
    assert decision.search_query == "diffusion models in surgical robotics"


async def test_assist_prompt_carries_the_user_message() -> None:
    provider = ScriptedProvider(payload={"mode": "conversational"})
    router = ChatRouter(llm_provider=provider)

    await router.route(ChatRequest(text=AMBIGUOUS_TEXT))

    assert provider.calls == 1
    system_message, user_message = provider.last_messages
    assert isinstance(system_message, BaseModel)
    assert user_message.content == f"Message: {AMBIGUOUS_TEXT}"


async def test_assist_conversational_reply_keeps_deterministic_fallback() -> None:
    provider = ScriptedProvider(payload={"mode": "conversational"})
    decision = await ChatRouter(llm_provider=provider).route(
        ChatRequest(text=AMBIGUOUS_TEXT)
    )

    assert provider.calls == 1
    assert decision.mode is ChatMode.CONVERSATIONAL
    assert decision.needs_stored_research is False


async def test_assist_personal_memory_reply_maps_to_personal_memory() -> None:
    provider = ScriptedProvider(payload={"mode": "personal_memory", "topic": ""})
    decision = await ChatRouter(llm_provider=provider).route(
        ChatRequest(text="any thoughts on what I have been up to lately")
    )

    assert decision.mode is ChatMode.PERSONAL_MEMORY
    assert decision.needs_user_memory is True
    assert decision.allows_live_discovery is False


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "shopping", "topic": "x"},
        {"mode": "research", "topic": "x", "surprise": True},
    ],
)
async def test_invalid_assist_payload_falls_back_silently(payload: dict[str, Any]) -> None:
    provider = ScriptedProvider(payload=payload)
    router = ChatRouter(llm_provider=provider)

    decision = await router.route(ChatRequest(text=AMBIGUOUS_TEXT))

    assert provider.calls == 1
    assert decision.mode is ChatMode.CONVERSATIONAL


async def test_provider_exception_falls_back_to_deterministic_decision() -> None:
    provider = ScriptedProvider(exc=RuntimeError("provider down"))
    decision = await ChatRouter(llm_provider=provider).route(
        ChatRequest(text=AMBIGUOUS_TEXT)
    )

    assert provider.calls == 1
    assert decision.mode is ChatMode.CONVERSATIONAL


async def test_assist_timeout_falls_back_to_deterministic_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(router_module, "_ASSIST_TIMEOUT_SECONDS", 0.05)
    provider = ScriptedProvider(
        payload={"mode": "research", "topic": "late topic"}, delay_seconds=0.5
    )
    decision = await ChatRouter(llm_provider=provider).route(
        ChatRequest(text=AMBIGUOUS_TEXT)
    )

    assert provider.calls == 1
    assert decision.mode is ChatMode.CONVERSATIONAL


async def test_assist_failure_log_is_sanitized(caplog: pytest.LogCaptureFixture) -> None:
    provider = ScriptedProvider(exc=RuntimeError("boom"))
    router = ChatRouter(llm_provider=provider)

    with caplog.at_level(logging.DEBUG, logger="research_radar.chat.router"):
        decision = await router.route(ChatRequest(text=AMBIGUOUS_TEXT))

    assert decision.mode is ChatMode.CONVERSATIONAL
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_records, "expected a sanitized debug log line"
    joined = " ".join(r.getMessage() for r in debug_records)
    assert AMBIGUOUS_TEXT not in joined
    assert "boom" not in joined


async def test_short_ambiguous_text_skips_assist_entirely() -> None:
    """Below the documented three-token substance threshold no assist runs."""

    provider = ScriptedProvider(payload={"mode": "research", "topic": "x"})
    decision = await ChatRouter(llm_provider=provider).route(ChatRequest(text="ok then"))

    assert provider.calls == 0
    assert decision.mode is ChatMode.CONVERSATIONAL


async def test_ambiguous_text_without_provider_stays_conversational() -> None:
    decision = await route_text(AMBIGUOUS_TEXT)

    assert decision.mode is ChatMode.CONVERSATIONAL


def test_route_decision_is_frozen() -> None:
    decision = RouteDecision(
        mode=ChatMode.RESEARCH_STORED,
        needs_user_memory=False,
        needs_stored_research=True,
        allows_live_discovery=True,
        search_query="topic",
    )

    try:
        decision.mode = ChatMode.PERSONAL_MEMORY  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("RouteDecision must be frozen")


# ---------------------------------------------------------------------------
# search_query normalization.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("find recent work about medical VLA reasoning", "medical VLA reasoning"),
        ("any papers about world models?? ", "world models"),
        ("what is the recent work on RLHF", "RLHF"),
        (
            "compare world models and VLA for medical AI.",
            "world models and VLA for medical AI",
        ),
        ("search for diffusion policy papers", "diffusion policy papers"),
        (
            "hey, can you find recent papers on VLA?",
            "VLA",
        ),
        (
            "please could you look up state of the art in world models?",
            "state of the art in world models",
        ),
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_search_query(raw: str, expected: str) -> None:
    assert normalize_search_query(raw) == expected


def test_routing_assist_response_defaults() -> None:
    assist = RoutingAssistResponse()

    assert assist.mode == "conversational"
    assert assist.topic == ""


async def test_new_research_topic_without_project_routes_to_research_stored() -> None:
    """A new research query without project hint routes to RESEARCH_STORED allowing live."""

    decision = await route_text("find recent papers on graph neural networks")

    assert decision.mode is ChatMode.RESEARCH_STORED
    assert decision.allows_live_discovery is True
    assert decision.needs_stored_research is True
    assert decision.project_hint is None
    assert decision.search_query == "graph neural networks"


async def test_research_request_with_memory_question_enables_both_channels() -> None:
    """A query asking for papers tailored to user interests enables memory and discovery."""

    decision = await route_text("find papers about my research interests in medical robotics")

    assert decision.mode is ChatMode.RESEARCH_STORED
    assert decision.allows_live_discovery is True
    assert decision.needs_stored_research is True
    assert decision.needs_user_memory is True
    assert decision.project_hint is None
