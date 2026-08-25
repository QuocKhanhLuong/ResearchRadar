"""Contract tests for chat value types (ChatMode, EvidenceScope, requests)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import research_radar.chat as chat_package
from research_radar.chat.models import (
    ChatMode,
    ChatRequest,
    ChatResponse,
    EvidenceScope,
)


def test_chat_mode_values_match_contract() -> None:
    assert ChatMode.CONVERSATIONAL == "conversational"
    assert ChatMode.PERSONAL_MEMORY == "personal_memory"
    assert ChatMode.RESEARCH_STORED == "research_stored"
    assert ChatMode.RESEARCH_LIVE == "research_live"
    assert ChatMode.PROJECT_RESEARCH == "project_research"


def test_evidence_scope_values_match_contract() -> None:
    assert EvidenceScope.NONE == "none"
    assert EvidenceScope.USER_MEMORY == "user_memory"
    assert EvidenceScope.STORED_CARDS == "stored_cards"
    assert EvidenceScope.STORED_METADATA == "stored_metadata"
    assert EvidenceScope.DISCOVERY_METADATA == "discovery_metadata"
    assert EvidenceScope.MIXED == "mixed"


def test_research_live_exists_for_service_assignment_only() -> None:
    """RESEARCH_LIVE belongs to ChatService; the router never returns it."""

    assert isinstance(ChatMode.RESEARCH_LIVE, ChatMode)


def test_chat_request_defaults() -> None:
    request = ChatRequest(text="hello")

    assert request.text == "hello"
    assert request.discord_user_id is None
    assert request.channel_id is None
    assert request.message_id is None
    assert request.project_hint is None


def test_chat_request_is_frozen() -> None:
    request = ChatRequest(text="hello")

    try:
        request.text = "goodbye"  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("ChatRequest must be frozen")


def test_chat_response_defaults() -> None:
    response = ChatResponse(text="hi", mode=ChatMode.CONVERSATIONAL)

    assert response.paper_ids == ()
    assert response.gap_ids == ()
    assert response.used_user_memory is False
    assert response.live_discovery_used is False
    assert response.evidence_scope is EvidenceScope.NONE
    assert response.degraded is False


def test_chat_response_is_frozen() -> None:
    response = ChatResponse(text="hi", mode=ChatMode.CONVERSATIONAL)

    try:
        response.text = "changed"  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("ChatResponse must be frozen")


def test_package_reexports_public_surface() -> None:
    assert chat_package.ChatMode is ChatMode
    assert chat_package.EvidenceScope is EvidenceScope
    assert chat_package.ChatRequest is ChatRequest
    assert chat_package.ChatResponse is ChatResponse
    assert set(chat_package.__all__) == {
        "ChatMode",
        "ChatRequest",
        "ChatResponse",
        "ChatRouter",
        "EvidenceScope",
        "RouteDecision",
    }
