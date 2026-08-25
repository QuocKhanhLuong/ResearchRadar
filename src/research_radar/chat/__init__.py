"""Public surface for the personal research chat package."""

from __future__ import annotations

from research_radar.chat.models import ChatMode, ChatRequest, ChatResponse, EvidenceScope
from research_radar.chat.router import ChatRouter, RouteDecision

__all__ = [
    "ChatMode",
    "ChatRequest",
    "ChatResponse",
    "ChatRouter",
    "EvidenceScope",
    "RouteDecision",
]

# ---------------------------------------------------------------------------
# Future re-exports append here. ChatService (chat.service), EvidencePacket
# (chat.evidence), and build_chat_prompt (chat.prompt) are added to this
# package by later workers; extend __all__ and add their imports below this
# marker as a clean append. Do not import modules that do not exist yet.
# ---------------------------------------------------------------------------
