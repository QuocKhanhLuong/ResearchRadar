"""Value types for the personal research chat surface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ChatMode(StrEnum):
    """Coarse intent classification assigned to one chat turn.

    ``RESEARCH_LIVE`` is never produced by :mod:`research_radar.chat.router`;
    the router only grants permission via
    ``RouteDecision.allows_live_discovery`` and ``ChatService`` decides whether
    live discovery actually runs (and therefore whether the turn becomes
    ``RESEARCH_LIVE``).
    """

    CONVERSATIONAL = "conversational"
    PERSONAL_MEMORY = "personal_memory"
    RESEARCH_STORED = "research_stored"
    RESEARCH_LIVE = "research_live"
    PROJECT_RESEARCH = "project_research"


class EvidenceScope(StrEnum):
    """Which evidence classes an answer is allowed to rest on."""

    NONE = "none"
    USER_MEMORY = "user_memory"
    STORED_CARDS = "stored_cards"
    STORED_METADATA = "stored_metadata"
    DISCOVERY_METADATA = "discovery_metadata"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """One inbound user chat turn with optional Discord provenance."""

    text: str
    discord_user_id: str | None = None
    channel_id: str | None = None
    message_id: str | None = None
    project_hint: str | None = None


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Final answer for a chat turn plus bounded provenance metadata."""

    text: str
    mode: ChatMode
    paper_ids: tuple[str, ...] = ()
    gap_ids: tuple[str, ...] = ()
    used_user_memory: bool = False
    live_discovery_used: bool = False
    evidence_scope: EvidenceScope = EvidenceScope.NONE
    degraded: bool = False
