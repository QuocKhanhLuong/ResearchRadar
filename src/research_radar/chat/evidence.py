"""Evidence packet assembly and citation-id validation for the chat surface.

Every id carried by an :class:`EvidencePacket` is a canonical SQLite id that
has already been read back from storage. A semantic/vector hit is a candidate
only; it never becomes evidence on its own. ``validate_cited_ids`` enforces the
matching output-side rule: any id the language model cites that is not in the
packet's allowed sets is dropped before it can reach a :class:`ChatResponse`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from research_radar.chat.models import EvidenceScope
from research_radar.memory.models import UserMemoryContext


@dataclass(frozen=True, slots=True)
class StoredEvidenceItem:
    """One stored paper resolved from SQLite, with its PaperCard status."""

    paper_id: str  # canonical SQLite id, always resolved
    title: str
    year: int | None
    venue: str | None
    abstract: str | None
    has_paper_card: bool
    card_summary: str | None = None


@dataclass(frozen=True, slots=True)
class DiscoveryEvidenceItem:
    """One live-discovered paper after ingestion and SQLite re-resolution."""

    paper_id: str  # canonical SQLite id after ingestion; never a raw provider id
    title: str
    year: int | None
    venue: str | None
    abstract: str | None


@dataclass(frozen=True, slots=True)
class ProjectMemory:
    """Explicit canonical project state loaded from SQLite."""

    project_id: str
    name: str
    constraints: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    rejected_ideas: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvidencePacket:
    """The bounded evidence a single chat turn may ground its answer in."""

    stored: tuple[StoredEvidenceItem, ...] = ()
    discovery: tuple[DiscoveryEvidenceItem, ...] = ()
    gap_ids: tuple[str, ...] = ()
    project: ProjectMemory | None = None
    user_memory: UserMemoryContext = UserMemoryContext()
    live_discovery_used: bool = False

    @property
    def allowed_paper_ids(self) -> set[str]:
        """Return every paper id the answer is permitted to cite."""

        return {item.paper_id for item in self.stored} | {
            item.paper_id for item in self.discovery
        }

    @property
    def allowed_gap_ids(self) -> set[str]:
        """Return every candidate-gap id the answer is permitted to cite."""

        return set(self.gap_ids)

    @property
    def evidence_scope(self) -> EvidenceScope:
        """Derive which evidence classes this packet actually rests on.

        Nothing present maps to ``NONE``; advisory user memory alone maps to
        ``USER_MEMORY``; stored items all carrying PaperCards map to
        ``STORED_CARDS`` while any cardless stored item degrades the scope to
        ``STORED_METADATA``; discovery-only packets map to
        ``DISCOVERY_METADATA``; any combination of classes is ``MIXED``.
        """

        has_stored = bool(self.stored)
        has_discovery = bool(self.discovery)
        has_memory = self.user_memory.available
        if not (has_stored or has_discovery or has_memory):
            return EvidenceScope.NONE
        if not has_stored and not has_discovery:
            if has_memory:
                return EvidenceScope.USER_MEMORY
            return EvidenceScope.NONE
        if has_stored and not has_discovery and not has_memory:
            if all(item.has_paper_card for item in self.stored):
                return EvidenceScope.STORED_CARDS
            return EvidenceScope.STORED_METADATA
        if has_discovery and not has_stored and not has_memory:
            return EvidenceScope.DISCOVERY_METADATA
        return EvidenceScope.MIXED


def validate_cited_ids(
    text_ids: Iterable[str],
    packet: EvidencePacket,
) -> tuple[str, ...]:
    """Keep only cited ids that belong to the packet's allowed sets.

    Ids are checked against ``allowed_paper_ids | allowed_gap_ids``, deduped,
    stripped of surrounding whitespace, and returned in first-mentioned order.
    Anything unknown — a provider id, a vector-index id, or an outright
    fabrication — is silently dropped rather than surfaced.
    """

    allowed = packet.allowed_paper_ids | packet.allowed_gap_ids
    seen: set[str] = set()
    kept: list[str] = []
    for raw_id in text_ids:
        candidate = (raw_id or "").strip()
        if not candidate or candidate in seen or candidate not in allowed:
            continue
        seen.add(candidate)
        kept.append(candidate)
    return tuple(kept)
