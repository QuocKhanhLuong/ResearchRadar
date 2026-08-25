"""Tests for EvidencePacket scope derivation and citation-id validation."""

from __future__ import annotations

from research_radar.chat.evidence import (
    DiscoveryEvidenceItem,
    EvidencePacket,
    ProjectMemory,
    StoredEvidenceItem,
    validate_cited_ids,
)
from research_radar.chat.models import EvidenceScope
from research_radar.memory.models import MemoryClass, MemoryFact, UserMemoryContext


def _stored(paper_id: str = "p1", *, has_card: bool = True) -> StoredEvidenceItem:
    return StoredEvidenceItem(
        paper_id=paper_id,
        title="A stored paper",
        year=2024,
        venue="Venue",
        abstract="An abstract.",
        has_paper_card=has_card,
        card_summary="A claim." if has_card else None,
    )


def _discovery(paper_id: str = "d1") -> DiscoveryEvidenceItem:
    return DiscoveryEvidenceItem(
        paper_id=paper_id,
        title="A discovered paper",
        year=2025,
        venue="Venue",
        abstract=None,
    )


def _memory_context(*facts: str) -> UserMemoryContext:
    return UserMemoryContext(
        facts=tuple(
            MemoryFact(fact=fact, memory_class=MemoryClass.PREFERENCE) for fact in facts
        ),
        backend="fake",
        degraded=False,
    )


# ---------------------------------------------------------------------------
# Allowed id sets.
# ---------------------------------------------------------------------------


def test_allowed_paper_ids_unite_stored_and_discovery() -> None:
    packet = EvidencePacket(stored=(_stored("s1"),), discovery=(_discovery("d1"),))

    assert packet.allowed_paper_ids == {"s1", "d1"}


def test_allowed_gap_ids_mirror_the_packet_tuple() -> None:
    packet = EvidencePacket(gap_ids=("g1", "g2"))

    assert packet.allowed_gap_ids == {"g1", "g2"}


# ---------------------------------------------------------------------------
# Evidence-scope derivation.
# ---------------------------------------------------------------------------


def test_empty_packet_scope_is_none() -> None:
    assert EvidencePacket().evidence_scope is EvidenceScope.NONE


def test_project_only_packet_scope_is_none() -> None:
    """Explicit project state alone is not one of the scoped evidence classes."""

    packet = EvidencePacket(project=ProjectMemory(project_id="prj", name="medvla"))

    assert packet.evidence_scope is EvidenceScope.NONE


def test_memory_only_packet_scope_is_user_memory() -> None:
    packet = EvidencePacket(user_memory=_memory_context("I like small models"))

    assert packet.evidence_scope is EvidenceScope.USER_MEMORY


def test_all_card_stored_packet_scope_is_stored_cards() -> None:
    packet = EvidencePacket(stored=(_stored("s1"), _stored("s2")))

    assert packet.evidence_scope is EvidenceScope.STORED_CARDS


def test_any_cardless_stored_item_degrades_scope_to_metadata() -> None:
    packet = EvidencePacket(stored=(_stored("s1"), _stored("s2", has_card=False)))

    assert packet.evidence_scope is EvidenceScope.STORED_METADATA


def test_discovery_only_packet_scope_is_discovery_metadata() -> None:
    packet = EvidencePacket(discovery=(_discovery(),))

    assert packet.evidence_scope is EvidenceScope.DISCOVERY_METADATA


def test_stored_plus_discovery_scope_is_mixed() -> None:
    packet = EvidencePacket(stored=(_stored(),), discovery=(_discovery(),))

    assert packet.evidence_scope is EvidenceScope.MIXED


def test_memory_plus_stored_scope_is_mixed() -> None:
    packet = EvidencePacket(stored=(_stored(),), user_memory=_memory_context("fact"))

    assert packet.evidence_scope is EvidenceScope.MIXED


def test_memory_plus_discovery_scope_is_mixed() -> None:
    packet = EvidencePacket(discovery=(_discovery(),), user_memory=_memory_context("f"))

    assert packet.evidence_scope is EvidenceScope.MIXED


# ---------------------------------------------------------------------------
# Citation-id validation.
# ---------------------------------------------------------------------------


def test_validate_cited_ids_keeps_known_paper_and_gap_ids_in_order() -> None:
    packet = EvidencePacket(
        stored=(_stored("s1"), _stored("s2")),
        discovery=(_discovery("d1"),),
        gap_ids=("g1",),
    )

    result = validate_cited_ids(["d1", "s2", "g1", "s1"], packet)

    assert result == ("d1", "s2", "g1", "s1")


def test_validate_cited_ids_drops_unknown_ids() -> None:
    packet = EvidencePacket(stored=(_stored("s1"),), gap_ids=("g1",))

    result = validate_cited_ids(["s1", "provider:xyz", "vector-index-9", "g1"], packet)

    assert result == ("s1", "g1")


def test_validate_cited_ids_deduplicates_preserving_first_position() -> None:
    packet = EvidencePacket(stored=(_stored("s1"), _stored("s2")))

    result = validate_cited_ids(["s2", "s1", "s2"], packet)

    assert result == ("s2", "s1")


def test_validate_cited_ids_handles_blank_and_whitespace_entries() -> None:
    packet = EvidencePacket(stored=(_stored("s1"),))

    assert validate_cited_ids(["", "  ", " s1 "], packet) == ("s1",)


def test_validate_cited_ids_on_empty_packet_drops_everything() -> None:
    assert validate_cited_ids(["anything"], EvidencePacket()) == ()
