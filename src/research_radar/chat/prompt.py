"""Bounded prompt construction for the personal research chat.

``build_chat_prompt`` renders one chat turn as a vendor-neutral
``list[LLMMessage]`` with sections in a fixed order:

1. ``SYSTEM RULES``
2. ``USER MEMORY (ADVISORY — NOT SCIENTIFIC EVIDENCE)``
3. ``EXPLICIT PROJECT MEMORY (CANONICAL USER/PROJECT STATE)``
4. ``STORED SCIENTIFIC EVIDENCE (CANONICAL)``
5. ``LIVE DISCOVERY EVIDENCE (METADATA/ABSTRACT-LEVEL ONLY)``
6. ``QUESTION``

Empty sections are omitted entirely, except ``SYSTEM RULES`` and ``QUESTION``.
Rendering style (id labelling, truncation ellipsis, frozen budget dataclass)
follows the established ``research_radar.research.ask`` conventions.

The evidence-packet types (``EvidencePacket``, ``StoredEvidenceItem``,
``DiscoveryEvidenceItem``, ``ProjectMemory``) are defined by
``research_radar.chat.evidence`` per PHASE_CONTRACTS section 8 and are owned by
W6. That module was not present in this worktree at implementation time, so it
is imported under ``TYPE_CHECKING`` only; this function relies purely on the
contracted attribute surface and never edits or recreates that module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from research_radar.chat.models import ChatRequest
from research_radar.chat.router import RouteDecision
from research_radar.memory.models import MemoryFact
from research_radar.reader.llm.base import LLMMessage

if TYPE_CHECKING:
    from research_radar.chat.evidence import (
        DiscoveryEvidenceItem,
        EvidencePacket,
        ProjectMemory,
        StoredEvidenceItem,
    )

__all__ = ["ChatPromptBudget", "build_chat_prompt"]

SYSTEM_RULES_HEADER = "SYSTEM RULES"
USER_MEMORY_HEADER = "USER MEMORY (ADVISORY — NOT SCIENTIFIC EVIDENCE)"
PROJECT_MEMORY_HEADER = "EXPLICIT PROJECT MEMORY (CANONICAL USER/PROJECT STATE)"
STORED_EVIDENCE_HEADER = "STORED SCIENTIFIC EVIDENCE (CANONICAL)"
DISCOVERY_EVIDENCE_HEADER = "LIVE DISCOVERY EVIDENCE (METADATA/ABSTRACT-LEVEL ONLY)"
QUESTION_HEADER = "QUESTION"

_SYSTEM_RULES = (
    "You are ResearchRadar, a private single-user research assistant. "
    "Follow these rules strictly.\n"
    "1. USER MEMORY is what the user has said about themselves. It is advisory "
    "context only: it is NEVER scientific support and NEVER evidence that a "
    "claim is true.\n"
    "2. Never present a user hypothesis, belief, preference, or goal as a "
    'published result or as something "the literature shows".\n'
    "3. Never claim that the literature contains no work on a topic. The "
    "corpus available here is partial and bounded; absence from this packet is "
    "not absence from the literature.\n"
    "4. Cite ONLY the paper ids and gap ids that appear in the evidence "
    "sections, using their exact ids. Never invent, alter, or guess an id.\n"
    "5. Distinguish STORED SCIENTIFIC EVIDENCE items that carry an analysed "
    "PaperCard (full text was read) from abstract/metadata-only items, and say "
    "explicitly whenever your answer rests only on abstract or live-discovery "
    "metadata.\n"
    "6. Do not invent deep experimental claims (numbers, ablations, benchmark "
    "results) from metadata alone. If the question needs that depth, say the "
    "answer requires reading the papers in full.\n"
    "7. When EXPLICIT PROJECT MEMORY and USER MEMORY conflict, EXPLICIT "
    "PROJECT MEMORY is canonical and outranks USER MEMORY. Say so instead of "
    "silently picking one.\n"
    "8. If the available evidence is insufficient, say plainly that it is "
    "insufficient rather than inventing an answer."
)


@dataclass(frozen=True, slots=True)
class ChatPromptBudget:
    """Adjustable caps bounding one rendered chat prompt."""

    max_user_memory_facts: int = 8
    max_stored_items: int = 8
    max_discovery_items: int = 10
    max_items_per_project_list: int = 6
    max_abstract_chars: int = 800
    max_summary_chars: int = 600
    max_fact_chars: int = 280


def _truncate_text(text: str, max_chars: int) -> str:
    """Collapse whitespace, then truncate at a word boundary with an ellipsis.

    Follows the ask.py budgeting convention but never splits mid-word: the cut
    is backed off to the last whitespace inside the budget.
    """

    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    kept = collapsed[:max_chars].rsplit(" ", 1)[0]
    if not kept:
        kept = collapsed[:max_chars]
    return kept + "..."


def _temporal_suffix(fact: MemoryFact) -> str:
    """Render temporal validity so superseded facts stay visibly older."""

    parts: list[str] = []
    if fact.valid_at is not None:
        parts.append(f"valid from {fact.valid_at.date().isoformat()}")
    if fact.invalid_at is not None:
        parts.append(f"superseded after {fact.invalid_at.date().isoformat()}")
    return f" ({'; '.join(parts)})" if parts else ""


def _render_memory_fact(fact: MemoryFact, budget: ChatPromptBudget) -> str:
    """Render one user-memory fact with class label and temporal validity."""

    label = f"[{fact.memory_class.value}] " if fact.memory_class is not None else ""
    text = _truncate_text(fact.fact, budget.max_fact_chars)
    return f"- {label}{text}{_temporal_suffix(fact)}"


def _paper_header(paper_id: str) -> str:
    """Label one paper block exactly like the /ask evidence packet does."""

    return f"--- Paper {paper_id} ---"


def _meta_lines(
    *,
    title: str,
    year: int | None,
    venue: str | None,
    abstract: str | None,
    budget: ChatPromptBudget,
) -> list[str]:
    """Render the shared title/year/venue/abstract metadata lines."""

    lines = [f"Title: {title}"]
    if year is not None:
        lines.append(f"Year: {year}")
    if venue:
        lines.append(f"Venue: {venue}")
    abstract_line = _truncate_text(abstract or "N/A", budget.max_abstract_chars)
    lines.append(f"Abstract: {abstract_line}")
    return lines


def _render_stored_item(item: StoredEvidenceItem, budget: ChatPromptBudget) -> list[str]:
    """Render one stored item, marking whether full text was analysed."""

    lines = [
        _paper_header(item.paper_id),
        *_meta_lines(
            title=item.title,
            year=item.year,
            venue=item.venue,
            abstract=item.abstract,
            budget=budget,
        ),
    ]
    if item.has_paper_card:
        lines.append("Evidence depth: analysed full text (PaperCard)")
        if item.card_summary:
            summary = _truncate_text(item.card_summary, budget.max_summary_chars)
            lines.append(f"PaperCard summary: {summary}")
    else:
        lines.append("Evidence depth: abstract/metadata only (no PaperCard)")
    return lines


def _render_discovery_item(
    item: DiscoveryEvidenceItem, budget: ChatPromptBudget
) -> list[str]:
    """Render one live-discovery item; these are always metadata-level only."""

    lines = [
        _paper_header(item.paper_id),
        *_meta_lines(
            title=item.title,
            year=item.year,
            venue=item.venue,
            abstract=item.abstract,
            budget=budget,
        ),
    ]
    lines.append("Evidence depth: abstract/metadata only (live discovery, no PaperCard)")
    return lines


def _render_project(project: ProjectMemory, budget: ChatPromptBudget) -> list[str]:
    """Render explicit project state; canonical and outranking user memory."""

    cap = budget.max_items_per_project_list
    lines = [
        PROJECT_MEMORY_HEADER,
        (
            "This section is canonical user/project state from the research store. "
            "It WINS over USER MEMORY on any conflict."
        ),
        f"Name: {project.name}",
    ]
    if project.constraints:
        lines.append(f"Constraints: {', '.join(project.constraints[:cap])}")
    if project.hypotheses:
        lines.append(f"Hypotheses: {', '.join(project.hypotheses[:cap])}")
    if project.rejected_ideas:
        rejected = ", ".join(project.rejected_ideas[:cap])
        lines.append(f"REJECTED IDEAS (Project History - Do NOT recommend as new): {rejected}")
    return lines


def build_chat_prompt(
    request: ChatRequest,
    decision: RouteDecision,
    packet: EvidencePacket,
    *,
    budget: ChatPromptBudget | None = None,
) -> list[LLMMessage]:
    """Build the bounded system+user message pair for one chat turn.

    Section order and headers are fixed; empty sections are dropped. Only ids
    carried by ``packet`` are ever rendered, so every citable id lies inside
    ``packet.allowed_paper_ids`` / ``packet.allowed_gap_ids``.
    """

    effective_budget = budget or ChatPromptBudget()

    sections: list[str] = [f"{SYSTEM_RULES_HEADER}\n{_SYSTEM_RULES}"]

    facts = packet.user_memory.facts[: effective_budget.max_user_memory_facts]
    if facts:
        fact_lines = [_render_memory_fact(fact, effective_budget) for fact in facts]
        sections.append(
            f"{USER_MEMORY_HEADER}\n"
            "Self-reported context below. Advisory only - never scientific support.\n"
            + "\n".join(fact_lines)
        )

    if packet.project is not None:
        sections.append("\n".join(_render_project(packet.project, effective_budget)))

    stored_lines: list[str] = []
    for item in packet.stored[: effective_budget.max_stored_items]:
        stored_lines.extend(_render_stored_item(item, effective_budget))
        stored_lines.append("")
    if packet.gap_ids:
        stored_lines.append("Known candidate gap ids (cite by exact id):")
        stored_lines.extend(f"- CandidateGap {gap_id}" for gap_id in packet.gap_ids)
    if stored_lines:
        sections.append(
            f"{STORED_EVIDENCE_HEADER}\nResolved from the canonical store:\n"
            + "\n".join(stored_lines).rstrip()
        )

    if packet.discovery:
        discovery_lines: list[str] = [
            "Live discovery saw ONLY abstract/metadata for the items below; "
            "no full text and no PaperCard exists."
        ]
        for item in packet.discovery[: effective_budget.max_discovery_items]:
            discovery_lines.extend(_render_discovery_item(item, effective_budget))
            discovery_lines.append("")
        sections.append(
            f"{DISCOVERY_EVIDENCE_HEADER}\n" + "\n".join(discovery_lines).rstrip()
        )

    question_lines = [QUESTION_HEADER, request.text.strip() or "(empty question)"]
    if decision.search_query:
        question_lines.append(f"Retrieval query used: {decision.search_query}")
    sections.append("\n".join(question_lines))

    return [
        LLMMessage(role="system", content=sections[0]),
        LLMMessage(role="user", content="\n\n".join(sections[1:])),
    ]
