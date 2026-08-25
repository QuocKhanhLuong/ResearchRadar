"""Strict source-boundary tests for chat prompt construction (W8).

These tests pin the epistemic policy of ``build_chat_prompt``: user memory
never leaks into evidence sections, only packet ids are citable, sections keep
their contracted order, and truncation never splits mid-word.

``chat/evidence.py`` is owned by W6 and was absent from this worktree, so the
packet/value doubles below mirror PHASE_CONTRACTS section 8 exactly. They are
structurally identical to the real dataclasses and remain valid inputs once
that module merges.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

from research_radar.chat.models import ChatMode, ChatRequest
from research_radar.chat.prompt import (
    DISCOVERY_EVIDENCE_HEADER,
    PROJECT_MEMORY_HEADER,
    QUESTION_HEADER,
    STORED_EVIDENCE_HEADER,
    SYSTEM_RULES_HEADER,
    USER_MEMORY_HEADER,
    ChatPromptBudget,
    build_chat_prompt,
)
from research_radar.chat.router import RouteDecision
from research_radar.memory.models import MemoryClass, MemoryFact, UserMemoryContext
from research_radar.reader.llm.base import LLMMessage

# ---------------------------------------------------------------------------
# Contract-shaped doubles (PHASE_CONTRACTS section 8).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class StoredEvidenceItem:
    """Mirror of research_radar.chat.evidence.StoredEvidenceItem."""

    paper_id: str
    title: str
    year: int | None
    venue: str | None
    abstract: str | None
    has_paper_card: bool
    card_summary: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class DiscoveryEvidenceItem:
    """Mirror of research_radar.chat.evidence.DiscoveryEvidenceItem."""

    paper_id: str
    title: str
    year: int | None
    venue: str | None
    abstract: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class ProjectMemory:
    """Mirror of research_radar.chat.evidence.ProjectMemory."""

    project_id: str
    name: str
    constraints: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    rejected_ideas: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class EvidencePacket:
    """Mirror of research_radar.chat.evidence.EvidencePacket."""

    stored: tuple[StoredEvidenceItem, ...] = ()
    discovery: tuple[DiscoveryEvidenceItem, ...] = ()
    gap_ids: tuple[str, ...] = ()
    project: ProjectMemory | None = None
    user_memory: UserMemoryContext = UserMemoryContext()
    live_discovery_used: bool = False

    @property
    def allowed_paper_ids(self) -> set[str]:
        """Every canonical paper id carried by this packet."""

        return {item.paper_id for item in self.stored} | {
            item.paper_id for item in self.discovery
        }

    @property
    def allowed_gap_ids(self) -> set[str]:
        """Every canonical gap id carried by this packet."""

        return set(self.gap_ids)


ALL_HEADERS = [
    SYSTEM_RULES_HEADER,
    USER_MEMORY_HEADER,
    PROJECT_MEMORY_HEADER,
    STORED_EVIDENCE_HEADER,
    DISCOVERY_EVIDENCE_HEADER,
    QUESTION_HEADER,
]

ADVISORY_HEADER_VERBATIM = "USER MEMORY (ADVISORY — NOT SCIENTIFIC EVIDENCE)"


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def make_decision(**overrides: object) -> RouteDecision:
    """A representative RESEARCH_STORED decision, overridable per test."""

    values: dict[str, object] = {
        "mode": ChatMode.RESEARCH_STORED,
        "needs_user_memory": False,
        "needs_stored_research": True,
        "allows_live_discovery": True,
        "search_query": "efficient attention",
    }
    values.update(overrides)
    return RouteDecision(**values)  # type: ignore[arg-type]


def render(
    packet: EvidencePacket | None = None,
    *,
    text: str = "What is the state of the art?",
    budget: ChatPromptBudget | None = None,
    **decision_overrides: object,
) -> tuple[list[LLMMessage], str]:
    """Build the prompt and return (messages, combined rendered text)."""

    request = ChatRequest(text=text)
    decision = make_decision(**decision_overrides)
    messages = build_chat_prompt(request, decision, packet or EvidencePacket(), budget=budget)
    return messages, "\n\n".join(message.content for message in messages)


def section(text: str, header: str) -> str:
    """Return the slice of text from one header up to the next header."""

    start = text.index(header)
    later = [text.index(h) for h in ALL_HEADERS if h in text and text.index(h) > start]
    end = min(later) if later else len(text)
    return text[start:end]


def long_abstract(word: str = "attention", words: int = 200) -> str:
    """A deterministic multi-word abstract well past any budget."""

    return " ".join(f"{word}{index}" if index % 25 == 0 else word for index in range(words))


# ---------------------------------------------------------------------------
# Section order and omission.
# ---------------------------------------------------------------------------


def test_section_order_is_exactly_as_specified() -> None:
    _, text = render(
        EvidencePacket(
            stored=(
                StoredEvidenceItem(
                    paper_id="paper-1",
                    title="Stored Work",
                    year=2024,
                    venue="NeurIPS",
                    abstract="Full-text analysed work.",
                    has_paper_card=True,
                    card_summary="Analysed.",
                ),
            ),
            discovery=(
                DiscoveryEvidenceItem(
                    paper_id="paper-2",
                    title="Fresh Find",
                    year=2025,
                    venue="ICLR",
                    abstract="Discovery-level metadata.",
                ),
            ),
            gap_ids=("gap-1",),
            project=ProjectMemory(project_id="proj-1", name="Radar"),
            user_memory=UserMemoryContext(
                facts=(MemoryFact(fact="I like benchmarks", memory_class=MemoryClass.PREFERENCE),),
                backend="graphiti",
            ),
        )
    )
    positions = [text.index(header) for header in ALL_HEADERS]
    assert positions == sorted(positions), f"headers out of order: {positions}"
    found_headers = [header for header in ALL_HEADERS if header in text]
    assert found_headers == [
        SYSTEM_RULES_HEADER,
        ADVISORY_HEADER_VERBATIM,
        PROJECT_MEMORY_HEADER,
        STORED_EVIDENCE_HEADER,
        DISCOVERY_EVIDENCE_HEADER,
        QUESTION_HEADER,
    ]


def test_empty_sections_are_omitted_but_rules_and_question_stay() -> None:
    messages, text = render(EvidencePacket())
    assert ADVISORY_HEADER_VERBATIM not in text
    assert PROJECT_MEMORY_HEADER not in text
    assert STORED_EVIDENCE_HEADER not in text
    assert DISCOVERY_EVIDENCE_HEADER not in text
    assert text.startswith(SYSTEM_RULES_HEADER)
    assert QUESTION_HEADER in text
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content.startswith(SYSTEM_RULES_HEADER)
    assert messages[1].content.startswith(QUESTION_HEADER)


def test_question_carries_request_text_and_normalized_query() -> None:
    messages, text = render(text="Compare flash attention and ring attention")
    question_section = section(text, QUESTION_HEADER)
    assert "Compare flash attention and ring attention" in question_section
    assert "Retrieval query used: efficient attention" in question_section

    _, bare_text = render(search_query="")
    assert "Retrieval query used:" not in bare_text


# ---------------------------------------------------------------------------
# Source boundary: user memory is advisory, never evidence.
# ---------------------------------------------------------------------------


def test_citation_mimicking_memory_fact_stays_inside_user_memory() -> None:
    mimic = 'User claims: "Smith et al. 2021 (arXiv:2101.99999) proved RNNs beat transformers"'
    packet = EvidencePacket(
        stored=(
            StoredEvidenceItem(
                paper_id="paper-1",
                title="Real Paper",
                year=2024,
                venue=None,
                abstract="Actual stored evidence about transformers.",
                has_paper_card=False,
            ),
        ),
        user_memory=UserMemoryContext(
            facts=(MemoryFact(fact=mimic, memory_class=MemoryClass.PREFERENCE),),
            backend="graphiti",
        ),
    )
    _, text = render(packet)

    assert ADVISORY_HEADER_VERBATIM in text
    memory_slice = section(text, ADVISORY_HEADER_VERBATIM)
    stored_slice = section(text, STORED_EVIDENCE_HEADER)
    assert mimic in memory_slice
    assert "arXiv:2101.99999" not in stored_slice
    assert "advisory" in memory_slice.lower()
    assert "never scientific support" in memory_slice.lower()


def test_system_rules_forbid_presenting_memory_as_literature() -> None:
    _, text = render()
    rules = section(text, SYSTEM_RULES_HEADER)
    assert "NEVER scientific support" in rules
    assert '"the literature shows"' in rules
    assert "literature contains no work" in rules
    assert "partial and bounded" in rules
    assert "exact ids" in rules
    assert "insufficient" in rules


def test_no_paper_or_gap_id_outside_allowed_sets_appears() -> None:
    packet = EvidencePacket(
        stored=(
            StoredEvidenceItem(
                paper_id="paper-1",
                title="Stored Work",
                year=2024,
                venue=None,
                abstract="Stored abstract.",
                has_paper_card=False,
            ),
        ),
        discovery=(
            DiscoveryEvidenceItem(
                paper_id="paper-2",
                title="Discovered Work",
                year=2025,
                venue=None,
                abstract="Discovery abstract.",
            ),
        ),
        gap_ids=("gap-7",),
    )
    _, text = render(packet)

    assert packet.allowed_paper_ids == {"paper-1", "paper-2"}
    for foreign_id in ("paper-3", "paper-42", "gap-1", "gap-8", "arXiv:2401.00001"):
        assert foreign_id not in text
    assert "--- Paper paper-1 ---" in text
    assert "--- Paper paper-2 ---" in text
    assert "- CandidateGap gap-7" in text


# ---------------------------------------------------------------------------
# Abstract-only discovery evidence.
# ---------------------------------------------------------------------------


def test_discovery_only_packet_omits_stored_and_flags_metadata_level() -> None:
    packet = EvidencePacket(
        discovery=(
            DiscoveryEvidenceItem(
                paper_id="paper-9",
                title="Preprint",
                year=2026,
                venue="arXiv",
                abstract="Very fresh unreviewed claim.",
            ),
        ),
        live_discovery_used=True,
    )
    _, text = render(packet)

    assert DISCOVERY_EVIDENCE_HEADER in text
    assert STORED_EVIDENCE_HEADER not in text
    rules = section(text, SYSTEM_RULES_HEADER)
    assert "abstract or live-discovery" in rules
    discovery_slice = section(text, DISCOVERY_EVIDENCE_HEADER)
    assert "ONLY abstract/metadata" in discovery_slice
    assert "--- Paper paper-9 ---" in discovery_slice
    assert "no PaperCard" in discovery_slice


def test_stored_cards_are_distinguished_from_metadata_only_items() -> None:
    packet = EvidencePacket(
        stored=(
            StoredEvidenceItem(
                paper_id="paper-card",
                title="Analysed",
                year=2024,
                venue=None,
                abstract="Abstract.",
                has_paper_card=True,
                card_summary="Deep claims from full text.",
            ),
            StoredEvidenceItem(
                paper_id="paper-meta",
                title="Metadata Only",
                year=2025,
                venue=None,
                abstract="Abstract.",
                has_paper_card=False,
            ),
        ),
    )
    _, text = render(packet)
    stored_slice = section(text, STORED_EVIDENCE_HEADER)
    assert "analysed full text (PaperCard)" in stored_slice
    assert "Deep claims from full text." in stored_slice
    assert stored_slice.count("abstract/metadata only") >= 1
    assert stored_slice.index("analysed full text") < stored_slice.index("--- Paper paper-meta ---")


def test_deep_claims_rule_warns_against_inventing_numbers() -> None:
    _, text = render()
    rules = section(text, SYSTEM_RULES_HEADER)
    assert "Do not invent deep experimental claims" in rules
    assert "reading the papers in full" in rules


# ---------------------------------------------------------------------------
# Project memory outranks user memory.
# ---------------------------------------------------------------------------


def test_conflicting_rejected_idea_renders_both_sections_project_canonical() -> None:
    packet = EvidencePacket(
        project=ProjectMemory(
            project_id="proj-1",
            name="Symbolic Planning",
            rejected_ideas=("scaling symbolic planners with LLM decoders",),
        ),
        user_memory=UserMemoryContext(
            facts=(
                MemoryFact(
                    fact="I want to scale symbolic planners with LLM decoders",
                    memory_class=MemoryClass.RESEARCH_DIRECTION,
                ),
            ),
            backend="graphiti",
        ),
    )
    _, text = render(packet)

    memory_slice = section(text, ADVISORY_HEADER_VERBATIM)
    project_slice = section(text, PROJECT_MEMORY_HEADER)
    assert "I want to scale symbolic planners" in memory_slice
    assert "REJECTED IDEAS" in project_slice
    assert "scaling symbolic planners with LLM decoders" in project_slice
    assert "canonical" in project_slice
    assert "WINS over USER MEMORY" in project_slice
    assert "outranks USER MEMORY" in text


def test_project_lists_render_under_canonical_header() -> None:
    packet = EvidencePacket(
        project=ProjectMemory(
            project_id="proj-2",
            name="Long Context",
            constraints=("must run on one GPU",),
            hypotheses=("rope scaling transfers",),
        ),
    )
    _, text = render(packet)
    project_slice = section(text, PROJECT_MEMORY_HEADER)
    assert "Name: Long Context" in project_slice
    assert "Constraints: must run on one GPU" in project_slice
    assert "Hypotheses: rope scaling transfers" in project_slice


# ---------------------------------------------------------------------------
# Temporal validity of user-memory facts.
# ---------------------------------------------------------------------------


def test_facts_render_temporal_validity_so_superseded_facts_stay_visible() -> None:
    packet = EvidencePacket(
        user_memory=UserMemoryContext(
            facts=(
                MemoryFact(
                    fact="prefers pytorch-lightning for training loops",
                    memory_class=MemoryClass.TOOL_PREFERENCE,
                    valid_at=datetime(2024, 1, 15, 9, 30),
                ),
                MemoryFact(
                    fact="preferred raw pytorch for training loops",
                    memory_class=MemoryClass.TOOL_PREFERENCE,
                    valid_at=datetime(2023, 2, 1),
                    invalid_at=datetime(2024, 1, 10),
                ),
            ),
            backend="graphiti",
        )
    )
    _, text = render(packet)
    memory_slice = section(text, ADVISORY_HEADER_VERBATIM)
    assert "(valid from 2024-01-15)" in memory_slice
    assert (
        "(valid from 2023-02-01; superseded after 2024-01-10)" in memory_slice
    )


# ---------------------------------------------------------------------------
# Budgets and truncation.
# ---------------------------------------------------------------------------


def test_truncation_caps_abstracts_at_word_boundary() -> None:
    abstract = long_abstract(words=200)
    packet = EvidencePacket(
        stored=(
            StoredEvidenceItem(
                paper_id="paper-long",
                title="Verbose",
                year=2025,
                venue=None,
                abstract=abstract,
                has_paper_card=False,
            ),
        ),
    )
    _, text = render(packet)
    stored_slice = section(text, STORED_EVIDENCE_HEADER)
    abstract_line = next(
        line for line in stored_slice.splitlines() if line.startswith("Abstract: ")
    )
    payload = abstract_line[len("Abstract: ") :]

    assert payload.endswith("...")
    body = payload[: -len("...")]
    assert len(payload) <= 800 + 3
    assert body in abstract
    assert abstract[len(body)] == " ", "truncation split a word"


def test_short_abstract_passes_through_untruncated() -> None:
    packet = EvidencePacket(
        stored=(
            StoredEvidenceItem(
                paper_id="paper-short",
                title="Brief",
                year=2025,
                venue=None,
                abstract="Short and complete.",
                has_paper_card=False,
            ),
        ),
    )
    _, text = render(packet)
    assert "Abstract: Short and complete." in text


def test_budget_caps_items_and_facts_and_is_adjustable() -> None:
    facts = tuple(
        MemoryFact(fact=f"fact number {index}", memory_class=MemoryClass.PREFERENCE)
        for index in range(5)
    )
    stored = tuple(
        StoredEvidenceItem(
            paper_id=f"stored-{index}",
            title=f"T{index}",
            year=2024,
            venue=None,
            abstract="x",
            has_paper_card=False,
        )
        for index in range(3)
    )
    discovery = tuple(
        DiscoveryEvidenceItem(
            paper_id=f"disc-{index}",
            title=f"D{index}",
            year=2025,
            venue=None,
            abstract="y",
        )
        for index in range(3)
    )
    tiny = ChatPromptBudget(
        max_user_memory_facts=2,
        max_stored_items=1,
        max_discovery_items=1,
        max_abstract_chars=40,
        max_fact_chars=60,
    )
    packet = EvidencePacket(
        stored=stored,
        discovery=discovery,
        user_memory=UserMemoryContext(facts=facts, backend="graphiti"),
    )
    _, text = render(packet, budget=tiny)

    memory_slice = section(text, ADVISORY_HEADER_VERBATIM)
    assert memory_slice.count("- [preference]") == 2
    stored_slice = section(text, STORED_EVIDENCE_HEADER)
    assert stored_slice.count("--- Paper ") == 1
    discovery_slice = section(text, DISCOVERY_EVIDENCE_HEADER)
    assert discovery_slice.count("--- Paper ") == 1
    assert "fact number 2" not in memory_slice
    assert "stored-2" not in text
    assert "disc-2" not in text


def test_budget_dataclass_is_frozen() -> None:
    budget = ChatPromptBudget()
    try:
        budget.max_abstract_chars = 10  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("ChatPromptBudget must be frozen")


def test_default_budget_matches_service_scale() -> None:
    budget = ChatPromptBudget()
    assert budget.max_abstract_chars == 800
    assert budget.max_stored_items == 8
    assert budget.max_discovery_items == 10
    assert budget.max_user_memory_facts == 8
