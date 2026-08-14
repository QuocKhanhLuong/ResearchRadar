"""Comprehensive unit tests for Research Memory Question Answering (/ask V1.2 Hardened)."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TypeVar

import pytest
from pydantic import BaseModel

from research_radar.models.gap import CandidateGap, GapProvenance
from research_radar.models.paper import Paper
from research_radar.models.paper_card import EvidenceClaim, PaperCard, StructuredEvidence
from research_radar.reader.llm.base import LLMMessage
from research_radar.research.ask import (
    AskBudget,
    AskContext,
    AskLLMResponse,
    AskService,
    format_evidence_packet,
    sanitize_llm_response,
)
from research_radar.storage.database import Database
from research_radar.storage.repositories import ResearchRepository, StoredPaper

ModelT = TypeVar("ModelT", bound=BaseModel)


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class FakeLLMProvider:
    """Mock vendor-neutral LLMProvider for testing."""

    def __init__(
        self,
        response_text: str = "Mock answer based on stored papers.",
        referenced_paper_ids: list[str] | None = None,
        referenced_gap_ids: list[str] | None = None,
        raise_exception: bool = False,
    ) -> None:
        self.response_text = response_text
        self.referenced_paper_ids = referenced_paper_ids
        self.referenced_gap_ids = referenced_gap_ids
        self.raise_exception = raise_exception
        self.last_messages: list[LLMMessage] = []

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[ModelT],
    ) -> ModelT:
        self.last_messages = messages
        if self.raise_exception:
            raise RuntimeError("Underlying LLM service unavailable.")

        if response_model is AskLLMResponse:
            pids = self.referenced_paper_ids
            gids = self.referenced_gap_ids
            if pids is None and len(messages) >= 2:
                pids = re.findall(r"--- Paper ([^\s]+) ---", messages[1].content)
            if gids is None and len(messages) >= 2:
                gids = re.findall(r"--- CandidateGap ([^\s]+) ---", messages[1].content)
            return AskLLMResponse(
                answer=self.response_text,
                referenced_paper_ids=pids or [],
                referenced_gap_ids=gids or [],
                is_sufficient_evidence=True,
            )  # type: ignore[return-value]
        raise ValueError(f"Unsupported model {response_model}")


@pytest.fixture
def repo(tmp_path_factory: object) -> ResearchRepository:
    db_file = tmp_path_factory.mktemp("db") / "test_ask.db"  # type: ignore[attr-defined]
    db = Database.create(f"sqlite:///{db_file}")
    db.initialize_schema()
    repository = ResearchRepository(db)
    yield repository
    db.dispose()


def test_sanitize_llm_response_removes_forbidden_language() -> None:
    raw = "No one has studied spectral regularization in MRI. This is the first paper to try it."
    cleaned = sanitize_llm_response(raw)

    assert "No one has studied" not in cleaned
    assert "This is the first" not in cleaned
    assert "Within the papers currently stored in ResearchRadar" in cleaned


# 1. Relevant project seed still receives a useful boost and outranks slightly stronger global match
@pytest.mark.asyncio
async def test_relevant_project_seed_outranks_slightly_stronger_global_match(
    repo: ResearchRepository,
) -> None:
    proj = repo.create_project(name="MRI Reconstruction Project", goal="Reconstruct MRI fast")

    # Global paper with stronger lexical score (matches "mri", "enhancement" = 6.0)
    p_global = Paper(
        id="p-global",
        title="MRI Enhancement Methods",
        abstract="A general method.",
        source="arxiv",
    )
    pid_global = repo.upsert_merged_paper(p_global)

    # Project paper with slightly weaker lexical match (title "mri" = 3.0,
    # abstract "enhancement" = 2.0 -> 5.0)
    p_proj = Paper(
        id="p-proj",
        title="Spectral MRI Regularization",
        abstract="A regularizer for image enhancement.",
        source="arxiv",
    )
    pid_proj = repo.upsert_merged_paper(p_proj)
    repo.add_paper_to_project(proj.id, pid_proj, relation="seed")

    fake_llm = FakeLLMProvider()
    service = AskService(repo, llm_provider=fake_llm)

    res = await service.ask("mri enhancement", project_id_or_name="MRI Reconstruction Project")

    assert pid_proj in res.referenced_paper_ids
    assert pid_global in res.referenced_paper_ids
    assert res.referenced_paper_ids[0] == pid_proj


# 2. Unrelated seed paper with 0 match does NOT outrank highly relevant global paper
@pytest.mark.asyncio
async def test_unrelated_seed_paper_does_not_outrank_highly_relevant_global_paper(
    repo: ResearchRepository,
) -> None:
    proj = repo.create_project(name="MRI Project", goal="MRI Study")

    # Project seed paper with completely unrelated text (0 matches for query "diffusion mri")
    p_unrelated_seed = Paper(
        id="p-unrelated-seed",
        title="Genomics Data Preprocessing Pipeline",
        abstract="DNA sequencing algorithms.",
        source="arxiv",
    )
    pid_seed = repo.upsert_merged_paper(p_unrelated_seed)
    repo.add_paper_to_project(proj.id, pid_seed, relation="seed")

    # Highly relevant global paper (matches "diffusion", "mri")
    p_global = Paper(
        id="p-global-mri",
        title="Diffusion Priors for Accelerated MRI",
        abstract="Diffusion methods in MRI.",
        source="arxiv",
    )
    pid_global = repo.upsert_merged_paper(p_global)

    fake_llm = FakeLLMProvider()
    service = AskService(repo, llm_provider=fake_llm)

    res = await service.ask("diffusion mri", project_id_or_name="MRI Project")

    # The unrelated seed paper must NOT be retrieved (due to zero relevance gate)
    assert pid_seed not in res.referenced_paper_ids
    assert res.referenced_paper_ids[0] == pid_global


# 3. Zero-match project paper does not consume max_evidence slot
@pytest.mark.asyncio
async def test_zero_match_project_paper_does_not_consume_max_evidence_slot(
    repo: ResearchRepository,
) -> None:
    proj = repo.create_project(name="Cardiac Project", goal="Cardiac MRI")

    # 3 unrelated project papers (0 lexical match for "segmentation")
    for i in range(3):
        p = Paper(
            id=f"p-zero-{i}",
            title=f"Unrelated Astronomy Physics Paper {i}",
            source="arxiv",
        )
        pid = repo.upsert_merged_paper(p)
        repo.add_paper_to_project(proj.id, pid, relation="seed")

    # 2 matching global papers
    p_match1 = Paper(id="p-seg-1", title="Cardiac Segmentation Methods", source="arxiv")
    pid_match1 = repo.upsert_merged_paper(p_match1)
    p_match2 = Paper(id="p-seg-2", title="Neural Network Segmentation", source="arxiv")
    pid_match2 = repo.upsert_merged_paper(p_match2)

    fake_llm = FakeLLMProvider()
    service = AskService(repo, llm_provider=fake_llm)

    res = await service.ask("segmentation", project_id_or_name="Cardiac Project", max_evidence=2)

    assert len(res.referenced_paper_ids) == 2
    assert pid_match1 in res.referenced_paper_ids
    assert pid_match2 in res.referenced_paper_ids


# 4. Project goal/hypothesis/rejected ideas still available even with no matching project papers
@pytest.mark.asyncio
async def test_project_memory_available_even_with_no_matching_project_papers(
    repo: ResearchRepository,
) -> None:
    repo.create_project(
        name="Project Zero Match",
        goal="Achieve 100x acceleration",
        hypotheses=["Sparse priors are sufficient"],
        constraints=["Latencies < 10ms"],
        rejected_ideas=["Pure RNN autoregression"],
    )

    # Only a global paper matches the query
    p_glob = Paper(id="p-glob-fast", title="Acceleration Algorithms", source="arxiv")
    repo.upsert_merged_paper(p_glob)

    fake_llm = FakeLLMProvider()
    service = AskService(repo, llm_provider=fake_llm)

    await service.ask("acceleration", project_id_or_name="Project Zero Match")

    prompt = fake_llm.last_messages[1].content
    assert "Achieve 100x acceleration" in prompt
    assert "Sparse priors are sufficient" in prompt
    assert "Latencies < 10ms" in prompt
    assert "Pure RNN autoregression" in prompt


# 5. AskContext is used as single source of truth for allowed IDs
@pytest.mark.asyncio
async def test_ask_context_is_single_source_of_truth(repo: ResearchRepository) -> None:
    p1 = Paper(id="p-source", title="Source Paper Title", source="arxiv")
    pid = repo.upsert_merged_paper(p1)

    cand = CandidateGap(
        id="cand-source",
        title="Source Gap Title",
        description="Desc",
        gap_type="explicit",
        research_question="RQ?",
        supporting_papers=[pid],
        evidence_count=1,
        search_scope="Scope",
        provenance=GapProvenance(retrievals=[], corpus_paper_ids=[pid], corpus_description="Desc"),
        created_at=_utc_now(),
    )
    repo.save_candidate(cand)

    service = AskService(repo)
    ctx = service.build_ask_context("Source Paper Title Source Gap Title")

    assert pid in ctx.allowed_paper_ids
    assert cand.id in ctx.allowed_gap_ids


# 6. Structured tasks, modalities, evaluation conditions, claims rendered with status
@pytest.mark.asyncio
async def test_structured_evidence_fields_and_statuses_rendered(repo: ResearchRepository) -> None:
    p1 = Paper(id="p-struct", title="Structural Robustness Paper", source="arxiv")
    pid = repo.upsert_merged_paper(p1)

    card = PaperCard(
        paper_id=pid,
        problem="Scanner domain shift degradation",
        tasks=[
            StructuredEvidence(value="segmentation", status="observed"),
            StructuredEvidence(value="tracking", status="explicitly_absent"),
        ],
        modalities=[
            StructuredEvidence(value="MRI", status="observed"),
            StructuredEvidence(value="CT", status="unknown"),
        ],
        evaluation_conditions=[
            StructuredEvidence(value="low_snr", status="observed"),
            StructuredEvidence(value="scanner_shift", status="explicitly_absent"),
        ],
        methods=["SpectralNorm"],
        main_claims=[EvidenceClaim(claim="Improves SNR by 4dB", source_section="Results")],
        limitations=["High memory requirement"],
    )
    repo.upsert_paper_card(card)

    service = AskService(repo)
    ctx = service.build_ask_context("Structural Robustness Paper")
    packet = format_evidence_packet(ctx, AskBudget())

    assert "segmentation [observed]" in packet
    assert "tracking [explicitly_absent]" in packet
    assert "MRI [observed]" in packet
    assert "CT [unknown]" in packet
    assert "low_snr [observed]" in packet
    assert "scanner_shift [explicitly_absent]" in packet
    assert '"Improves SNR by 4dB" [Results]' in packet
    assert "SpectralNorm" in packet
    assert "High memory requirement" in packet


# 7. Evidence packet budget enforced
def test_evidence_packet_budget_enforced() -> None:
    p = StoredPaper(
        id="p1",
        canonical_key="arxiv:p1",
        title="Very Long Title" * 10,
        abstract="Very Long Abstract" * 50,
        authors=["Author 1"],
        publication_year=2026,
        venue=None,
        doi=None,
        url=None,
        citation_count=0,
        primary_source="arxiv",
        sources=(),
        first_discovered_at=_utc_now(),
        created_at=_utc_now(),
        updated_at=_utc_now(),
    )
    ctx = AskContext(query="test", retrieved_papers=[p])
    budget = AskBudget(max_total_context_chars=120)
    packet = format_evidence_packet(ctx, budget)

    assert len(packet) <= 120
    assert packet.endswith("...")


# 8. Safe LLM error handling
@pytest.mark.asyncio
async def test_safe_llm_error_handling(repo: ResearchRepository) -> None:
    p1 = Paper(id="p-err", title="Error Test Paper", source="arxiv")
    pid = repo.upsert_merged_paper(p1)

    fake_llm = FakeLLMProvider(raise_exception=True)
    service = AskService(repo, llm_provider=fake_llm)

    res = await service.ask("Error Test Paper")

    assert not res.is_sufficient_evidence
    assert res.answer == "I couldn't synthesize an answer from the stored evidence right now."
    assert pid in res.referenced_paper_ids
