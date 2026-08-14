"""Comprehensive unit tests for Research Memory Question Answering (/ask V1.1)."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TypeVar

import pytest
from pydantic import BaseModel

from research_radar.models.gap import CandidateGap, CriticReview, GapProvenance
from research_radar.models.paper import Paper
from research_radar.reader.llm.base import LLMMessage
from research_radar.research.ask import AskLLMResponse, AskService, sanitize_llm_response
from research_radar.storage.database import Database
from research_radar.storage.repositories import ResearchRepository

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
    ) -> None:
        self.response_text = response_text
        self.referenced_paper_ids = referenced_paper_ids
        self.referenced_gap_ids = referenced_gap_ids
        self.last_messages: list[LLMMessage] = []

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[ModelT],
    ) -> ModelT:
        self.last_messages = messages
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


# 1. Project-linked paper outranks slightly stronger global lexical match
@pytest.mark.asyncio
async def test_project_linked_paper_outranks_stronger_global_lexical_match(
    repo: ResearchRepository,
) -> None:
    proj = repo.create_project(name="MRI Reconstruction Project", goal="Reconstruct MRI fast")

    p_global = Paper(
        id="p-global",
        title="Diffusion Model for MRI Reconstruction",
        abstract="A diffusion model applied to MRI reconstruction tasks.",
        source="arxiv",
    )
    repo.upsert_merged_paper(p_global)

    p_proj = Paper(
        id="p-proj",
        title="Spectral MRI Regularization",
        abstract="A method for MRI image enhancement.",
        source="arxiv",
    )
    pid = repo.upsert_merged_paper(p_proj)
    repo.add_paper_to_project(proj.id, pid, relation="seed")

    fake_llm = FakeLLMProvider()
    service = AskService(repo, llm_provider=fake_llm)

    res = await service.ask("diffusion model mri", project_id_or_name="MRI Reconstruction Project")

    assert res.referenced_paper_ids[0] == pid


# 2. Global evidence can still appear as secondary evidence
@pytest.mark.asyncio
async def test_global_evidence_appears_as_secondary_evidence(repo: ResearchRepository) -> None:
    proj = repo.create_project(name="Project Alpha", goal="Goal Alpha")

    p_proj = Paper(id="p-proj", title="Project Paper Title", source="arxiv")
    pid_proj = repo.upsert_merged_paper(p_proj)
    repo.add_paper_to_project(proj.id, pid_proj, relation="seed")

    p_glob = Paper(id="p-glob", title="Global Paper Title", source="arxiv")
    pid_glob = repo.upsert_merged_paper(p_glob)

    fake_llm = FakeLLMProvider()
    service = AskService(repo, llm_provider=fake_llm)

    res = await service.ask("Paper Title", project_id_or_name="Project Alpha", max_evidence=5)

    assert pid_proj in res.referenced_paper_ids
    assert pid_glob in res.referenced_paper_ids


# 3. Project-linked gap retrieved
@pytest.mark.asyncio
async def test_project_linked_gap_retrieved(repo: ResearchRepository) -> None:
    proj = repo.create_project(name="Gap Project", goal="Find Gaps")
    cand = CandidateGap(
        id="cand-1",
        title="Sample Gap Title",
        description="Sample Desc",
        gap_type="explicit",
        research_question="Sample RQ?",
        supporting_papers=[],
        evidence_count=1,
        search_scope="Scope",
        provenance=GapProvenance(retrievals=[], corpus_paper_ids=[], corpus_description="Desc"),
        created_at=_utc_now(),
    )
    repo.save_candidate(cand)
    repo.add_gap_to_project(proj.id, cand.id, status="active")

    fake_llm = FakeLLMProvider()
    service = AskService(repo, llm_provider=fake_llm)

    res = await service.ask("Sample Gap", project_id_or_name="Gap Project")
    assert cand.id in res.referenced_gap_ids


# 4. Latest Critic review included
@pytest.mark.asyncio
async def test_latest_critic_review_included_in_evidence(repo: ResearchRepository) -> None:
    cand = CandidateGap(
        id="cand-critic",
        title="Critic Test Gap",
        description="Desc",
        gap_type="coverage",
        research_question="RQ?",
        supporting_papers=[],
        evidence_count=1,
        search_scope="Scope",
        provenance=GapProvenance(retrievals=[], corpus_paper_ids=[], corpus_description="Desc"),
        created_at=_utc_now(),
    )
    repo.save_candidate(cand)

    rev1 = CriticReview(
        candidate_id=cand.id,
        review_version=1,
        decision="downgraded",
        rationale="Weak preliminary evidence",
        caveats=["Caveat 1"],
        created_at=_utc_now(),
    )
    rev2 = CriticReview(
        candidate_id=cand.id,
        review_version=2,
        decision="rejected",
        rationale="Direct contradictory evidence found in latest paper",
        caveats=["Invalidated by Paper X"],
        created_at=_utc_now(),
    )
    repo.save_critic_review(rev1)
    repo.save_critic_review(rev2)

    fake_llm = FakeLLMProvider()
    service = AskService(repo, llm_provider=fake_llm)

    await service.ask("Critic Test Gap")
    prompt_content = fake_llm.last_messages[1].content
    assert "Latest Critic Review (v2): decision=rejected" in prompt_content
    assert "Direct contradictory evidence found" in prompt_content


# 5. Rejected ideas included in AskContext
@pytest.mark.asyncio
async def test_rejected_ideas_included_in_ask_context(repo: ResearchRepository) -> None:
    repo.create_project(
        name="Project Beta",
        goal="Test Goal",
        rejected_ideas=["GAN reconstruction due to instability"],
    )

    fake_llm = FakeLLMProvider()
    service = AskService(repo, llm_provider=fake_llm)

    await service.ask("What are the project ideas?", project_id_or_name="Project Beta")
    assert "GAN reconstruction due to instability" in fake_llm.last_messages[1].content


# 6. Rejected idea is visible in LLM prompt
@pytest.mark.asyncio
async def test_rejected_idea_visible_in_llm_prompt(repo: ResearchRepository) -> None:
    repo.create_project(
        name="Project Gamma",
        rejected_ideas=["Pure MSE loss"],
    )
    fake_llm = FakeLLMProvider()
    service = AskService(repo, llm_provider=fake_llm)

    await service.ask("Any rejected ideas?", project_id_or_name="Project Gamma")
    prompt_user_msg = fake_llm.last_messages[1].content
    assert (
        "REJECTED IDEAS (Project History - Do NOT recommend as new): Pure MSE loss"
        in prompt_user_msg
    )


# 7. Project paper relation affects ranking
@pytest.mark.asyncio
async def test_project_paper_relation_affects_ranking(repo: ResearchRepository) -> None:
    proj = repo.create_project(name="Project Delta")

    p_bg = Paper(id="p-bg", title="Method Overview", source="arxiv")
    pid_bg = repo.upsert_merged_paper(p_bg)
    repo.add_paper_to_project(proj.id, pid_bg, relation="background")

    p_seed = Paper(id="p-seed", title="Method Overview", source="arxiv")
    pid_seed = repo.upsert_merged_paper(p_seed)
    repo.add_paper_to_project(proj.id, pid_seed, relation="seed")

    fake_llm = FakeLLMProvider()
    service = AskService(repo, llm_provider=fake_llm)

    res = await service.ask("Method Overview", project_id_or_name="Project Delta")
    assert res.referenced_paper_ids[0] == pid_seed


# 8. Resolved/rejected gap is not presented as active
@pytest.mark.asyncio
async def test_resolved_rejected_gap_not_presented_as_active(repo: ResearchRepository) -> None:
    proj = repo.create_project(name="Project Epsilon")

    cand = CandidateGap(
        id="cand-resolved",
        title="Resolved Gap",
        description="Desc",
        gap_type="evaluation",
        research_question="RQ?",
        supporting_papers=[],
        evidence_count=1,
        search_scope="Scope",
        provenance=GapProvenance(retrievals=[], corpus_paper_ids=[], corpus_description="Desc"),
        created_at=_utc_now(),
    )
    repo.save_candidate(cand)
    repo.add_gap_to_project(proj.id, cand.id, status="resolved")

    fake_llm = FakeLLMProvider()
    service = AskService(repo, llm_provider=fake_llm)

    await service.ask("Resolved Gap", project_id_or_name="Project Epsilon")
    user_prompt = fake_llm.last_messages[1].content
    assert "Project Gap Link Status: resolved (PAST RESOLVED/REJECTED GAP" in user_prompt


# 9. Fake LLM paper IDs removed
# 10. Fake LLM gap IDs removed
# 11. Valid IDs preserved
@pytest.mark.asyncio
async def test_fake_llm_ids_removed_and_valid_ids_preserved(repo: ResearchRepository) -> None:
    p1 = Paper(id="p-valid", title="Valid Paper", source="arxiv")
    pid = repo.upsert_merged_paper(p1)

    cand = CandidateGap(
        id="cand-valid",
        title="Valid Gap",
        description="Desc",
        gap_type="coverage",
        research_question="RQ?",
        supporting_papers=[pid],
        evidence_count=1,
        search_scope="Scope",
        provenance=GapProvenance(retrievals=[], corpus_paper_ids=[pid], corpus_description="Desc"),
        created_at=_utc_now(),
    )
    repo.save_candidate(cand)

    fake_llm = FakeLLMProvider(
        response_text="Answer with mixed IDs.",
        referenced_paper_ids=[pid, "p-FAKE-123", pid],
        referenced_gap_ids=["cand-FAKE-999", cand.id, "cand-FAKE-999"],
    )
    service = AskService(repo, llm_provider=fake_llm)

    res = await service.ask("Valid Paper Valid Gap")

    assert res.referenced_paper_ids == [pid]
    assert res.referenced_gap_ids == [cand.id]
    assert "p-FAKE-123" not in res.referenced_paper_ids
    assert "cand-FAKE-999" not in res.referenced_gap_ids


# 12. No-project /ask still works globally
@pytest.mark.asyncio
async def test_no_project_ask_works_globally(repo: ResearchRepository) -> None:
    p1 = Paper(id="p-global-only", title="Global MRI Segmentation", source="arxiv")
    pid = repo.upsert_merged_paper(p1)

    fake_llm = FakeLLMProvider("Answer using global paper.")
    service = AskService(repo, llm_provider=fake_llm)

    res = await service.ask("MRI Segmentation")
    assert pid in res.referenced_paper_ids
    assert res.is_sufficient_evidence


# 13. Bounded evidence count respected
@pytest.mark.asyncio
async def test_bounded_evidence_count_respected(repo: ResearchRepository) -> None:
    for i in range(15):
        p = Paper(id=f"p-many-{i}", title=f"Many Paper {i}", source="arxiv")
        repo.upsert_merged_paper(p)

    fake_llm = FakeLLMProvider()
    service = AskService(repo, llm_provider=fake_llm)

    res = await service.ask("Many Paper", max_evidence=4)
    assert len(res.referenced_paper_ids) <= 4
