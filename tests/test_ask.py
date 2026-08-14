"""Unit tests for Research Memory Question Answering (/ask V1)."""

from __future__ import annotations

from typing import TypeVar

import pytest
from pydantic import BaseModel

from research_radar.models.paper import Paper
from research_radar.reader.llm.base import LLMMessage
from research_radar.research.ask import AskLLMResponse, AskService, sanitize_llm_response
from research_radar.storage.database import Database
from research_radar.storage.repositories import ResearchRepository

ModelT = TypeVar("ModelT", bound=BaseModel)


class FakeLLMProvider:
    """Mock vendor-neutral LLMProvider for testing."""

    def __init__(self, response_text: str = "Mock answer based on stored papers.") -> None:
        self.response_text = response_text
        self.last_messages: list[LLMMessage] = []

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[ModelT],
    ) -> ModelT:
        self.last_messages = messages
        if response_model is AskLLMResponse:
            return AskLLMResponse(
                answer=self.response_text,
                referenced_paper_ids=["p1"],
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


@pytest.mark.asyncio
async def test_ask_service_fallback_without_llm(repo: ResearchRepository) -> None:
    p1 = Paper(
        id="p1",
        title="Spectral Regularization for MRI",
        abstract="We use spectral regularization for MRI reconstruction.",
        source="arxiv",
    )
    pid = repo.upsert_merged_paper(p1)

    service = AskService(repo, llm_provider=None)
    res = await service.ask("What methods are used for MRI?")

    assert res.is_sufficient_evidence
    assert pid in res.referenced_paper_ids
    assert "Based on the analyzed project corpus" in res.answer


@pytest.mark.asyncio
async def test_ask_service_calls_llm_with_evidence_context(repo: ResearchRepository) -> None:
    p1 = Paper(
        id="p1",
        title="Diffusion Models in Ultrasound",
        abstract="We evaluate diffusion models on 3D ultrasound data.",
        source="arxiv",
    )
    repo.upsert_merged_paper(p1)

    fake_llm = FakeLLMProvider(
        "Within the papers currently stored in ResearchRadar, diffusion models "
        "were evaluated on ultrasound."
    )
    service = AskService(repo, llm_provider=fake_llm)

    res = await service.ask("How are diffusion models used in ultrasound?")

    assert res.is_sufficient_evidence
    assert "Within the papers currently stored in ResearchRadar" in res.answer
    assert len(fake_llm.last_messages) == 2
    assert "STRICT RULES" in fake_llm.last_messages[0].content


@pytest.mark.asyncio
async def test_ask_service_scoped_to_project(repo: ResearchRepository) -> None:
    proj = repo.create_project(name="Ultrasound Project", goal="Improve ultrasound resolution")
    p1 = Paper(
        id="p1",
        title="Resolution in Ultrasound",
        abstract="Improving spatial resolution in ultrasound.",
        source="arxiv",
    )
    pid = repo.upsert_merged_paper(p1)
    repo.add_paper_to_project(proj.id, pid)

    fake_llm = FakeLLMProvider("Project goal focused on resolution.")
    service = AskService(repo, llm_provider=fake_llm)

    res = await service.ask("What is the project goal?", project_id_or_name="Ultrasound Project")

    assert res.is_sufficient_evidence
    assert "Project Scope" in fake_llm.last_messages[1].content
