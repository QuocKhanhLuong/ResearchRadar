"""Tests for V2B Coverage Gap mining over attributable dimension matrices."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from research_radar.gap.coverage import CoverageGapMiner
from research_radar.gap.service import GapService
from research_radar.models.paper import Paper
from research_radar.models.paper_card import PaperCard
from research_radar.research.scout import ScoutResult
from research_radar.storage.database import Database
from research_radar.storage.repositories import (
    ResearchRepository,
    ScopedCorpusResult,
    StoredPaper,
    StoredPaperCard,
)


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class FakeScout:
    def __init__(self, return_papers: list[Paper] | None = None) -> None:
        self.return_papers = return_papers or []

    async def search(self, query: str, limit: int) -> ScoutResult:
        return ScoutResult(
            papers=self.return_papers,
            warnings=[],
            provider_counts={"fake": len(self.return_papers)},
        )


def test_coverage_gap_miner_finds_unobserved_combinations() -> None:
    now = _utc_now()

    p1 = StoredPaper(
        id="p1",
        canonical_key="k1",
        title="Diffusion for Brain MRI",
        abstract=None,
        authors=[],
        publication_year=2023,
        venue=None,
        doi=None,
        url=None,
        citation_count=0,
        primary_source="arxiv",
        sources=(),
        first_discovered_at=now,
        created_at=now,
        updated_at=now,
    )
    p2 = StoredPaper(
        id="p2",
        canonical_key="k2",
        title="Diffusion for Cardiac MRI",
        abstract=None,
        authors=[],
        publication_year=2024,
        venue=None,
        doi=None,
        url=None,
        citation_count=0,
        primary_source="arxiv",
        sources=(),
        first_discovered_at=now,
        created_at=now,
        updated_at=now,
    )
    p3 = StoredPaper(
        id="p3",
        canonical_key="k3",
        title="GAN for Liver MRI",
        abstract=None,
        authors=[],
        publication_year=2023,
        venue=None,
        doi=None,
        url=None,
        citation_count=0,
        primary_source="arxiv",
        sources=(),
        first_discovered_at=now,
        created_at=now,
        updated_at=now,
    )
    p4 = StoredPaper(
        id="p4",
        canonical_key="k4",
        title="Transformer for Liver MRI",
        abstract=None,
        authors=[],
        publication_year=2024,
        venue=None,
        doi=None,
        url=None,
        citation_count=0,
        primary_source="arxiv",
        sources=(),
        first_discovered_at=now,
        created_at=now,
        updated_at=now,
    )

    # p1, p2 use Diffusion method on Brain/Cardiac datasets
    # p3, p4 use GAN/Transformer methods on Liver dataset
    # Neither p1 nor p2 use Liver dataset (unobserved combination)
    card1 = StoredPaperCard(
        card=PaperCard(paper_id="p1", methods=["Diffusion Model"], datasets=["Brain MRI"]),
        source_url=None,
        document_sha256=None,
        selected_sections=(),
        llm_provider=None,
        llm_model=None,
        created_at=now,
        updated_at=now,
    )
    card2 = StoredPaperCard(
        card=PaperCard(paper_id="p2", methods=["Diffusion Model"], datasets=["Brain MRI"]),
        source_url=None,
        document_sha256=None,
        selected_sections=(),
        llm_provider=None,
        llm_model=None,
        created_at=now,
        updated_at=now,
    )
    card3 = StoredPaperCard(
        card=PaperCard(paper_id="p3", methods=["GAN"], datasets=["Liver MRI Dataset"]),
        source_url=None,
        document_sha256=None,
        selected_sections=(),
        llm_provider=None,
        llm_model=None,
        created_at=now,
        updated_at=now,
    )
    card4 = StoredPaperCard(
        card=PaperCard(paper_id="p4", methods=["Transformer"], datasets=["Liver MRI Dataset"]),
        source_url=None,
        document_sha256=None,
        selected_sections=(),
        llm_provider=None,
        llm_model=None,
        created_at=now,
        updated_at=now,
    )

    corpus = ScopedCorpusResult(
        cards=(card1, card2, card3, card4),
        papers=(p1, p2, p3, p4),
        corpus_paper_ids=("p1", "p2", "p3", "p4"),
        missing_cards_paper_ids=(),
        total_matching_papers=4,
    )

    miner = CoverageGapMiner()
    candidates = miner.mine_coverage_gaps("MRI synthesis", corpus)

    assert len(candidates) >= 1
    cand = candidates[0]
    assert cand.gap_type == "coverage"
    is_unobserved = (
        "no evidence was observed" in cand.description.lower()
        or "unobserved" in cand.title.lower()
    )
    assert is_unobserved


@pytest.mark.asyncio
async def test_gap_service_supports_coverage_type(tmp_path_factory: object) -> None:
    db_file = tmp_path_factory.mktemp("db") / "test_coverage.db"  # type: ignore[attr-defined]
    db = Database.create(f"sqlite:///{db_file}")
    db.initialize_schema()
    repo = ResearchRepository(db)

    p1 = Paper(id="p1", title="Diffusion Brain", source="arxiv")
    p2 = Paper(id="p2", title="Diffusion Cardiac", source="arxiv")
    p3 = Paper(id="p3", title="GAN Liver", source="openalex")
    p4 = Paper(id="p4", title="Transformer Liver", source="openalex")

    id1 = repo.upsert_merged_paper(p1)
    id2 = repo.upsert_merged_paper(p2)
    id3 = repo.upsert_merged_paper(p3)
    id4 = repo.upsert_merged_paper(p4)

    card1 = PaperCard(
        paper_id=id1, problem="MRI", methods=["Diffusion Model"], datasets=["Brain Dataset"]
    )
    card2 = PaperCard(
        paper_id=id2, problem="MRI", methods=["Diffusion Model"], datasets=["Brain Dataset"]
    )
    card3 = PaperCard(
        paper_id=id3, problem="MRI", methods=["GAN"], datasets=["Liver Dataset"]
    )
    card4 = PaperCard(
        paper_id=id4, problem="MRI", methods=["Transformer"], datasets=["Liver Dataset"]
    )

    repo.upsert_paper_card(card1)
    repo.upsert_paper_card(card2)
    repo.upsert_paper_card(card3)
    repo.upsert_paper_card(card4)

    fake_scout = FakeScout(return_papers=[])
    gap_service = GapService(repo, fake_scout)

    res = await gap_service.analyze_gaps("MRI", count=1, gap_type="coverage")

    assert not res.is_insufficient_evidence
    assert len(res.candidates) == 1
    assert res.candidates[0].gap_type == "coverage"

    db.dispose()
