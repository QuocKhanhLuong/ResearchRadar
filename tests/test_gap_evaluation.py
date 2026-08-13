"""Tests for V2C Evaluation Gap mining over underrepresented conditions."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from research_radar.gap.evaluation import EvaluationGapMiner
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


def test_evaluation_gap_miner_finds_underrepresented_conditions() -> None:
    now = _utc_now()

    p1 = StoredPaper(
        id="p1",
        canonical_key="k1",
        title="Super Resolution Model A",
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
        title="Super Resolution Model B",
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

    # Both papers evaluate Super Resolution CNN using PSNR/SSIM, but NEITHER tests scanner shift
    card1 = StoredPaperCard(
        card=PaperCard(
            paper_id="p1", methods=["Super Resolution CNN"], metrics=["PSNR", "SSIM"]
        ),
        source_url=None,
        document_sha256=None,
        selected_sections=(),
        llm_provider=None,
        llm_model=None,
        created_at=now,
        updated_at=now,
    )
    card2 = StoredPaperCard(
        card=PaperCard(
            paper_id="p2", methods=["Super Resolution CNN"], metrics=["PSNR", "SSIM"]
        ),
        source_url=None,
        document_sha256=None,
        selected_sections=(),
        llm_provider=None,
        llm_model=None,
        created_at=now,
        updated_at=now,
    )

    corpus = ScopedCorpusResult(
        cards=(card1, card2),
        papers=(p1, p2),
        corpus_paper_ids=("p1", "p2"),
        missing_cards_paper_ids=(),
        total_matching_papers=2,
    )

    miner = EvaluationGapMiner()
    candidates = miner.mine_evaluation_gaps("Super Resolution", corpus)

    assert len(candidates) >= 1
    cand = candidates[0]
    assert cand.gap_type == "evaluation"
    is_underrep = (
        "underrepresented" in cand.title.lower()
        or "underrepresented" in cand.description.lower()
    )
    assert is_underrep


@pytest.mark.asyncio
async def test_gap_service_supports_evaluation_type(tmp_path_factory: object) -> None:
    db_file = tmp_path_factory.mktemp("db") / "test_eval.db"  # type: ignore[attr-defined]
    db = Database.create(f"sqlite:///{db_file}")
    db.initialize_schema()
    repo = ResearchRepository(db)

    p1 = Paper(id="p1", title="SuperRes 1", source="arxiv")
    p2 = Paper(id="p2", title="SuperRes 2", source="arxiv")

    id1 = repo.upsert_merged_paper(p1)
    id2 = repo.upsert_merged_paper(p2)

    card1 = PaperCard(
        paper_id=id1, problem="Super Resolution", methods=["SR-CNN"], metrics=["PSNR"]
    )
    card2 = PaperCard(
        paper_id=id2, problem="Super Resolution", methods=["SR-CNN"], metrics=["SSIM"]
    )

    repo.upsert_paper_card(card1)
    repo.upsert_paper_card(card2)

    fake_scout = FakeScout(return_papers=[])
    gap_service = GapService(repo, fake_scout)

    res = await gap_service.analyze_gaps("Super Resolution", count=1, gap_type="evaluation")

    assert not res.is_insufficient_evidence
    assert len(res.candidates) == 1
    assert res.candidates[0].gap_type == "evaluation"

    db.dispose()
