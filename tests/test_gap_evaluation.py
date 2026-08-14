"""Tests for V2C Evaluation Gap mining over underrepresented conditions."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from research_radar.gap.evaluation import EvaluationGapMiner
from research_radar.gap.service import GapService
from research_radar.models.paper import Paper
from research_radar.models.paper_card import PaperCard, StructuredEvidence
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


def test_evaluation_gap_miner_finds_explicitly_absent_conditions() -> None:
    now = _utc_now()

    p1 = StoredPaper(
        id="p1", canonical_key="k1", title="Super Resolution Model A", abstract=None,
        authors=[], publication_year=2023, venue=None, doi=None, url=None, citation_count=0,
        primary_source="arxiv", sources=(), first_discovered_at=now, created_at=now, updated_at=now,
    )
    p2 = StoredPaper(
        id="p2", canonical_key="k2", title="Super Resolution Model B", abstract=None,
        authors=[], publication_year=2024, venue=None, doi=None, url=None, citation_count=0,
        primary_source="arxiv", sources=(), first_discovered_at=now, created_at=now, updated_at=now,
    )

    # p1 explicitly states scanner shift was not evaluated; p2 has unknown
    card1 = StoredPaperCard(
        card=PaperCard(
            paper_id="p1",
            methods=["Super Resolution CNN"],
            evaluation_conditions=[
                StructuredEvidence(
                    value="scanner shift",
                    status="explicitly_absent",
                    supporting_text="We do not evaluate under scanner shift.",
                )
            ],
        ),
        source_url=None, document_sha256=None, selected_sections=(),
        llm_provider=None, llm_model=None, created_at=now, updated_at=now,
    )
    card2 = StoredPaperCard(
        card=PaperCard(paper_id="p2", methods=["Super Resolution CNN"]),
        source_url=None, document_sha256=None, selected_sections=(),
        llm_provider=None, llm_model=None, created_at=now, updated_at=now,
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


def test_evaluation_gap_unknown_conditions_returns_no_candidates() -> None:
    now = _utc_now()

    p1 = StoredPaper(
        id="p1", canonical_key="k1", title="Model A", abstract=None,
        authors=[], publication_year=2023, venue=None, doi=None, url=None, citation_count=0,
        primary_source="arxiv", sources=(), first_discovered_at=now, created_at=now, updated_at=now,
    )
    p2 = StoredPaper(
        id="p2", canonical_key="k2", title="Model B", abstract=None,
        authors=[], publication_year=2024, venue=None, doi=None, url=None, citation_count=0,
        primary_source="arxiv", sources=(), first_discovered_at=now, created_at=now, updated_at=now,
    )

    # Both cards have unknown evaluation conditions (no explicit evaluation_conditions items)
    card1 = StoredPaperCard(
        card=PaperCard(paper_id="p1", methods=["SR-CNN"], metrics=["PSNR"]),
        source_url=None, document_sha256=None, selected_sections=(),
        llm_provider=None, llm_model=None, created_at=now, updated_at=now,
    )
    card2 = StoredPaperCard(
        card=PaperCard(paper_id="p2", methods=["SR-CNN"], metrics=["SSIM"]),
        source_url=None, document_sha256=None, selected_sections=(),
        llm_provider=None, llm_model=None, created_at=now, updated_at=now,
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

    # All unknown => insufficient evidence, no candidate!
    assert len(candidates) == 0


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
        paper_id=id1,
        problem="Super Resolution",
        methods=["SR-CNN"],
        evaluation_conditions=[
            StructuredEvidence(value="scanner shift", status="explicitly_absent")
        ],
    )
    card2 = PaperCard(
        paper_id=id2, problem="Super Resolution", methods=["SR-CNN"]
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


def test_evaluation_gap_attributable_evidence_refs_do_not_fabricate_text() -> None:
    now = _utc_now()
    p1 = StoredPaper(
        id="p1", canonical_key="k1", title="Model A", abstract=None, authors=[],
        publication_year=2024, venue=None, doi=None, url=None, citation_count=0,
        primary_source="arxiv", sources=(), first_discovered_at=now, created_at=now, updated_at=now,
    )
    p2 = StoredPaper(
        id="p2", canonical_key="k2", title="Model B", abstract=None, authors=[],
        publication_year=2024, venue=None, doi=None, url=None, citation_count=0,
        primary_source="arxiv", sources=(), first_discovered_at=now, created_at=now, updated_at=now,
    )

    card1 = StoredPaperCard(
        card=PaperCard(
            paper_id="p1",
            methods=["U-Net"],
            evaluation_conditions=[
                StructuredEvidence(
                    value="noise robustness",
                    status="explicitly_absent",
                    supporting_text="Noise robustness was not evaluated.",
                )
            ],
        ),
        source_url=None, document_sha256=None, selected_sections=(),
        llm_provider=None, llm_model=None, created_at=now, updated_at=now,
    )
    card2 = StoredPaperCard(
        card=PaperCard(paper_id="p2", methods=["U-Net"]),
        source_url=None, document_sha256=None, selected_sections=(),
        llm_provider=None, llm_model=None, created_at=now, updated_at=now,
    )

    corpus = ScopedCorpusResult(
        cards=(card1, card2),
        papers=(p1, p2),
        corpus_paper_ids=("p1", "p2"),
        missing_cards_paper_ids=(),
        total_matching_papers=2,
    )
    miner = EvaluationGapMiner()
    candidates = miner.mine_evaluation_gaps("Segmentation", corpus)

    assert len(candidates) >= 1
    for cand in candidates:
        for ref in cand.provenance.supporting_evidence:
            if ref.claim_or_field == "evaluation_conditions":
                assert ref.supporting_text == "Noise robustness was not evaluated."


def test_evaluation_gap_does_not_fabricate_supporting_text_when_none() -> None:
    now = _utc_now()
    p1 = StoredPaper(
        id="p1", canonical_key="k1", title="Model A", abstract=None, authors=[],
        publication_year=2024, venue=None, doi=None, url=None, citation_count=0,
        primary_source="arxiv", sources=(), first_discovered_at=now, created_at=now, updated_at=now,
    )
    p2 = StoredPaper(
        id="p2", canonical_key="k2", title="Model B", abstract=None, authors=[],
        publication_year=2024, venue=None, doi=None, url=None, citation_count=0,
        primary_source="arxiv", sources=(), first_discovered_at=now, created_at=now, updated_at=now,
    )

    card1 = StoredPaperCard(
        card=PaperCard(
            paper_id="p1",
            methods=["U-Net"],
            evaluation_conditions=[
                StructuredEvidence(
                    value="noise robustness",
                    status="explicitly_absent",
                    source_section="Limitations",
                    supporting_text=None,  # No supporting text extracted!
                )
            ],
        ),
        source_url=None, document_sha256=None, selected_sections=(),
        llm_provider=None, llm_model=None, created_at=now, updated_at=now,
    )
    card2 = StoredPaperCard(
        card=PaperCard(paper_id="p2", methods=["U-Net"]),
        source_url=None, document_sha256=None, selected_sections=(),
        llm_provider=None, llm_model=None, created_at=now, updated_at=now,
    )

    corpus = ScopedCorpusResult(
        cards=(card1, card2),
        papers=(p1, p2),
        corpus_paper_ids=("p1", "p2"),
        missing_cards_paper_ids=(),
        total_matching_papers=2,
    )
    miner = EvaluationGapMiner()
    candidates = miner.mine_evaluation_gaps("Segmentation", corpus)

    assert len(candidates) >= 1
    eval_refs = [
        ref
        for ref in candidates[0].provenance.supporting_evidence
        if ref.claim_or_field == "evaluation_conditions"
    ]
    assert len(eval_refs) == 1
    assert eval_refs[0].supporting_text is None
    assert eval_refs[0].source_section == "Limitations"

