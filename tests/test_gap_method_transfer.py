"""Comprehensive unit tests for V2E Method-Transfer Gap Engine."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from research_radar.bot.commands.gap import render_gap_embed
from research_radar.gap.critic import CriticService
from research_radar.gap.method_transfer import (
    MethodTransferGapMiner,
    assess_transfer_feasibility,
    infer_method_class,
)
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


def _make_stored_card(
    paper_id: str,
    title: str,
    methods: list[str],
    task: str = "segmentation",
    modality: str = "mri",
    evaluation_condition: str | None = None,
    now: datetime | None = None,
) -> tuple[StoredPaper, StoredPaperCard]:
    if now is None:
        now = _utc_now()

    paper = StoredPaper(
        id=paper_id,
        canonical_key=f"key-{paper_id}",
        title=title,
        abstract=f"Abstract for {title}",
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

    eval_conds = []
    if evaluation_condition:
        eval_conds.append(StructuredEvidence(value=evaluation_condition, status="observed"))

    card = StoredPaperCard(
        card=PaperCard(
            paper_id=paper_id,
            problem=f"{task} problem",
            methods=methods,
            tasks=[StructuredEvidence(value=task, status="observed")],
            modalities=[StructuredEvidence(value=modality, status="observed")],
            evaluation_conditions=eval_conds,
        ),
        source_url=None,
        document_sha256=None,
        selected_sections=(),
        llm_provider=None,
        llm_model=None,
        created_at=now,
        updated_at=now,
    )

    return paper, card


# 1. task -> task transfer candidate works
def test_method_transfer_task_to_task_works() -> None:
    now = _utc_now()
    p1, c1 = _make_stored_card(
        "p1", "P1", ["Spectral Regularization"], task="segmentation", now=now
    )
    p2, c2 = _make_stored_card(
        "p2", "P2", ["Spectral Regularization"], task="segmentation", now=now
    )
    p3, c3 = _make_stored_card("p3", "P3", ["U-Net"], task="reconstruction", now=now)
    p4, c4 = _make_stored_card("p4", "P4", ["U-Net"], task="reconstruction", now=now)

    corpus = ScopedCorpusResult(
        cards=(c1, c2, c3, c4),
        papers=(p1, p2, p3, p4),
        corpus_paper_ids=("p1", "p2", "p3", "p4"),
        missing_cards_paper_ids=(),
        total_matching_papers=4,
    )

    miner = MethodTransferGapMiner()
    cands = miner.mine_transfer_gaps("Medical Imaging", corpus)

    assert len(cands) >= 1
    cand = cands[0]
    assert cand.gap_type == "method_transfer"
    assert "Potential method transfer" in cand.title


# 2. modality -> modality transfer candidate works
def test_method_transfer_modality_to_modality_works() -> None:
    now = _utc_now()
    p1, c1 = _make_stored_card("p1", "P1", ["Spectral Regularization"], modality="mri", now=now)
    p2, c2 = _make_stored_card("p2", "P2", ["Spectral Regularization"], modality="mri", now=now)
    p3, c3 = _make_stored_card("p3", "P3", ["U-Net"], modality="ct", now=now)
    p4, c4 = _make_stored_card("p4", "P4", ["U-Net"], modality="ct", now=now)

    corpus = ScopedCorpusResult(
        cards=(c1, c2, c3, c4),
        papers=(p1, p2, p3, p4),
        corpus_paper_ids=("p1", "p2", "p3", "p4"),
        missing_cards_paper_ids=(),
        total_matching_papers=4,
    )

    miner = MethodTransferGapMiner()
    cands = miner.mine_transfer_gaps("Medical Imaging", corpus)

    mod_cands = [c for c in cands if "ct" in c.title.lower()]
    assert len(mod_cands) >= 1


# 3. task -> modality transfer is NEVER generated
def test_task_to_modality_transfer_never_generated() -> None:
    res = assess_transfer_feasibility(
        method="Spectral Regularization",
        source_dimension="task",
        source_context="segmentation",
        target_dimension="modality",
        target_context="mri",
    )
    assert not res.is_feasible
    assert res.score == 0.0


# 4. modality -> evaluation_condition transfer is NEVER generated
def test_modality_to_evaluation_condition_transfer_never_generated() -> None:
    res = assess_transfer_feasibility(
        method="Spectral Regularization",
        source_dimension="modality",
        source_context="mri",
        target_dimension="evaluation_condition",
        target_context="scanner shift",
    )
    assert not res.is_feasible
    assert res.score == 0.0


# 5. Unknown-heavy target dimension produces no candidate
def test_method_transfer_unknown_heavy_target_produces_no_candidate() -> None:
    now = _utc_now()
    # 4 cards in corpus, but evaluation condition is observed in only 1 card (< 50% coverage)
    p1, c1 = _make_stored_card(
        "p1", "P1", ["Spectral Regularization"], evaluation_condition="scanner shift", now=now
    )
    p2, c2 = _make_stored_card(
        "p2", "P2", ["Spectral Regularization"], evaluation_condition="scanner shift", now=now
    )
    p3, c3 = _make_stored_card("p3", "P3", ["U-Net"], now=now)
    p4, c4 = _make_stored_card("p4", "P4", ["U-Net"], now=now)

    corpus = ScopedCorpusResult(
        cards=(c1, c2, c3, c4),
        papers=(p1, p2, p3, p4),
        corpus_paper_ids=("p1", "p2", "p3", "p4"),
        missing_cards_paper_ids=(),
        total_matching_papers=4,
    )

    miner = MethodTransferGapMiner()
    cands = miner.mine_transfer_gaps("Medical Imaging", corpus)
    eval_cands = [c for c in cands if "scanner shift" in c.title.lower()]
    assert len(eval_cands) == 0


# 6. Task-specific head rejected for unrelated task
def test_task_specific_head_rejected_for_cross_task() -> None:
    res = assess_transfer_feasibility(
        method="YOLOv8 bounding-box detector head",
        source_dimension="task",
        source_context="detection",
        target_dimension="task",
        target_context="reconstruction",
    )
    assert not res.is_feasible
    assert res.score == 0.0
    assert res.method_class == "task_specific_head"


# 7. Generic regularizer transfer remains feasible
def test_generic_regularizer_remains_feasible() -> None:
    res = assess_transfer_feasibility(
        method="Spectral Regularization",
        source_dimension="task",
        source_context="segmentation",
        target_dimension="task",
        target_context="reconstruction",
    )
    assert res.is_feasible
    assert res.score == 0.8
    assert res.method_class == "generic_regularizer"


# 8. Unknown method class lowers confidence
def test_unknown_method_class_lowers_confidence() -> None:
    assert infer_method_class("Custom Mysterious Algorithm") == "unknown"
    res = assess_transfer_feasibility(
        method="Custom Mysterious Algorithm",
        source_dimension="task",
        source_context="segmentation",
        target_dimension="task",
        target_context="reconstruction",
    )
    assert res.is_feasible
    assert res.score == 0.5


# 9. EvidenceRefs never contain fabricated summaries & missing supporting_text is None
def test_evidencerefs_never_contain_fabricated_provenance_summaries() -> None:
    now = _utc_now()
    p1, c1 = _make_stored_card(
        "p1", "P1", ["Spectral Regularization"], task="segmentation", now=now
    )
    p2, c2 = _make_stored_card(
        "p2", "P2", ["Spectral Regularization"], task="segmentation", now=now
    )
    p3, c3 = _make_stored_card("p3", "P3", ["U-Net"], task="reconstruction", now=now)
    p4, c4 = _make_stored_card("p4", "P4", ["U-Net"], task="reconstruction", now=now)

    corpus = ScopedCorpusResult(
        cards=(c1, c2, c3, c4),
        papers=(p1, p2, p3, p4),
        corpus_paper_ids=("p1", "p2", "p3", "p4"),
        missing_cards_paper_ids=(),
        total_matching_papers=4,
    )

    miner = MethodTransferGapMiner()
    cands = miner.mine_transfer_gaps("Medical Imaging", corpus)
    cand = cands[0]

    for ref in cand.provenance.supporting_evidence:
        if ref.claim_or_field == "methods":
            assert ref.supporting_text is None
        assert "demonstrated in source context" not in (ref.supporting_text or "")
        assert "represented in" not in (ref.supporting_text or "")


# 10. Critic downgrades fresh target paper
@pytest.mark.asyncio
async def test_critic_downgrades_fresh_target_paper_for_method_transfer() -> None:
    now = _utc_now()
    p1, c1 = _make_stored_card(
        "p1", "P1", ["Spectral Regularization"], task="segmentation", now=now
    )
    p2, c2 = _make_stored_card(
        "p2", "P2", ["Spectral Regularization"], task="segmentation", now=now
    )
    p3, c3 = _make_stored_card("p3", "P3", ["U-Net"], task="reconstruction", now=now)
    p4, c4 = _make_stored_card("p4", "P4", ["U-Net"], task="reconstruction", now=now)

    corpus = ScopedCorpusResult(
        cards=(c1, c2, c3, c4),
        papers=(p1, p2, p3, p4),
        corpus_paper_ids=("p1", "p2", "p3", "p4"),
        missing_cards_paper_ids=(),
        total_matching_papers=4,
    )

    miner = MethodTransferGapMiner()
    cands = miner.mine_transfer_gaps("Medical Imaging", corpus)
    cand = cands[0]

    fresh_paper = Paper(
        id="p-fresh",
        title="Spectral Regularization for Reconstruction",
        abstract="We apply spectral regularization to reconstruction.",
        source="arxiv",
    )
    scout = FakeScout(return_papers=[fresh_paper])
    critic = CriticService(scout)

    review, updated_cand = await critic.review_candidate(cand, memory_cards=corpus.cards)

    assert review.decision == "downgraded"
    assert updated_cand.review_status == "downgraded"


# 11. Discord delegates type=method_transfer
@pytest.mark.asyncio
async def test_gap_service_supports_method_transfer_type(tmp_path_factory: object) -> None:
    db_file = tmp_path_factory.mktemp("db") / "test_method_transfer.db"  # type: ignore[attr-defined]
    db = Database.create(f"sqlite:///{db_file}")
    db.initialize_schema()
    repo = ResearchRepository(db)

    p1 = Paper(id="p1", title="P1", source="arxiv")
    p2 = Paper(id="p2", title="P2", source="arxiv")
    p3 = Paper(id="p3", title="P3", source="arxiv")
    p4 = Paper(id="p4", title="P4", source="arxiv")

    id1 = repo.upsert_merged_paper(p1)
    id2 = repo.upsert_merged_paper(p2)
    id3 = repo.upsert_merged_paper(p3)
    id4 = repo.upsert_merged_paper(p4)

    card1 = PaperCard(
        paper_id=id1,
        problem="Medical Imaging",
        methods=["Spectral Regularization"],
        tasks=[StructuredEvidence(value="segmentation", status="observed")],
    )
    card2 = PaperCard(
        paper_id=id2,
        problem="Medical Imaging",
        methods=["Spectral Regularization"],
        tasks=[StructuredEvidence(value="segmentation", status="observed")],
    )
    card3 = PaperCard(
        paper_id=id3,
        problem="Medical Imaging",
        methods=["U-Net"],
        tasks=[StructuredEvidence(value="reconstruction", status="observed")],
    )
    card4 = PaperCard(
        paper_id=id4,
        problem="Medical Imaging",
        methods=["U-Net"],
        tasks=[StructuredEvidence(value="reconstruction", status="observed")],
    )

    repo.upsert_paper_card(card1)
    repo.upsert_paper_card(card2)
    repo.upsert_paper_card(card3)
    repo.upsert_paper_card(card4)

    fake_scout = FakeScout(return_papers=[])
    gap_service = GapService(repo, fake_scout)

    res = await gap_service.analyze_gaps("Medical Imaging", count=1, gap_type="method_transfer")

    assert not res.is_insufficient_evidence
    assert len(res.candidates) == 1
    cand = res.candidates[0]
    assert cand.gap_type == "method_transfer"

    embed = render_gap_embed(cand)
    assert embed.title is not None
    assert "METHOD_TRANSFER" in embed.title

    db.dispose()
