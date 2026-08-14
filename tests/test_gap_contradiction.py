"""Comprehensive unit tests for V2D Contradiction Gap Engine."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from research_radar.bot.commands.gap import render_gap_embed
from research_radar.gap.contradiction import (
    ContradictionGapMiner,
    calculate_context_compatibility,
)
from research_radar.gap.critic import CriticService
from research_radar.gap.service import GapService
from research_radar.models.paper import Paper
from research_radar.models.paper_card import EvidenceClaim, PaperCard, StructuredEvidence
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
    claims: list[str],
    task: str = "segmentation",
    modality: str = "mri",
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

    card = StoredPaperCard(
        card=PaperCard(
            paper_id=paper_id,
            problem=f"{task} problem",
            tasks=[StructuredEvidence(value=task, status="observed")],
            modalities=[StructuredEvidence(value=modality, status="observed")],
            main_claims=[EvidenceClaim(claim=c) for c in claims],
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


# 1. Opposite claims + same context -> candidate
def test_contradiction_opposite_claims_same_context_generates_candidate() -> None:
    now = _utc_now()
    p1, c1 = _make_stored_card(
        "p1",
        "Paper A",
        ["Spectral regularization improves robustness under scanner shift."],
        task="segmentation",
        modality="mri",
        now=now,
    )
    p2, c2 = _make_stored_card(
        "p2",
        "Paper B",
        ["Spectral regularization degrades performance under scanner shift."],
        task="segmentation",
        modality="mri",
        now=now,
    )

    corpus = ScopedCorpusResult(
        cards=(c1, c2),
        papers=(p1, p2),
        corpus_paper_ids=("p1", "p2"),
        missing_cards_paper_ids=(),
        total_matching_papers=2,
    )

    miner = ContradictionGapMiner()
    cands = miner.mine_contradiction_gaps("Medical Imaging", corpus)

    assert len(cands) == 1
    cand = cands[0]
    assert cand.gap_type == "contradiction"
    assert "p1" in cand.supporting_papers or "p1" in cand.conflicting_papers
    assert "p2" in cand.supporting_papers or "p2" in cand.conflicting_papers
    assert "inconsistent" in " ".join(cand.caveats).lower()


# 2. Opposite claims + different task -> no candidate
def test_contradiction_different_task_blocks_candidate() -> None:
    now = _utc_now()
    p1, c1 = _make_stored_card(
        "p1",
        "Paper A",
        ["Spectral regularization improves robustness under scanner shift."],
        task="segmentation",
        modality="mri",
        now=now,
    )
    p2, c2 = _make_stored_card(
        "p2",
        "Paper B",
        ["Spectral regularization degrades performance under scanner shift."],
        task="reconstruction",
        modality="mri",
        now=now,
    )

    corpus = ScopedCorpusResult(
        cards=(c1, c2),
        papers=(p1, p2),
        corpus_paper_ids=("p1", "p2"),
        missing_cards_paper_ids=(),
        total_matching_papers=2,
    )

    miner = ContradictionGapMiner()
    cands = miner.mine_contradiction_gaps("Medical Imaging", corpus)
    assert len(cands) == 0


# 3. Opposite claims + different modality -> no candidate
def test_contradiction_different_modality_blocks_candidate() -> None:
    now = _utc_now()
    p1, c1 = _make_stored_card(
        "p1",
        "Paper A",
        ["Spectral regularization improves robustness under scanner shift."],
        task="segmentation",
        modality="mri",
        now=now,
    )
    p2, c2 = _make_stored_card(
        "p2",
        "Paper B",
        ["Spectral regularization degrades performance under scanner shift."],
        task="segmentation",
        modality="ct",
        now=now,
    )

    corpus = ScopedCorpusResult(
        cards=(c1, c2),
        papers=(p1, p2),
        corpus_paper_ids=("p1", "p2"),
        missing_cards_paper_ids=(),
        total_matching_papers=2,
    )

    miner = ContradictionGapMiner()
    cands = miner.mine_contradiction_gaps("Medical Imaging", corpus)
    assert len(cands) == 0


# 4. Same claim direction -> no contradiction
def test_contradiction_same_direction_returns_no_candidate() -> None:
    now = _utc_now()
    p1, c1 = _make_stored_card(
        "p1",
        "Paper A",
        ["Spectral regularization improves robustness under scanner shift."],
        now=now,
    )
    p2, c2 = _make_stored_card(
        "p2",
        "Paper B",
        ["Spectral regularization enhances robustness under scanner shift."],
        now=now,
    )

    corpus = ScopedCorpusResult(
        cards=(c1, c2),
        papers=(p1, p2),
        corpus_paper_ids=("p1", "p2"),
        missing_cards_paper_ids=(),
        total_matching_papers=2,
    )

    miner = ContradictionGapMiner()
    cands = miner.mine_contradiction_gaps("Medical Imaging", corpus)
    assert len(cands) == 0


# 5. Unknown / neutral polarity -> no contradiction
def test_contradiction_neutral_polarity_returns_no_candidate() -> None:
    now = _utc_now()
    p1, c1 = _make_stored_card(
        "p1",
        "Paper A",
        ["We analyze spectral regularization under scanner shift."],
        now=now,
    )
    p2, c2 = _make_stored_card(
        "p2",
        "Paper B",
        ["Spectral regularization degrades performance under scanner shift."],
        now=now,
    )

    corpus = ScopedCorpusResult(
        cards=(c1, c2),
        papers=(p1, p2),
        corpus_paper_ids=("p1", "p2"),
        missing_cards_paper_ids=(),
        total_matching_papers=2,
    )

    miner = ContradictionGapMiner()
    cands = miner.mine_contradiction_gaps("Medical Imaging", corpus)
    assert len(cands) == 0


# 6. Different metrics -> no direct contradiction
def test_contradiction_different_metrics_reduces_compatibility() -> None:
    card1 = PaperCard(
        paper_id="p1",
        tasks=[StructuredEvidence(value="segmentation", status="observed")],
        modalities=[StructuredEvidence(value="mri", status="observed")],
    )
    card2 = PaperCard(
        paper_id="p2",
        tasks=[StructuredEvidence(value="reconstruction", status="observed")],
        modalities=[StructuredEvidence(value="mri", status="observed")],
    )

    score = calculate_context_compatibility(card1, card2)
    assert score == 0.0  # Mismatched tasks return 0.0


# 7. Missing context fields reduce confidence
def test_contradiction_missing_context_reduces_confidence() -> None:
    card1 = PaperCard(
        paper_id="p1",
        tasks=[StructuredEvidence(value="segmentation", status="observed")],
    )
    card2 = PaperCard(
        paper_id="p2",
        tasks=[StructuredEvidence(value="segmentation", status="observed")],
    )

    score = calculate_context_compatibility(card1, card2)
    assert 0.0 < score < 1.0  # Missing modalities/datasets penalize compatibility score


# 8. EvidenceRefs preserve both source claims
def test_contradiction_evidencerefs_preserve_both_source_claims() -> None:
    now = _utc_now()
    p1, c1 = _make_stored_card(
        "p1",
        "Paper A",
        ["Spectral regularization improves robustness under scanner shift."],
        now=now,
    )
    p2, c2 = _make_stored_card(
        "p2",
        "Paper B",
        ["Spectral regularization degrades performance under scanner shift."],
        now=now,
    )

    corpus = ScopedCorpusResult(
        cards=(c1, c2),
        papers=(p1, p2),
        corpus_paper_ids=("p1", "p2"),
        missing_cards_paper_ids=(),
        total_matching_papers=2,
    )

    miner = ContradictionGapMiner()
    cands = miner.mine_contradiction_gaps("Medical Imaging", corpus)
    assert len(cands) == 1
    cand = cands[0]

    assert len(cand.provenance.supporting_evidence) == 1
    assert len(cand.provenance.conflicting_evidence) == 1

    supp_ref = cand.provenance.supporting_evidence[0]
    conf_ref = cand.provenance.conflicting_evidence[0]

    assert "improves" in supp_ref.supporting_text
    assert "degrades" in conf_ref.supporting_text


# 9. Deterministic candidate ID
def test_contradiction_candidate_id_is_deterministic() -> None:
    now = _utc_now()
    p1, c1 = _make_stored_card(
        "p1",
        "Paper A",
        ["Spectral regularization improves robustness under scanner shift."],
        now=now,
    )
    p2, c2 = _make_stored_card(
        "p2",
        "Paper B",
        ["Spectral regularization degrades performance under scanner shift."],
        now=now,
    )

    corpus = ScopedCorpusResult(
        cards=(c1, c2),
        papers=(p1, p2),
        corpus_paper_ids=("p1", "p2"),
        missing_cards_paper_ids=(),
        total_matching_papers=2,
    )

    miner = ContradictionGapMiner()
    cands1 = miner.mine_contradiction_gaps("Medical Imaging", corpus)
    cands2 = miner.mine_contradiction_gaps("Medical Imaging", corpus)

    assert cands1[0].id == cands2[0].id


# 10. Critic can downgrade context-mismatch or fresh overlap cases
@pytest.mark.asyncio
async def test_critic_downgrades_fresh_overlap_for_contradiction() -> None:
    now = _utc_now()
    p1, c1 = _make_stored_card(
        "p1",
        "Paper A",
        ["Spectral regularization improves robustness under scanner shift."],
        now=now,
    )
    p2, c2 = _make_stored_card(
        "p2",
        "Paper B",
        ["Spectral regularization degrades performance under scanner shift."],
        now=now,
    )

    corpus = ScopedCorpusResult(
        cards=(c1, c2),
        papers=(p1, p2),
        corpus_paper_ids=("p1", "p2"),
        missing_cards_paper_ids=(),
        total_matching_papers=2,
    )

    miner = ContradictionGapMiner()
    cands = miner.mine_contradiction_gaps("Medical Imaging", corpus)
    cand = cands[0]

    fresh_paper = Paper(
        id="p-fresh",
        title="Analysis of Spectral Regularization Under Scanner Shift",
        abstract="We investigate spectral regularization robustness.",
        source="arxiv",
    )
    scout = FakeScout(return_papers=[fresh_paper])
    critic = CriticService(scout)

    review, updated_cand = await critic.review_candidate(cand, memory_cards=corpus.cards)

    assert review.decision == "downgraded"
    assert updated_cand.review_status == "downgraded"


# 11. Metadata-only overlap never rejects
@pytest.mark.asyncio
async def test_critic_metadata_overlap_never_rejects_contradiction() -> None:
    now = _utc_now()
    p1, c1 = _make_stored_card(
        "p1",
        "Paper A",
        ["Spectral regularization improves robustness under scanner shift."],
        now=now,
    )
    p2, c2 = _make_stored_card(
        "p2",
        "Paper B",
        ["Spectral regularization degrades performance under scanner shift."],
        now=now,
    )

    corpus = ScopedCorpusResult(
        cards=(c1, c2),
        papers=(p1, p2),
        corpus_paper_ids=("p1", "p2"),
        missing_cards_paper_ids=(),
        total_matching_papers=2,
    )

    miner = ContradictionGapMiner()
    cands = miner.mine_contradiction_gaps("Medical Imaging", corpus)
    cand = cands[0]

    fresh_paper = Paper(
        id="p-fresh",
        title="Spectral Regularization Scanner Shift Study",
        abstract="Keywords matching title metadata.",
        source="arxiv",
    )
    scout = FakeScout(return_papers=[fresh_paper])
    critic = CriticService(scout)

    review, updated_cand = await critic.review_candidate(cand, memory_cards=corpus.cards)

    assert review.decision != "rejected"


# 12. Discord type=contradiction delegates correctly
@pytest.mark.asyncio
async def test_gap_service_supports_contradiction_type(tmp_path_factory: object) -> None:
    db_file = tmp_path_factory.mktemp("db") / "test_contradiction.db"  # type: ignore[attr-defined]
    db = Database.create(f"sqlite:///{db_file}")
    db.initialize_schema()
    repo = ResearchRepository(db)

    p1 = Paper(id="p1", title="Paper A", source="arxiv")
    p2 = Paper(id="p2", title="Paper B", source="arxiv")

    id1 = repo.upsert_merged_paper(p1)
    id2 = repo.upsert_merged_paper(p2)

    card1 = PaperCard(
        paper_id=id1,
        problem="Medical Imaging",
        tasks=[StructuredEvidence(value="segmentation", status="observed")],
        modalities=[StructuredEvidence(value="mri", status="observed")],
        main_claims=[
            EvidenceClaim(
                claim="Spectral regularization improves robustness under scanner shift."
            )
        ],
    )
    card2 = PaperCard(
        paper_id=id2,
        problem="Medical Imaging",
        tasks=[StructuredEvidence(value="segmentation", status="observed")],
        modalities=[StructuredEvidence(value="mri", status="observed")],
        main_claims=[
            EvidenceClaim(
                claim="Spectral regularization degrades performance under scanner shift."
            )
        ],
    )

    repo.upsert_paper_card(card1)
    repo.upsert_paper_card(card2)

    fake_scout = FakeScout(return_papers=[])
    gap_service = GapService(repo, fake_scout)

    res = await gap_service.analyze_gaps(
        "Medical Imaging", count=1, gap_type="contradiction"
    )

    assert not res.is_insufficient_evidence
    assert len(res.candidates) == 1
    cand = res.candidates[0]
    assert cand.gap_type == "contradiction"

    # Test embed rendering
    embed = render_gap_embed(cand)
    assert embed.title is not None
    assert "CONTRADICTION" in embed.title

    db.dispose()
