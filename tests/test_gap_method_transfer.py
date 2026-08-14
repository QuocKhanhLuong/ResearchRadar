"""Comprehensive unit tests for V2E Method-Transfer Gap Engine."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from research_radar.bot.commands.gap import render_gap_embed
from research_radar.gap.critic import CriticService
from research_radar.gap.method_transfer import (
    MethodTransferGapMiner,
    assess_transfer_feasibility,
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
            methods=methods,
            tasks=[StructuredEvidence(value=task, status="observed")],
            modalities=[StructuredEvidence(value=modality, status="observed")],
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


# 1. Method observed in source + target context exists + combination absent -> candidate
def test_method_transfer_generates_candidate() -> None:
    now = _utc_now()
    # Source context: Spectral Regularization used in Segmentation (p1, p2)
    p1, c1 = _make_stored_card(
        "p1", "P1", ["Spectral Regularization"], task="segmentation", now=now
    )
    p2, c2 = _make_stored_card(
        "p2", "P2", ["Spectral Regularization"], task="segmentation", now=now
    )

    # Target context: Reconstruction task exists in corpus (p3, p4), but uses U-Net
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
    assert "reconstruction" in cand.title.lower()
    assert "transfer hypothesis" in " ".join(cand.caveats).lower()


# 2. Method already used in target context -> no candidate
def test_method_transfer_already_used_in_target_returns_no_candidate() -> None:
    now = _utc_now()
    # Spectral Regularization used in both segmentation (p1, p2) and reconstruction (p3, p4)
    p1, c1 = _make_stored_card(
        "p1", "P1", ["Spectral Regularization"], task="segmentation", now=now
    )
    p2, c2 = _make_stored_card(
        "p2", "P2", ["Spectral Regularization"], task="segmentation", now=now
    )
    p3, c3 = _make_stored_card(
        "p3", "P3", ["Spectral Regularization"], task="reconstruction", now=now
    )
    p4, c4 = _make_stored_card(
        "p4", "P4", ["Spectral Regularization"], task="reconstruction", now=now
    )

    corpus = ScopedCorpusResult(
        cards=(c1, c2, c3, c4),
        papers=(p1, p2, p3, p4),
        corpus_paper_ids=("p1", "p2", "p3", "p4"),
        missing_cards_paper_ids=(),
        total_matching_papers=4,
    )

    miner = MethodTransferGapMiner()
    cands = miner.mine_transfer_gaps("Medical Imaging", corpus)

    # Method is already used in reconstruction => no transfer candidate for reconstruction!
    recon_cands = [c for c in cands if "reconstruction" in c.title.lower()]
    assert len(recon_cands) == 0


# 3. Weak source support (< 2 papers) -> no candidate
def test_method_transfer_weak_source_support_returns_no_candidate() -> None:
    now = _utc_now()
    # Each method has only 1 paper (weak source support < 2 papers for all methods)
    p1, c1 = _make_stored_card(
        "p1", "P1", ["Spectral Regularization"], task="segmentation", now=now
    )
    p2, c2 = _make_stored_card("p2", "P2", ["Other Method"], task="segmentation", now=now)
    p3, c3 = _make_stored_card("p3", "P3", ["U-Net"], task="reconstruction", now=now)
    p4, c4 = _make_stored_card("p4", "P4", ["VNet"], task="reconstruction", now=now)

    corpus = ScopedCorpusResult(
        cards=(c1, c2, c3, c4),
        papers=(p1, p2, p3, p4),
        corpus_paper_ids=("p1", "p2", "p3", "p4"),
        missing_cards_paper_ids=(),
        total_matching_papers=4,
    )

    miner = MethodTransferGapMiner()
    cands = miner.mine_transfer_gaps("Medical Imaging", corpus)
    assert len(cands) == 0


# 4. Target context represented in < 2 papers -> no candidate
def test_method_transfer_target_context_insufficient_returns_no_candidate() -> None:
    now = _utc_now()
    p1, c1 = _make_stored_card(
        "p1", "P1", ["Spectral Regularization"], task="segmentation", now=now
    )
    p2, c2 = _make_stored_card(
        "p2", "P2", ["Spectral Regularization"], task="segmentation", now=now
    )
    # Only 1 paper in reconstruction task
    p3, c3 = _make_stored_card("p3", "P3", ["U-Net"], task="reconstruction", now=now)

    corpus = ScopedCorpusResult(
        cards=(c1, c2, c3),
        papers=(p1, p2, p3),
        corpus_paper_ids=("p1", "p2", "p3"),
        missing_cards_paper_ids=(),
        total_matching_papers=3,
    )

    miner = MethodTransferGapMiner()
    cands = miner.mine_transfer_gaps("Medical Imaging", corpus)
    assert len(cands) == 0


# 5. Incompatible transfer (e.g. bounding-box detector head -> MRI reconstruction) -> no candidate
def test_method_transfer_incompatible_transfer_is_skipped() -> None:
    res = assess_transfer_feasibility(
        method="YOLOv8 bounding-box detector head",
        source_context="object detection",
        target_context="MRI reconstruction",
    )
    assert not res.is_feasible
    assert res.score == 0.0


# 6. Feasible transfer has valid score
def test_method_transfer_feasible_transfer_has_valid_score() -> None:
    res = assess_transfer_feasibility(
        method="Spectral Regularization",
        source_context="image segmentation",
        target_context="MRI reconstruction",
    )
    assert res.is_feasible
    assert res.score > 0.5


# 7. Deterministic candidate ID
def test_method_transfer_candidate_id_is_deterministic() -> None:
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
    cands1 = miner.mine_transfer_gaps("Medical Imaging", corpus)
    cands2 = miner.mine_transfer_gaps("Medical Imaging", corpus)

    assert cands1[0].id == cands2[0].id


# 8. Source/target EvidenceRefs preserved
def test_method_transfer_evidencerefs_preserve_source_and_target() -> None:
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

    assert len(cand.provenance.supporting_evidence) >= 2
    claims = [ref.claim_or_field for ref in cand.provenance.supporting_evidence]
    assert "methods" in claims
    assert "tasks" in claims


# 9. Fresh direct target paper -> Critic downgrade
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
        title="Spectral Regularization for MRI Reconstruction",
        abstract="We apply spectral regularization to MRI reconstruction.",
        source="arxiv",
    )
    scout = FakeScout(return_papers=[fresh_paper])
    critic = CriticService(scout)

    review, updated_cand = await critic.review_candidate(cand, memory_cards=corpus.cards)

    assert review.decision == "downgraded"
    assert updated_cand.review_status == "downgraded"


# 10. Structured target evidence -> Critic reject
@pytest.mark.asyncio
async def test_critic_rejects_when_structured_evidence_resolves_method_transfer() -> None:
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

    # Memory card p5 explicitly resolves the transfer hypothesis
    _, c5 = _make_stored_card(
        "p5", "P5", ["Spectral Regularization"], task="reconstruction", now=now
    )
    all_cards = (c1, c2, c3, c4, c5)

    scout = FakeScout(return_papers=[])
    critic = CriticService(scout)

    review, updated_cand = await critic.review_candidate(cand, memory_cards=all_cards)

    assert review.decision == "rejected"
    assert updated_cand.review_status == "rejected"


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
