"""Tests for explicit gap miner grouping and candidate generation."""

from __future__ import annotations

from datetime import UTC, datetime

from research_radar.gap.miner import ExplicitGapMiner
from research_radar.models.paper_card import PaperCard
from research_radar.storage.repositories import ScopedCorpusResult, StoredPaper, StoredPaperCard


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def test_repeated_limitation_across_two_papers_creates_candidate() -> None:
    now = _utc_now()
    p1 = StoredPaper(
        id="p1",
        canonical_key="key1",
        title="Paper A on MRI",
        abstract="Abstract A",
        authors=["Author A"],
        publication_year=2023,
        venue=None,
        doi=None,
        url="http://example.com/p1",
        citation_count=10,
        primary_source="arxiv",
        sources=(),
        first_discovered_at=now,
        created_at=now,
        updated_at=now,
    )
    p2 = StoredPaper(
        id="p2",
        canonical_key="key2",
        title="Paper B on MRI",
        abstract="Abstract B",
        authors=["Author B"],
        publication_year=2024,
        venue=None,
        doi=None,
        url="http://example.com/p2",
        citation_count=5,
        primary_source="openalex",
        sources=(),
        first_discovered_at=now,
        created_at=now,
        updated_at=now,
    )

    card1 = StoredPaperCard(
        card=PaperCard(
            paper_id="p1",
            limitations=["Performance degrades severely under scanner domain shift."],
        ),
        source_url="http://example.com/p1.pdf",
        document_sha256="sha1",
        selected_sections=("limitations",),
        llm_provider="test",
        llm_model="test",
        created_at=now,
        updated_at=now,
    )
    card2 = StoredPaperCard(
        card=PaperCard(
            paper_id="p2",
            limitations=["Generalization across scanner domain shift remains limited."],
        ),
        source_url="http://example.com/p2.pdf",
        document_sha256="sha2",
        selected_sections=("limitations",),
        llm_provider="test",
        llm_model="test",
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

    miner = ExplicitGapMiner()
    candidates = miner.mine_gaps("MRI reconstruction", corpus)

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.gap_type == "explicit"
    assert set(cand.supporting_papers) == {"p1", "p2"}
    assert cand.evidence_count == 2
    assert cand.evidence_score is not None and cand.evidence_score >= 0.5


def test_single_paper_limitation_remains_weak_lead() -> None:
    now = _utc_now()
    p1 = StoredPaper(
        id="p1",
        canonical_key="key1",
        title="Single Paper",
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
    card1 = StoredPaperCard(
        card=PaperCard(paper_id="p1", limitations=["Unique isolated limitation."]),
        source_url=None,
        document_sha256=None,
        selected_sections=(),
        llm_provider=None,
        llm_model=None,
        created_at=now,
        updated_at=now,
    )
    corpus = ScopedCorpusResult(
        cards=(card1,),
        papers=(p1,),
        corpus_paper_ids=("p1",),
        missing_cards_paper_ids=(),
        total_matching_papers=1,
    )

    miner = ExplicitGapMiner()
    candidates = miner.mine_gaps("Topic", corpus)
    # 1 paper is weak lead only, filtered out in V2A candidate generation
    assert len(candidates) == 0
