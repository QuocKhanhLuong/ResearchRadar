"""Tests for scoped corpus selection."""

from __future__ import annotations

from research_radar.gap.corpus import ScopedCorpusService
from research_radar.models.paper import Paper
from research_radar.models.paper_card import PaperCard
from research_radar.storage.database import Database
from research_radar.storage.repositories import ResearchRepository


def test_scoped_corpus_selection_by_topic(tmp_path_factory: object) -> None:
    db_file = tmp_path_factory.mktemp("db") / "test.db"  # type: ignore[attr-defined]
    db = Database.create(f"sqlite:///{db_file}")
    db.initialize_schema()
    repo = ResearchRepository(db)

    p1 = Paper(id="p1", title="Spectral MRI reconstruction method", source="arxiv")
    p2 = Paper(id="p2", title="Deep Learning for CT Reconstruction", source="openalex")
    p3 = Paper(id="p3", title="Unrelated NLP Transformer Models", source="arxiv")

    id1 = repo.upsert_merged_paper(p1)
    id2 = repo.upsert_merged_paper(p2)
    repo.upsert_merged_paper(p3)

    card1 = PaperCard(
        paper_id=id1,
        problem="3D MRI reconstruction",
        limitations=["Scanner domain shift"],
    )
    card2 = PaperCard(paper_id=id2, problem="CT Reconstruction", limitations=["Protocol variation"])

    repo.upsert_paper_card(card1)
    repo.upsert_paper_card(card2)

    service = ScopedCorpusService(repo)
    result = service.select_corpus("Spectral MRI", limit=10)

    assert len(result.cards) == 1
    assert result.cards[0].card.paper_id == id1
    assert result.total_matching_papers == 1
    assert id1 in result.corpus_paper_ids
    db.dispose()


def test_scoped_corpus_handles_empty_or_missing_cards(tmp_path_factory: object) -> None:
    db_file = tmp_path_factory.mktemp("db") / "test.db"  # type: ignore[attr-defined]
    db = Database.create(f"sqlite:///{db_file}")
    db.initialize_schema()
    repo = ResearchRepository(db)

    p1 = Paper(id="p1", title="Diffusion for Quantum Physics", source="arxiv")
    id1 = repo.upsert_merged_paper(p1)
    # p1 has no paper card

    service = ScopedCorpusService(repo)
    result = service.select_corpus("Quantum Physics", limit=10)

    assert len(result.cards) == 0
    assert len(result.papers) == 1
    assert result.missing_cards_paper_ids == (id1,)
    assert result.total_matching_papers == 1
    db.dispose()
