from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from research_radar.models import EvidenceClaim, Paper, PaperCard
from research_radar.storage import Database, ResearchRepository


@pytest.fixture
def repository(tmp_path: Path) -> ResearchRepository:
    database_path = tmp_path / "nested" / "research_radar.db"
    database = Database.create(f"sqlite:///{database_path}")
    database.initialize_schema()
    try:
        yield ResearchRepository(database)
    finally:
        database.dispose()


def _paper(
    *,
    identifier: str = "W123",
    source: str = "openalex",
    title: str = "Reliable Visual Anomaly Detection",
    doi: str | None = "10.1000/example",
    abstract: str | None = "A short abstract.",
    authors: list[str] | None = None,
    citation_count: int | None = 4,
    external_ids: dict[str, str] | None = None,
) -> Paper:
    return Paper(
        id=identifier,
        title=title,
        source=source,
        doi=doi,
        abstract=abstract,
        authors=authors or ["Ada Lovelace"],
        publication_year=2026,
        venue="Research Journal",
        citation_count=citation_count,
        external_ids=external_ids or {source: identifier},
    )


def test_paper_upsert_merges_provider_provenance_and_useful_metadata(
    repository: ResearchRepository,
) -> None:
    first_id = repository.upsert_merged_paper(_paper())
    second_id = repository.upsert_merged_paper(
        _paper(
            identifier="649218",
            source="semantic_scholar",
            title="Reliable Visual Anomaly Detection for Practical Deployment",
            doi="https://doi.org/10.1000/EXAMPLE",
            abstract=(
                "A much longer abstract that preserves the richer metadata from another provider."
            ),
            authors=["Ada Lovelace", "Grace Hopper"],
            citation_count=12,
            external_ids={"semantic_scholar": "649218"},
        )
    )

    stored = repository.get_paper(first_id)

    assert second_id == first_id
    assert stored is not None
    assert stored.doi == "10.1000/example"
    assert stored.citation_count == 12
    assert stored.authors == ["Ada Lovelace", "Grace Hopper"]
    assert stored.abstract is not None
    assert "much longer" in stored.abstract
    assert stored.external_ids == {
        "doi": "10.1000/example",
        "openalex": "W123",
        "semantic_scholar": "649218",
    }


def test_namespaced_provider_ids_do_not_duplicate_source_provenance(
    repository: ResearchRepository,
) -> None:
    paper_id = repository.upsert_merged_paper(
        _paper(
            identifier="openalex:W456",
            doi=None,
            external_ids={"openalex": "W456"},
        )
    )
    same_paper_id = repository.upsert_merged_paper(
        _paper(
            identifier="W456",
            doi=None,
            external_ids={"open_alex": "W456"},
        )
    )
    stored = repository.get_paper(paper_id)

    assert same_paper_id == paper_id
    assert stored is not None
    assert stored.external_ids == {"openalex": "W456"}


def test_doi_external_id_is_persisted_when_the_dedicated_field_is_missing(
    repository: ResearchRepository,
) -> None:
    paper_id = repository.upsert_merged_paper(
        _paper(
            doi=None,
            external_ids={"openalex": "W789", "doi": "https://doi.org/10.1000/External"},
        )
    )
    stored = repository.get_paper(paper_id)

    assert stored is not None
    assert stored.doi == "10.1000/external"


def test_arxiv_identity_bridges_versions_when_a_doi_arrives_later(
    repository: ResearchRepository,
) -> None:
    first_id = repository.upsert_merged_paper(
        _paper(
            identifier="arxiv:2401.01234v1",
            source="arxiv",
            title="An earlier title",
            doi=None,
            external_ids={"arxiv": "2401.01234v1"},
        )
    )
    second_id = repository.upsert_merged_paper(
        _paper(
            identifier="arxiv:2401.01234v2",
            source="arxiv",
            title="A revised title",
            doi="10.1000/arxiv-bridge",
            external_ids={"arxiv": "2401.01234v2"},
        )
    )
    stored = repository.get_paper(first_id)

    assert second_id == first_id
    assert stored is not None
    assert stored.doi == "10.1000/arxiv-bridge"


def test_reconciliation_moves_watch_history_and_card_to_stable_paper(
    repository: ResearchRepository,
) -> None:
    first_id = repository.upsert_merged_paper(
        _paper(doi=None, title="First provider title", abstract="short")
    )
    second_id = repository.upsert_merged_paper(
        _paper(
            identifier="S2",
            source="semantic_scholar",
            doi=None,
            title="Second provider title",
            abstract="A richer abstract from a different provider.",
            external_ids={"semantic_scholar": "S2"},
        )
    )
    topic = repository.add_watch_topic("Visual anomaly", "visual anomaly detection")
    repository.record_watch_discovery(topic.id, second_id, 0.75)
    repository.upsert_paper_card(PaperCard(paper_id=second_id, problem="Detect anomalies"))

    merged_id = repository.upsert_merged_paper(
        _paper(
            doi="10.1000/unified",
            title="Unified visual anomaly detection title",
            external_ids={"openalex": "W123", "semantic_scholar": "S2"},
        )
    )

    stored = repository.get_paper(first_id)
    pending = repository.list_pending_notifications(topic.id)

    assert merged_id == first_id
    assert repository.get_paper(second_id) is None
    assert stored is not None
    assert stored.external_ids["semantic_scholar"] == "S2"
    assert repository.get_paper_card(first_id) == PaperCard(
        paper_id=first_id,
        problem="Detect anomalies",
    )
    assert [item.paper.id for item in pending] == [first_id]


def test_short_titles_do_not_merge_without_a_stronger_identity(
    repository: ResearchRepository,
) -> None:
    first_id = repository.upsert_merged_paper(
        _paper(identifier="W-short", title="AI", doi=None, external_ids={"openalex": "W-short"})
    )
    second_id = repository.upsert_merged_paper(
        _paper(
            identifier="S-short",
            source="semantic_scholar",
            title="AI",
            doi=None,
            external_ids={"semantic_scholar": "S-short"},
        )
    )

    assert second_id != first_id


def test_paper_card_round_trip_retains_compact_analysis_provenance(
    repository: ResearchRepository,
) -> None:
    paper_id = repository.upsert_merged_paper(_paper())
    card = PaperCard(
        paper_id=paper_id,
        problem="Robust anomaly detection",
        contributions=["A deterministic score"],
        main_claims=[
            EvidenceClaim(
                claim="The score improves detection robustness.",
                source_section="Results",
                supporting_text="Our score improves robustness.",
            )
        ],
        limitations=["Only one evaluation condition"],
    )

    stored = repository.upsert_paper_card(
        card,
        source_url="https://example.test/paper.pdf",
        document_sha256="a" * 64,
        selected_sections={"Abstract": "excerpt", "Results": "excerpt"},
        llm_provider="remote",
        llm_model="model-name",
    )
    record = repository.get_paper_card_record(paper_id)

    assert stored == card
    assert repository.get_paper_card(paper_id) == card
    assert record is not None
    assert record.source_url == "https://example.test/paper.pdf"
    assert record.selected_sections == ("Abstract", "Results")
    assert record.llm_provider == "remote"
    assert record.card.main_claims[0].supporting_text == "Our score improves robustness."

    with pytest.raises(ValueError, match="unknown paper id"):
        repository.upsert_paper_card(PaperCard(paper_id="missing", problem="No paper"))


def test_watch_topics_discovery_notification_and_scan_state(repository: ResearchRepository) -> None:
    paper_id = repository.upsert_merged_paper(_paper())
    topic = repository.add_watch_topic("Medical reconstruction", "3D medical reconstruction")

    assert repository.record_watch_discovery(topic.id, paper_id, 0.8) is True
    assert repository.record_watch_discovery(topic.id, paper_id, 0.2) is False
    assert (
        repository.list_pending_notifications(topic.id, minimum_rank_score=0.7)[0].rank_score == 0.8
    )
    assert repository.mark_notified(topic.id, paper_id) == 1
    assert repository.list_pending_notifications(topic.id) == []
    assert repository.mark_watch_scan_failure(topic.id, "temporary provider outage") is True
    failed_topic = repository.get_watch_topic(topic.id)
    assert failed_topic is not None
    assert failed_topic.last_error == "temporary provider outage"
    assert repository.mark_watch_scan_success(topic.id) is True
    refreshed = repository.get_watch_topic(topic.id)
    assert refreshed is not None
    assert refreshed.last_scan_at is not None
    assert refreshed.last_error is None
    disabled_topic = repository.set_watch_topic_enabled(topic.id, False)
    assert disabled_topic is not None
    assert disabled_topic.enabled is False
    assert repository.list_enabled_watch_topics() == []

    with pytest.raises(ValueError, match="already exists"):
        repository.add_watch_topic(" medical reconstruction ", "another query")

    assert repository.remove_watch_topic("medical reconstruction") is True
    assert repository.list_watch_topics() == []


def test_local_search_and_digest_state_use_only_persisted_memory(
    repository: ResearchRepository,
) -> None:
    paper_id = repository.upsert_merged_paper(
        _paper(
            title="Spectral State Space Models for Imaging",
            abstract="A state space model improves medical image reconstruction.",
            doi="10.1000/state-space",
        )
    )
    topic = repository.add_watch_topic("State space", "spectral state space models")
    repository.record_watch_discovery(topic.id, paper_id, 0.91)
    repository.upsert_paper_card(
        PaperCard(paper_id=paper_id, contributions=["Efficient sequence model"])
    )
    stored = repository.get_paper(paper_id)
    assert stored is not None

    matches = repository.get_papers_for_local_lexical_search("state imaging")
    card_matches = repository.get_papers_for_local_lexical_search("efficient sequence")
    start = stored.first_discovered_at - timedelta(seconds=1)
    end = stored.first_discovered_at + timedelta(seconds=1)
    candidates = repository.list_digest_candidates(start, end)
    run = repository.claim_digest_run(start, end)

    assert [match.id for match in matches] == [paper_id]
    assert [match.id for match in card_matches] == [paper_id]
    assert len(candidates) == 1
    assert candidates[0].watch_topic_names == ("State space",)
    assert candidates[0].highest_rank_score == 0.91
    assert candidates[0].paper_card is not None
    assert run is not None
    assert repository.claim_digest_run(start, end) is None
    assert repository.mark_digest_failed(run.id, "Discord temporarily unavailable") is True
    retry = repository.claim_digest_run(start, end)
    assert retry is not None
    assert repository.mark_digest_sent(retry.id, paper_count=1) is True
    assert repository.get_last_successful_digest_end() == end

    recorded = repository.record_digest_run(start, end, status="sent", paper_count=1)
    assert recorded.status == "sent"
    assert recorded.paper_count == 1


def test_digest_includes_a_preexisting_paper_newly_seen_by_a_watch_topic(
    repository: ResearchRepository,
) -> None:
    paper_id = repository.upsert_merged_paper(_paper(doi="10.1000/old-paper"))
    stored = repository.get_paper(paper_id)
    assert stored is not None
    topic = repository.add_watch_topic("Anomaly", "visual anomaly")
    discovery_time = stored.first_discovered_at + timedelta(days=2)
    repository.record_watch_discovery(topic.id, paper_id, 0.6, seen_at=discovery_time)

    candidates = repository.list_digest_candidates(
        discovery_time - timedelta(seconds=1),
        discovery_time + timedelta(seconds=1),
    )

    assert [candidate.paper.id for candidate in candidates] == [paper_id]
    assert candidates[0].watch_topic_names == ("Anomaly",)
