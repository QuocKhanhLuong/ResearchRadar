"""Unit tests for canonicalization and the ingestion engine (no network, no LLM)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from research_radar.errors import ProviderUnavailableError
from research_radar.models import Paper
from research_radar.research.canonical import canonicalize, identity_summary
from research_radar.research.dedup import deduplicate, normalize_title
from research_radar.research.ingestion import IngestionService
from research_radar.research.scout import ScoutService
from research_radar.storage.database import Database, create_database, initialize_schema
from research_radar.storage.ingestion_repository import IngestionRepository
from research_radar.storage.repositories import ResearchRepository

DOI = "10.1000/radar"
StackBuilder = Callable[..., tuple[IngestionService, ResearchRepository, IngestionRepository]]


class FakeProvider:
    """Canned provider that records requested limits or raises a fixed error."""

    def __init__(
        self,
        name: str,
        papers: list[Paper] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.papers = list(papers or [])
        self.error = error
        self.received_limits: list[int] = []

    async def search(self, query: str, limit: int) -> list[Paper]:
        """Return canned papers unless constructed with an error."""

        self.received_limits.append(limit)
        if self.error is not None:
            raise self.error
        return self.papers


class FakeReader:
    """Records every PDF URL it is asked to read."""

    def __init__(self) -> None:
        self.urls: list[str] = []
        self.calls = 0

    async def read_url(self, url: str) -> object:
        """Record the URL and return a stand-in read outcome."""

        self.calls += 1
        self.urls.append(url)
        return SimpleNamespace(paper_id="fake-card")


@pytest.fixture
def database() -> Iterator[Database]:
    database = create_database("sqlite:///:memory:")
    initialize_schema(database)
    yield database
    database.dispose()


@pytest.fixture
def build_stack(database: Database) -> StackBuilder:
    """Create an ingestion service plus its repositories over one fresh database."""

    def build(
        providers: list[FakeProvider],
        *,
        reader: FakeReader | None = None,
        metadata_limit: int = 50,
    ) -> tuple[IngestionService, ResearchRepository, IngestionRepository]:
        repository = ResearchRepository(database)
        ingestion_repository = IngestionRepository(database)
        service = IngestionService(
            scout=ScoutService(providers),
            repository=repository,
            ingestion_repository=ingestion_repository,
            reader_service=reader,
            metadata_limit=metadata_limit,
        )
        return service, repository, ingestion_repository

    return build


def shared_doi_papers() -> list[Paper]:
    """Three provider records describing one work through differently shaped DOIs."""

    return [
        Paper(
            id="openalex:W123",
            title="Sparse Mixture Of Experts",
            abstract="openalex has the richest abstract",
            citation_count=5,
            url="https://example.test/w123",
            doi="https://doi.org/10.1000/RADAR",
            source="openalex",
            external_ids={"openalex": "W123"},
        ),
        Paper(
            id="semantic_scholar:s2-abc",
            title="sparse mixture of experts!",
            doi="10.1000/radar",
            source="semantic_scholar",
            external_ids={"semantic_scholar": "s2-abc"},
        ),
        Paper(
            id="arxiv:2101.00001",
            title="SPARSE   mixture of experts",
            doi="DOI:10.1000/radar",
            source="arxiv",
            external_ids={"arxiv": "2101.00001"},
        ),
    ]


def standalone_pdf_paper(number: int) -> Paper:
    """A distinct paper that also carries a direct-PDF url for auto-read selection."""

    return Paper(
        id=f"openalex:W{number}",
        title=f"Standalone Study Number {number}",
        doi=f"10.2000/{number}",
        citation_count=number,
        source="openalex",
        external_ids={"openalex": f"W{number}", "pdf_url": f"https://example.test/{number}.pdf"},
    )


def count_rows(database: Database, table: str) -> int:
    """Return the raw row count of one table for direct persistence assertions."""

    with database.engine.connect() as connection:
        return int(connection.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one())


def test_canonicalize_collapses_shared_normalized_doi_across_providers() -> None:
    canonical = canonicalize(shared_doi_papers())

    assert len(canonical) == 1
    entry = canonical[0]
    assert len(entry.contributing_papers) == 3
    assert entry.provider_ids == {
        "openalex": "W123",
        "semantic_scholar": "s2-abc",
        "arxiv": "2101.00001",
        "doi": "10.1000/radar",
    }
    assert entry.paper.source == "openalex"
    assert entry.paper.doi == "10.1000/radar"
    assert entry.paper.abstract == "openalex has the richest abstract"


def test_canonicalize_output_order_is_stable_for_a_given_input_order() -> None:
    left, right = standalone_pdf_paper(1), standalone_pdf_paper(2)

    first_pass = [entry.paper.id for entry in canonicalize([left, right])]
    second_pass = [entry.paper.id for entry in canonicalize([left, right])]

    assert first_pass == second_pass == ["openalex:W1", "openalex:W2"]


def test_canonicalize_grouping_matches_deduplicate_on_mixed_input() -> None:
    papers = [*shared_doi_papers(), standalone_pdf_paper(1), standalone_pdf_paper(2)]

    entries = canonicalize(papers)

    assert [entry.paper for entry in entries] == deduplicate(papers)


async def test_ingestion_persists_one_paper_with_three_provider_sources(
    build_stack: StackBuilder,
    database: Database,
) -> None:
    papers = shared_doi_papers()
    service, repository, ingestion_repository = build_stack(
        [
            FakeProvider("openalex", [papers[0]]),
            FakeProvider("semantic_scholar", [papers[1]]),
            FakeProvider("arxiv", [papers[2]]),
        ]
    )

    result = await service.ingest_research_topic(" sparse experts ")

    assert result.discovered_count == 3
    assert result.canonical_count == 1
    assert len(result.paper_ids) == 1
    assert count_rows(database, "papers") == 1
    sources = repository.list_paper_sources(result.paper_ids[0])
    provider_rows = [source for source in sources if source.provider != "doi"]
    assert sorted(source.provider for source in provider_rows) == [
        "arxiv",
        "openalex",
        "semantic_scholar",
    ]
    run_record = ingestion_repository.list_recent_runs()[0]
    assert run_record.status == "completed"
    assert (run_record.discovered_count, run_record.canonical_count) == (3, 1)


def test_canonicalize_keeps_distinct_dois_and_titles_apart() -> None:
    left = Paper(
        id="openalex:A",
        title="Causal Representation Learning",
        doi="10.3000/a",
        source="openalex",
        external_ids={"openalex": "A"},
    )
    right = Paper(
        id="semantic_scholar:B",
        title="Offline Reinforcement Learning At Scale",
        doi="10.3000/b",
        source="semantic_scholar",
        external_ids={"semantic_scholar": "B"},
    )

    canonical = canonicalize([left, right])

    assert len(canonical) == 2
    assert {entry.paper.doi for entry in canonical} == {"10.3000/a", "10.3000/b"}


def test_similar_titles_without_shared_identifiers_are_not_merged() -> None:
    left = Paper(
        id="openalex:C",
        title="Attention Is All You Need",
        source="openalex",
        external_ids={"openalex": "C"},
    )
    right = Paper(
        id="arxiv:D",
        title="Attention Is All You Needs Revisited",
        source="arxiv",
        external_ids={"arxiv": "D"},
    )
    assert normalize_title(left.title) != normalize_title(right.title)

    canonical = canonicalize([left, right])

    assert len(canonical) == 2


def test_identity_summary_reports_ordered_identity_keys() -> None:
    paper = Paper(
        id="arxiv:2101.00001",
        title="Sparse Mixture Of Experts",
        doi="10.1000/radar",
        source="arxiv",
        external_ids={"arxiv": "2101.00001"},
    )

    summary = identity_summary(paper)

    assert summary[0] == "doi:10.1000/radar"
    assert "arxiv:2101.00001" in summary
    assert summary[-1] == "title:sparse mixture of experts"


async def test_repeated_ingestion_does_not_duplicate_papers_but_records_two_runs(
    build_stack: StackBuilder,
    database: Database,
) -> None:
    papers = shared_doi_papers()
    providers = [
        FakeProvider("openalex", [papers[0]]),
        FakeProvider("semantic_scholar", [papers[1]]),
        FakeProvider("arxiv", [papers[2]]),
    ]
    service, _, ingestion_repository = build_stack(providers)

    first = await service.ingest_research_topic("sparse experts")
    second = await service.ingest_research_topic("sparse experts")

    assert first.paper_ids == second.paper_ids
    assert count_rows(database, "papers") == 1
    assert ingestion_repository.count_ingestion_runs() == 2


async def test_partial_provider_failure_still_yields_survivors_and_audit_rows(
    build_stack: StackBuilder,
) -> None:
    survivor = shared_doi_papers()[0]
    service, _, ingestion_repository = build_stack(
        [
            FakeProvider("openalex", [survivor]),
            FakeProvider("arxiv", error=ProviderUnavailableError("socket exploded")),
        ]
    )

    result = await service.ingest_research_topic("sparse experts")

    assert result.canonical_count == 1
    assert result.provider_counts == {"openalex": 1}
    assert result.warnings == ["arxiv was unavailable; results may be partial."]
    records = ingestion_repository.list_provider_retrievals(result.run_id)
    retrievals = {record.provider: record for record in records}
    assert retrievals["openalex"].status == "ok"
    assert retrievals["openalex"].result_count == 1
    assert retrievals["arxiv"].status == "failed"
    assert retrievals["arxiv"].result_count == 0
    assert retrievals["arxiv"].safe_error == "arxiv was unavailable; results may be partial."


async def test_all_providers_failing_marks_the_run_failed(
    build_stack: StackBuilder,
) -> None:
    service, _, ingestion_repository = build_stack(
        [
            FakeProvider("openalex", error=ProviderUnavailableError("one")),
            FakeProvider("arxiv", error=ProviderUnavailableError("two")),
        ]
    )

    with pytest.raises(ProviderUnavailableError):
        await service.ingest_research_topic("sparse experts")

    runs = ingestion_repository.list_recent_runs()
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert runs[0].safe_error
    assert "http" not in runs[0].safe_error.casefold()


async def test_auto_read_defaults_to_zero_and_never_calls_the_reader(
    build_stack: StackBuilder,
) -> None:
    papers = [standalone_pdf_paper(number) for number in (1, 2, 3)]
    reader = FakeReader()
    service, _, _ = build_stack([FakeProvider("openalex", papers)], reader=reader)

    result = await service.ingest_research_topic("standalone studies")

    assert reader.calls == 0
    assert result.read_paper_ids == []


async def test_auto_read_reads_at_most_the_requested_number_of_papers(
    build_stack: StackBuilder,
) -> None:
    papers = [standalone_pdf_paper(number) for number in (1, 2, 3)]
    reader = FakeReader()
    service, _, _ = build_stack([FakeProvider("openalex", papers)], reader=reader)

    result = await service.ingest_research_topic("standalone studies", auto_read=2)

    assert len(reader.urls) == 2
    assert len(result.read_paper_ids) == 2
    assert set(result.read_paper_ids).issubset(set(result.paper_ids))


async def test_limit_above_metadata_limit_is_clamped(build_stack: StackBuilder) -> None:
    provider = FakeProvider("openalex", [])
    service, _, _ = build_stack([provider])

    await service.ingest_research_topic("sparse experts", limit=500)

    assert provider.received_limits == [50]


async def test_blank_query_raises_before_any_run_is_created(
    build_stack: StackBuilder,
) -> None:
    service, _, ingestion_repository = build_stack([FakeProvider("openalex", [])])

    with pytest.raises(ValueError, match="empty"):
        await service.ingest_research_topic("   ")
    with pytest.raises(ValueError, match="empty"):
        await service.ingest_research_topic("")

    assert ingestion_repository.count_ingestion_runs() == 0


async def test_project_id_links_every_persisted_paper(build_stack: StackBuilder) -> None:
    papers = [standalone_pdf_paper(number) for number in (1, 2)]
    service, repository, _ = build_stack([FakeProvider("openalex", papers)])
    project = repository.create_project("Radar Project")

    result = await service.ingest_research_topic("standalone studies", project_id=project.id)

    links = repository.list_project_papers(project.id)
    assert {link.paper_id for link in links} == set(result.paper_ids)


async def test_failed_provider_safe_error_leaks_no_urls_or_credentials(
    build_stack: StackBuilder,
) -> None:
    service, _, ingestion_repository = build_stack(
        [
            FakeProvider("openalex", [shared_doi_papers()[0]]),
            FakeProvider(
                "arxiv",
                error=ProviderUnavailableError(
                    "GET https://api.arxiv.test/search?key=sk-secret123 timed out"
                ),
            ),
        ]
    )

    result = await service.ingest_research_topic("sparse experts")

    records = ingestion_repository.list_provider_retrievals(result.run_id)
    retrievals = {record.provider: record for record in records}
    stored_error = (retrievals["arxiv"].safe_error or "").casefold()
    assert stored_error.startswith("arxiv was unavailable")
    assert "http" not in stored_error
    assert "key" not in stored_error
    assert "sk-secret123" not in stored_error


async def test_provider_retrieval_rows_only_record_their_own_provider_ids(
    build_stack: StackBuilder,
) -> None:
    """Each provenance row must claim only the identifiers its provider returned.

    Attributing another provider's identifiers to a retrieval row would make
    the ingestion audit trail assert something untrue about where a record came
    from, which is the one thing this table exists to get right.
    """

    papers = shared_doi_papers()
    service, _, ingestion_repository = build_stack(
        [
            FakeProvider("openalex", [papers[0]]),
            FakeProvider("semantic_scholar", [papers[1]]),
            FakeProvider("arxiv", [papers[2]]),
        ]
    )

    result = await service.ingest_research_topic("sparse experts")

    retrievals = {
        row.provider: row for row in ingestion_repository.list_provider_retrievals(result.run_id)
    }
    assert set(retrievals) == {"openalex", "semantic_scholar", "arxiv"}

    expected = {
        paper.source: paper.id.split(":", maxsplit=1)[-1] for paper in papers
    }
    for provider, row in retrievals.items():
        assert row.status == "ok"
        assert row.result_count == 1
        assert row.external_ids == [expected[provider]]
        foreign = {value for name, value in expected.items() if name != provider}
        assert not foreign.intersection(row.external_ids)


async def test_project_name_is_resolved_to_a_storage_id(
    build_stack: StackBuilder,
    database: Database,
) -> None:
    """A project NAME must be resolved before the run records a foreign key."""

    papers = shared_doi_papers()
    service, repository, ingestion_repository = build_stack(
        [FakeProvider("openalex", [papers[0]])]
    )
    project = repository.create_project(name="Sparse Experts Review")

    result = await service.ingest_research_topic(
        "sparse experts", project_id="Sparse Experts Review"
    )

    run_record = ingestion_repository.list_recent_runs()[0]
    assert run_record.status == "completed"
    assert run_record.project_id == project.id
    assert [link.paper_id for link in repository.list_project_papers(project.id)] == (
        result.paper_ids
    )


async def test_unknown_project_fails_before_a_run_is_created(
    build_stack: StackBuilder,
    database: Database,
) -> None:
    """An unknown project is rejected loudly rather than opening a doomed run."""

    papers = shared_doi_papers()
    service, _, ingestion_repository = build_stack([FakeProvider("openalex", [papers[0]])])

    with pytest.raises(ValueError, match="not found"):
        await service.ingest_research_topic("sparse experts", project_id="no-such-project")

    assert ingestion_repository.count_ingestion_runs() == 0
    assert count_rows(database, "papers") == 0
