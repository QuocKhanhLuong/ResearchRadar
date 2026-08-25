"""W12 adversarial cross-component phase-regression tests.

Every test is hermetic: no network, no real LLM, no Pinecone credential, no
model download. A failure here should read like a bug report.
"""

from __future__ import annotations

import json
import logging
import socket
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel
from sqlalchemy import text

from research_radar.artifacts.local import LocalArtifactStore
from research_radar.config import Settings
from research_radar.errors import (
    LLMResponseError,
    LLMUnavailableError,
    ProviderUnavailableError,
)
from research_radar.main import build_application_bot
from research_radar.models import Paper, PaperDocument
from research_radar.providers.openalex import OpenAlexProvider
from research_radar.providers.semantic_scholar import SemanticScholarProvider
from research_radar.reader.cache import DocumentCache
from research_radar.reader.fetcher import FetchedPDF
from research_radar.reader.llm import LLMMessage, RemoteLLMProvider
from research_radar.reader.llm.base import ModelT
from research_radar.reader.service import ReaderService
from research_radar.research.canonical import canonicalize
from research_radar.research.hybrid import HybridRetriever
from research_radar.research.ingestion import IngestionService
from research_radar.research.scout import ScoutService
from research_radar.semantic.base import SemanticRecord
from research_radar.semantic.embedding import FakeEmbeddingProvider, embedding_fingerprint
from research_radar.semantic.index import FakeSemanticIndex
from research_radar.storage.database import Database, create_database, initialize_schema
from research_radar.storage.ingestion_repository import IngestionRepository
from research_radar.storage.repositories import ResearchRepository

CANARY_KEY = "sk-test-LEAK-CANARY-9182"
READER_URL = "https://papers.example/regression.pdf"
READER_PDF_BYTES = b"%PDF-1.4 regression-reader-fixture bytes"

_CARD_FIXTURE: dict[str, Any] = {
    "paper_id": "untrusted-id",
    "problem": "Bounded paper analysis",
    "contributions": ["Content-addressed caching"],
    "main_claims": [
        {
            "claim": "Parsing is predictable",
            "source_section": "Results",
            "supporting_text": "deterministic parser improves predictable extraction",
        }
    ],
}


# ---------------------------------------------------------------------------
# Shared hermetic fakes and helpers
# ---------------------------------------------------------------------------


class FakeProvider:
    """Canned scholarly provider that returns papers or raises a fixed error."""

    def __init__(
        self,
        name: str,
        papers: list[Paper] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.papers = list(papers or [])
        self.error = error

    async def search(self, query: str, limit: int) -> list[Paper]:
        """Return the canned papers unless constructed with an error."""

        if self.error is not None:
            raise self.error
        return self.papers


class _FakeFetcher:
    """Return canned bytes per URL without touching the network."""

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self._payloads = payloads
        self.fetch_calls = 0

    async def fetch(self, url: str) -> FetchedPDF:
        """Record the fetch and hand back the canned bytes."""

        self.fetch_calls += 1
        return FetchedPDF(content=self._payloads[url], source_url=url)


class _CannedParser:
    """Parse nothing and return a fixed document with verifiable sections."""

    def parse(self, content: bytes, *, source_url: str | None = None) -> PaperDocument:
        """Return the canned parsed document."""

        return PaperDocument(
            title="Regression Reader Paper",
            sections={
                "Abstract": "This paper studies bounded caching of parsed documents.",
                "Results": "The deterministic parser improves predictable extraction.",
            },
            full_text=(
                "Regression Reader Paper\n\n"
                "Abstract\nThis paper studies bounded caching of parsed documents.\n"
                "Results\nThe deterministic parser improves predictable extraction."
            ),
            source_url=source_url,
        )


class _CountingLLM:
    """Count structured-generation calls and always answer with a valid card."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[ModelT],
    ) -> ModelT:
        """Record the call and validate the canned card fixture."""

        self.calls += 1
        return response_model.model_validate(_CARD_FIXTURE)


class _ScriptedLLMTransport:
    """Queue canned HTTP responses while recording every outgoing request."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("unexpected extra HTTP request")
        return self._responses.pop(0)

    @property
    def payloads(self) -> list[dict[str, Any]]:
        """Decode every recorded request body as JSON."""

        return [json.loads(request.content) for request in self.requests]


class _BrokenSearchIndex:
    """Semantic index whose search always fails after recording the attempt."""

    backend = "broken"

    def __init__(self) -> None:
        self.search_calls = 0

    @property
    def available(self) -> bool:
        """Report available so the retriever really attempts the search."""

        return True

    def upsert(self, records: Sequence[SemanticRecord]) -> int:
        """Accept records without storing them."""

        return 0

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 10,
        entity_type: str | None = None,
    ) -> list[object]:
        """Fail exactly like an unreachable Pinecone backend."""

        self.search_calls += 1
        raise RuntimeError("vector index exploded")

    def delete(self, entity_ids: Sequence[str]) -> int:
        """Report zero removals."""

        return 0

    def status(self) -> None:
        """Return no status payload."""

        return None


class _Answer(BaseModel):
    """Minimal structured target model for remote-LLM wire tests."""

    answer: str
    confidence: int


def count_rows(database: Database, table: str) -> int:
    """Return the raw row count of one table for persistence assertions."""

    with database.engine.connect() as connection:
        return int(connection.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one())


def column_values(database: Database, table: str) -> list[str]:
    """Return every cell of a table stringified, for credential-leak audits."""

    with database.engine.connect() as connection:
        rows = connection.execute(text(f"SELECT * FROM {table}")).mappings().all()
    return [str(value) for row in rows for value in row.values()]


def build_ingestion_stack(
    database: Database,
    providers: list[FakeProvider],
) -> tuple[IngestionService, ResearchRepository, IngestionRepository]:
    """Wire an ingestion service and its repositories over one database."""

    repository = ResearchRepository(database)
    ingestion_repository = IngestionRepository(database)
    service = IngestionService(
        scout=ScoutService(providers),
        repository=repository,
        ingestion_repository=ingestion_repository,
    )
    return service, repository, ingestion_repository


def seed_paper(repository: ResearchRepository, slug: str, title: str) -> str:
    """Persist one canonical paper and return its stable storage id."""

    return repository.upsert_merged_paper(
        Paper(
            id=f"openalex:{slug}",
            title=title,
            abstract=f"{title} background material.",
            authors=["Ada Lovelace"],
            publication_year=2024,
            venue="Journal of Testing",
            doi=f"10.1000/{slug}",
            url=f"https://example.test/{slug}",
            citation_count=3,
            source="openalex",
            external_ids={"openalex": slug},
        )
    )


def index_paper(index: FakeSemanticIndex, paper_id: str, vector: list[float]) -> None:
    """Write one derived paper vector into the fake index."""

    index.upsert(
        [
            SemanticRecord(
                entity_id=f"paper:{paper_id}",
                entity_type="paper",
                paper_id=paper_id,
                vector=vector,
                publication_year=2024,
                embedding_schema_version="paper-v1",
                embedding_model="fake-embedding-v1",
            )
        ]
    )


def _llm_success(content: str) -> httpx.Response:
    """Return a 200 chat-completions envelope wrapping ``content``."""

    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


async def call_remote_llm(transport: _ScriptedLLMTransport, *, api_key: str | None) -> Any:
    """Run one generate_structured call against a scripted transport."""

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        provider = RemoteLLMProvider(
            base_url="https://llm.example/v1",
            model="test-model",
            api_key=api_key,
            client=client,
            timeout_seconds=2,
        )
        return await provider.generate_structured(
            [LLMMessage(role="user", content="Analyze this")], _Answer
        )


@pytest.fixture
def database() -> Iterator[Database]:
    """Provide one fresh initialized in-memory SQLite database."""

    database = create_database("sqlite:///:memory:")
    initialize_schema(database)
    yield database
    database.dispose()


# ---------------------------------------------------------------------------
# 1. CROSS-PROVIDER IDENTITY
# ---------------------------------------------------------------------------


async def test_shared_normalized_doi_from_three_providers_collapses_to_one_canonical_row(
    database: Database,
) -> None:
    """One DOI identity must yield one paper row and three provider rows."""

    papers = [
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

    canonical = canonicalize(papers)
    assert len(canonical) == 1
    assert len(canonical[0].contributing_papers) == 3

    _, repository, _ = build_ingestion_stack(database, [])
    stored_id = repository.upsert_merged_paper(canonical[0].paper)

    assert count_rows(database, "papers") == 1
    sources = repository.list_paper_sources(stored_id)
    provider_names = {source.provider for source in sources}
    assert {"openalex", "semantic_scholar", "arxiv"}.issubset(provider_names)
    non_doi_sources = [source for source in sources if source.provider != "doi"]
    assert {source.provider for source in non_doi_sources} == {
        "openalex",
        "semantic_scholar",
        "arxiv",
    }
    # Storage deliberately retains the normalized DOI as one extra identity row
    # so later scans resolve papers by DOI alone (find_paper_id_by_source);
    # anything besides those three providers plus "doi" is a provenance defect.
    assert count_rows(database, "paper_sources") == len(non_doi_sources) + 1
    assert provider_names - {"doi"} == {"openalex", "semantic_scholar", "arxiv"}
    stored = repository.get_paper(stored_id)
    assert stored is not None
    assert stored.doi == "10.1000/radar"


# ---------------------------------------------------------------------------
# 2. INGESTION IDEMPOTENCY
# ---------------------------------------------------------------------------


async def test_repeated_ingestion_of_identical_output_duplicates_no_papers_but_keeps_two_runs(
    database: Database,
) -> None:
    """Runs are an audit log; papers must not multiply across repeat ingests."""

    papers = [
        Paper(
            id="openalex:W1",
            title="Deterministic Dedup Study",
            doi="10.5000/dedup",
            source="openalex",
            external_ids={"openalex": "W1"},
        ),
        Paper(
            id="semantic_scholar:s2-dedup",
            title="deterministic dedup study!",
            doi="10.5000/dedup",
            source="semantic_scholar",
            external_ids={"semantic_scholar": "s2-dedup"},
        ),
    ]
    providers = [
        FakeProvider("openalex", [papers[0]]),
        FakeProvider("semantic_scholar", [papers[1]]),
    ]
    service, _, ingestion_repository = build_ingestion_stack(database, providers)

    first = await service.ingest_research_topic("dedup study")
    papers_after_first = count_rows(database, "papers")
    second = await service.ingest_research_topic("dedup study")

    assert first.paper_ids == second.paper_ids
    assert count_rows(database, "papers") == papers_after_first == 1
    assert ingestion_repository.count_ingestion_runs() == 2
    assert all(run.status == "completed" for run in ingestion_repository.list_recent_runs())


# ---------------------------------------------------------------------------
# 3. PARTIAL PROVIDER OUTAGE
# ---------------------------------------------------------------------------


async def test_partial_outage_keeps_survivor_papers_and_marks_failed_provider_in_audit(
    database: Database,
) -> None:
    """A down provider must warn, audit-fail, and never erase survivor results."""

    service, repository, ingestion_repository = build_ingestion_stack(
        database,
        [
            FakeProvider(
                "openalex",
                [
                    Paper(
                        id="openalex:W-survivor",
                        title="Survivor Study Alpha",
                        doi="10.7000/alpha",
                        source="openalex",
                        external_ids={"openalex": "W-survivor"},
                    )
                ],
            ),
            FakeProvider(
                "semantic_scholar",
                error=ProviderUnavailableError("socket exploded"),
            ),
            FakeProvider(
                "arxiv",
                [
                    Paper(
                        id="arxiv:2401.00099",
                        title="Survivor Study Beta",
                        doi="10.7000/beta",
                        source="arxiv",
                        external_ids={"arxiv": "2401.00099"},
                    )
                ],
            ),
        ],
    )

    result = await service.ingest_research_topic("survivor studies")

    assert result.warnings, "a partial outage must surface at least one warning"
    assert any("semantic_scholar" in warning for warning in result.warnings)
    assert "semantic_scholar" not in result.provider_counts
    assert count_rows(database, "papers") == 2
    assert len(result.paper_ids) == 2

    retrievals = {
        record.provider: record
        for record in ingestion_repository.list_provider_retrievals(result.run_id)
    }
    assert set(retrievals) == {"openalex", "semantic_scholar", "arxiv"}
    assert retrievals["semantic_scholar"].status == "failed"
    assert retrievals["semantic_scholar"].result_count == 0
    assert retrievals["openalex"].status == "ok"
    assert retrievals["arxiv"].status == "ok"

    stored_titles = {
        repository.get_paper(paper_id).title  # type: ignore[union-attr]
        for paper_id in result.paper_ids
    }
    assert stored_titles == {"Survivor Study Alpha", "Survivor Study Beta"}


async def test_all_three_providers_failing_raises_provider_unavailable_error(
    database: Database,
) -> None:
    """A total outage is an error, never a silently empty success."""

    service, _, _ = build_ingestion_stack(
        database,
        [
            FakeProvider("openalex", error=ProviderUnavailableError("one")),
            FakeProvider("semantic_scholar", error=ProviderUnavailableError("two")),
            FakeProvider("arxiv", error=ProviderUnavailableError("three")),
        ],
    )

    with pytest.raises(ProviderUnavailableError):
        await service.ingest_research_topic("doomed topic")


# ---------------------------------------------------------------------------
# 4/5. CONTENT-ADDRESSED ARTIFACTS
# ---------------------------------------------------------------------------


def _artifact_fixture(
    database: Database, tmp_path: Path
) -> tuple[ResearchRepository, IngestionRepository, LocalArtifactStore, str]:
    """Seed one paper plus its artifact store and ingestion repository."""

    repository = ResearchRepository(database)
    paper_id = seed_paper(repository, "artifact-fixture", "Artifact Fixture Paper")
    store = LocalArtifactStore(tmp_path / "artifacts")
    return repository, IngestionRepository(database), store, paper_id


def test_putting_identical_pdf_bytes_twice_yields_one_object_and_one_artifact_row(
    database: Database, tmp_path: Path
) -> None:
    """Same bytes must mean same object key, one file, one recorded reference."""

    _, ingestion_repository, store, paper_id = _artifact_fixture(database, tmp_path)
    payload = b"%PDF-1.4 identical-artifact-bytes"

    first_ref = store.put(paper_id=paper_id, content=payload, artifact_type="pdf")
    second_ref = store.put(paper_id=paper_id, content=payload, artifact_type="pdf")
    ingestion_repository.record_artifact(first_ref)
    ingestion_repository.record_artifact(second_ref)

    assert first_ref.object_key == second_ref.object_key
    pdf_files = list((store.root / "papers" / paper_id).glob("*.pdf"))
    assert len(pdf_files) == 1
    assert len(ingestion_repository.list_artifacts(paper_id)) == 1


def test_changed_pdf_bytes_create_a_second_distinct_readable_artifact_version(
    database: Database, tmp_path: Path
) -> None:
    """A new content digest must add a version row and keep both files readable."""

    _, ingestion_repository, store, paper_id = _artifact_fixture(database, tmp_path)
    payload_v1 = b"%PDF-1.4 artifact version one"
    payload_v2 = b"%PDF-1.4 artifact version two revised"

    ref_v1 = store.put(paper_id=paper_id, content=payload_v1, artifact_type="pdf")
    ref_v2 = store.put(paper_id=paper_id, content=payload_v2, artifact_type="pdf")
    ingestion_repository.record_artifact(ref_v1)
    ingestion_repository.record_artifact(ref_v2)

    assert ref_v1.sha256 != ref_v2.sha256
    assert ref_v1.object_key != ref_v2.object_key
    pdf_files = sorted((store.root / "papers" / paper_id).glob("*.pdf"))
    assert len(pdf_files) == 2
    assert store.read(paper_id=paper_id, sha256=ref_v1.sha256, artifact_type="pdf") == payload_v1
    assert store.read(paper_id=paper_id, sha256=ref_v2.sha256, artifact_type="pdf") == payload_v2
    assert len(ingestion_repository.list_artifacts(paper_id)) == 2


# ---------------------------------------------------------------------------
# 6. PAPERCARD CACHE HIT -- the cost-control guarantee
# ---------------------------------------------------------------------------


async def test_repeat_read_of_same_document_makes_no_second_llm_call(tmp_path: Path) -> None:
    """Two reads of byte-identical content must spend exactly one LLM call."""

    database = create_database(f"sqlite:///{tmp_path / 'reader_regression.db'}")
    initialize_schema(database)
    try:
        llm = _CountingLLM()
        service = ReaderService(
            fetcher=_FakeFetcher({READER_URL: READER_PDF_BYTES}),
            parser=_CannedParser(),
            llm=llm,
            repository=ResearchRepository(database),
            llm_provider_name="counting-test",
            document_cache=DocumentCache(
                store=LocalArtifactStore(tmp_path / "artifacts"),
                ingestion_repository=IngestionRepository(database),
            ),
        )

        first = await service.read_url(READER_URL)
        second = await service.read_url(READER_URL)

        assert llm.calls == 1
        assert second.from_cache is True
        assert second.card == first.card
        assert second.paper_id == first.paper_id

        refreshed = await service.read_url(READER_URL, force_refresh=True)

        assert llm.calls == 2
        assert refreshed.from_cache is False
    finally:
        database.dispose()


# ---------------------------------------------------------------------------
# 7. SEMANTIC HIT RESOLVES TO CANONICAL EVIDENCE
# ---------------------------------------------------------------------------


async def test_semantic_hit_resolves_through_sqlite_to_real_stored_paper(
    database: Database,
) -> None:
    """A vector candidate is only an address: evidence must load from SQLite."""

    repository = ResearchRepository(database)
    expected_title = "Sparse mixture of experts routing"
    paper_id = seed_paper(repository, "routing-paper", expected_title)

    embedding = FakeEmbeddingProvider(dimension=8)
    index = FakeSemanticIndex()
    index_paper(index, paper_id, embedding.embed_texts([expected_title])[0])

    retriever = HybridRetriever(
        repository=repository,
        embedding_provider=embedding,
        semantic_index=index,
    )

    candidates = retriever.retrieve("sparse mixture of experts routing")

    assert candidates, "an indexed, stored paper must surface as a candidate"
    resolved = repository.get_paper(candidates[0].paper_id)
    assert resolved is not None, "candidate id must resolve to a real StoredPaper"
    assert resolved.id == candidates[0].paper_id
    assert resolved.title == expected_title


# ---------------------------------------------------------------------------
# 8. DELETED / MISSING ENTITY DISCARDED
# ---------------------------------------------------------------------------


async def test_semantic_hit_for_missing_paper_is_discarded_without_raising(
    database: Database,
) -> None:
    """Stale index entries pointing nowhere must vanish, never crash retrieval."""

    repository = ResearchRepository(database)
    embedding = FakeEmbeddingProvider(dimension=8)
    index = FakeSemanticIndex()
    index_paper(index, "ghost-paper-id", embedding.embed_texts(["Ghost study"])[0])

    retriever = HybridRetriever(
        repository=repository,
        embedding_provider=embedding,
        semantic_index=index,
    )

    candidates = retriever.retrieve("ghost study")

    assert "ghost-paper-id" not in {candidate.paper_id for candidate in candidates}
    assert candidates == []


# ---------------------------------------------------------------------------
# 9. PINECONE UNAVAILABLE -> LEXICAL STILL WORKS, NO RETRY STORM
# ---------------------------------------------------------------------------


async def test_failing_vector_index_degrades_to_lexical_order_with_exactly_one_attempt(
    database: Database,
) -> None:
    """An index outage must not change results, raise, or hammer the endpoint."""

    repository = ResearchRepository(database)
    seed_paper(repository, "alpha", "Sparse mixture of experts routing")
    seed_paper(repository, "beta", "Sparse expert systems")
    seed_paper(repository, "gamma", "Crystallography of proteins")

    embedding = FakeEmbeddingProvider(dimension=8)
    broken_index = _BrokenSearchIndex()
    degraded = HybridRetriever(
        repository=repository,
        embedding_provider=embedding,
        semantic_index=broken_index,
    )
    lexical_only = HybridRetriever(repository=repository)

    degraded_candidates = degraded.retrieve("sparse routing experts")
    lexical_candidates = lexical_only.retrieve("sparse routing experts")

    assert [candidate.paper_id for candidate in degraded_candidates] == [
        candidate.paper_id for candidate in lexical_candidates
    ]
    assert degraded_candidates
    assert all(candidate.semantic_rank is None for candidate in degraded_candidates)
    assert broken_index.search_calls == 1


# ---------------------------------------------------------------------------
# 10. EMBEDDING SCHEMA CHANGE IS VISIBLE
# ---------------------------------------------------------------------------


def test_embedding_fingerprint_reports_schema_version_model_and_dimension() -> None:
    """The paper fingerprint must expose exactly what a silent reindex hides."""

    provider = FakeEmbeddingProvider(dimension=8, model_id="fake-a")
    fingerprint = embedding_fingerprint(provider, entity_type="paper")
    assert fingerprint == ("paper-v1", "fake-a", 8)


def test_swapping_embedding_model_or_dimension_changes_the_fingerprint() -> None:
    """A different model_id or dimension must produce a different tuple."""

    baseline = embedding_fingerprint(
        FakeEmbeddingProvider(dimension=8, model_id="fake-a"), entity_type="paper"
    )
    other_model = embedding_fingerprint(
        FakeEmbeddingProvider(dimension=8, model_id="fake-b"), entity_type="paper"
    )
    other_dimension = embedding_fingerprint(
        FakeEmbeddingProvider(dimension=16, model_id="fake-a"), entity_type="paper"
    )

    assert baseline != other_model
    assert baseline != other_dimension


# ---------------------------------------------------------------------------
# 11-13. REMOTE LLM STRUCTURED CALLS
# ---------------------------------------------------------------------------


async def test_remote_llm_valid_json_response_validates_into_pydantic_model() -> None:
    """One well-formed chat-completions exchange yields one validated model."""

    transport = _ScriptedLLMTransport([_llm_success('{"answer":"Grounded","confidence":3}')])

    answer = await call_remote_llm(transport, api_key=None)

    assert answer == _Answer(answer="Grounded", confidence=3)
    assert len(transport.requests) == 1
    assert transport.payloads[0]["response_format"] == {"type": "json_object"}


async def test_response_format_rejection_retries_exactly_once_without_response_format() -> None:
    """An endpoint rejecting response_format earns exactly one fallback retry."""

    transport = _ScriptedLLMTransport(
        [
            httpx.Response(
                400,
                json={"error": {"message": "Unsupported parameter: response_format"}},
            ),
            _llm_success({"answer": "Fallback ok", "confidence": 2}),
        ]
    )

    answer = await call_remote_llm(transport, api_key=None)

    assert answer == _Answer(answer="Fallback ok", confidence=2)
    assert len(transport.requests) == 2
    assert "response_format" not in transport.payloads[1]
    fallback_message = transport.payloads[1]["messages"][-1]
    assert fallback_message["role"] == "system"


async def test_unrelated_bad_request_is_never_retried() -> None:
    """A 400 that does not name response_format must fail after one request."""

    transport = _ScriptedLLMTransport(
        [httpx.Response(400, json={"error": {"message": "max_tokens is too large"}})]
    )

    with pytest.raises(LLMUnavailableError, match="HTTP 400"):
        await call_remote_llm(transport, api_key=None)

    assert len(transport.requests) == 1


async def test_server_error_is_never_retried() -> None:
    """A 500 must surface immediately after exactly one request."""

    transport = _ScriptedLLMTransport([httpx.Response(500, text="boom")])

    with pytest.raises(LLMUnavailableError, match="HTTP 500"):
        await call_remote_llm(transport, api_key=None)

    assert len(transport.requests) == 1


async def test_schema_invalid_structured_output_raises_without_any_retry() -> None:
    """A 200 whose JSON breaks the schema raises LLMResponseError, unretried."""

    transport = _ScriptedLLMTransport([_llm_success('{"answer":"confidence is missing"}')])

    with pytest.raises(LLMResponseError, match="invalid structured response"):
        await call_remote_llm(transport, api_key=None)

    assert len(transport.requests) == 1


# ---------------------------------------------------------------------------
# 14. API KEY NEVER LEAKS
# ---------------------------------------------------------------------------


async def test_api_keys_travel_only_in_headers_never_in_request_urls() -> None:
    """OpenAlex, Semantic Scholar, and the remote LLM must keep keys in headers."""

    captured: list[httpx.Request] = []

    def openalex_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"results": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(openalex_handler)) as client:
        assert await OpenAlexProvider(client, api_key=CANARY_KEY).search("topic", limit=1) == []
    assert CANARY_KEY in captured[-1].headers["authorization"]
    assert CANARY_KEY not in str(captured[-1].url)

    def s2_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"data": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(s2_handler)) as client:
        assert (
            await SemanticScholarProvider(client, api_key=CANARY_KEY).search("topic", limit=1)
            == []
        )
    assert CANARY_KEY in captured[-1].headers["x-api-key"]
    assert CANARY_KEY not in str(captured[-1].url)

    def llm_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _llm_success('{"answer":"ok","confidence":1}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(llm_handler)) as client:
        provider = RemoteLLMProvider(
            base_url="https://llm.example/v1",
            model="test-model",
            api_key=CANARY_KEY,
            client=client,
            timeout_seconds=2,
        )
        await provider.generate_structured([LLMMessage(role="user", content="hi")], _Answer)
    assert captured[-1].headers["authorization"] == f"Bearer {CANARY_KEY}"
    assert CANARY_KEY not in str(captured[-1].url)
    assert CANARY_KEY not in captured[-1].content.decode()


@pytest.mark.parametrize("status", [401, 500])
async def test_provider_and_llm_error_messages_never_contain_the_api_key(
    status: int,
) -> None:
    """Exceptions from failing providers and LLM calls must hide the sentinel."""

    def openalex_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="denied")

    async with httpx.AsyncClient(transport=httpx.MockTransport(openalex_handler)) as client:
        with pytest.raises(ProviderUnavailableError) as openalex_error:
            await OpenAlexProvider(client, api_key=CANARY_KEY).search("topic", limit=1)
    assert CANARY_KEY not in str(openalex_error.value)
    assert CANARY_KEY not in repr(openalex_error.value)

    def s2_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="denied")

    async with httpx.AsyncClient(transport=httpx.MockTransport(s2_handler)) as client:
        with pytest.raises(ProviderUnavailableError) as s2_error:
            await SemanticScholarProvider(client, api_key=CANARY_KEY).search("topic", limit=1)
    assert CANARY_KEY not in str(s2_error.value)
    assert CANARY_KEY not in repr(s2_error.value)

    transport = _ScriptedLLMTransport([httpx.Response(status, text="denied")])
    with pytest.raises(LLMUnavailableError) as llm_error:
        await call_remote_llm(transport, api_key=CANARY_KEY)
    assert CANARY_KEY not in str(llm_error.value)
    assert CANARY_KEY not in repr(llm_error.value)


async def test_debug_logs_around_failing_calls_never_contain_the_api_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Whatever the failure paths log at DEBUG level must stay credential-free."""

    def openalex_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="denied")

    async with httpx.AsyncClient(transport=httpx.MockTransport(openalex_handler)) as client:
        scout = ScoutService([OpenAlexProvider(client, api_key=CANARY_KEY)])
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ProviderUnavailableError):
                await scout.search("topic", 5)
    assert CANARY_KEY not in caplog.text

    transport = _ScriptedLLMTransport([httpx.Response(500, text="denied")])
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(LLMUnavailableError):
            await call_remote_llm(transport, api_key=CANARY_KEY)
    assert CANARY_KEY not in caplog.text


async def test_ingestion_audit_tables_never_persist_the_api_key(database: Database) -> None:
    """Even a provider error carrying the key must not reach any audit column."""

    service, _, _ = build_ingestion_stack(
        database,
        [
            FakeProvider(
                "openalex",
                [
                    Paper(
                        id="openalex:W-clean",
                        title="Clean Survivor Study",
                        doi="10.9000/clean",
                        source="openalex",
                        external_ids={"openalex": "W-clean"},
                    )
                ],
            ),
            FakeProvider(
                "semantic_scholar",
                error=ProviderUnavailableError(f"auth rejected key {CANARY_KEY}"),
            ),
        ],
    )

    result = await service.ingest_research_topic("leak hunt")

    assert count_rows(database, "papers") == 1
    assert any("semantic_scholar" in warning for warning in result.warnings)
    for table in ("ingestion_runs", "provider_retrievals"):
        assert column_values(database, table), f"{table} should hold audit rows"
        for value in column_values(database, table):
            assert CANARY_KEY not in value


# ---------------------------------------------------------------------------
# 15. EXISTING SUITE UNBROKEN -- offline composition still builds
# ---------------------------------------------------------------------------


async def test_application_bot_constructs_offline_with_safe_default_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default-safe settings must compose the full bot with zero network access."""

    db_file = tmp_path / "phase_regression.db"
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("ARTIFACT_ROOT", str(artifacts_dir))

    def _forbid_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("a network socket was created during construction")

    monkeypatch.setattr(socket, "socket", _forbid_network)

    settings = Settings(
        database_url=f"sqlite:///{db_file}",
        artifact_root=str(artifacts_dir),
        llm_provider="mock",
        embedding_provider="disabled",
        semantic_index="disabled",
        _env_file=None,
    )

    bot = build_application_bot(settings)
    try:
        assert bot.tree.get_commands(), "composed bot should register slash commands"
    finally:
        await bot.close_owned_resources()
        await bot.close()
