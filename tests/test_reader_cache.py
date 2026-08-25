"""Unit tests for the reader document/card caches and the cached read flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from research_radar.artifacts.base import ARTIFACT_SUFFIXES, sha256_hex
from research_radar.artifacts.local import LocalArtifactStore
from research_radar.errors import LLMUnavailableError
from research_radar.models import Paper, PaperDocument
from research_radar.reader.cache import DocumentCache
from research_radar.reader.fetcher import FetchedPDF
from research_radar.reader.llm.base import LLMMessage, ModelT
from research_radar.reader.service import ReaderService
from research_radar.storage import Database, ResearchRepository
from research_radar.storage.ingestion_repository import IngestionRepository

URL = "https://papers.example/cached.pdf"

PAYLOAD_V1 = b"%PDF-1.4 cached-reader-fixture one"
PAYLOAD_V2 = b"%PDF-1.4 cached-reader-fixture two revised"

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


class _FakeFetcher:
    """Return canned bytes per URL without touching the network."""

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self._payloads = payloads
        self.fetch_calls = 0

    async def fetch(self, url: str) -> FetchedPDF:
        self.fetch_calls += 1
        return FetchedPDF(content=self._payloads[url], source_url=url)


class _FakeParser:
    """Return a canned PaperDocument and count parse invocations."""

    def __init__(self) -> None:
        self.parse_calls = 0

    def parse(self, content: bytes, *, source_url: str | None = None) -> PaperDocument:
        self.parse_calls += 1
        return PaperDocument(
            title="Cached Reader Paper",
            sections={
                "Abstract": "This paper studies bounded caching of parsed documents.",
                "Results": "The deterministic parser improves predictable extraction.",
            },
            full_text=(
                "Cached Reader Paper\n\n"
                "Abstract\nThis paper studies bounded caching of parsed documents.\n"
                "Results\nThe deterministic parser improves predictable extraction."
            ),
            source_url=source_url,
        )


class _CountingLLM:
    """Return a valid PaperCard and count structured generation calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[ModelT],
    ) -> ModelT:
        self.calls += 1
        return response_model.model_validate(_CARD_FIXTURE)


class _UnavailableLLM(_CountingLLM):
    """Always fail like an unconfigured language model."""

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[ModelT],
    ) -> ModelT:
        self.calls += 1
        raise LLMUnavailableError("No language model is configured.")


@pytest.fixture
def database(tmp_path: Path) -> Database:
    db = Database.create(f"sqlite:///{tmp_path / 'reader_cache.db'}")
    db.initialize_schema()
    try:
        yield db
    finally:
        db.dispose()


def _make_document_cache(
    tmp_path: Path, database: Database
) -> tuple[DocumentCache, LocalArtifactStore]:
    store = LocalArtifactStore(tmp_path / "artifacts")
    cache = DocumentCache(store=store, ingestion_repository=IngestionRepository(database))
    return cache, store


def _canned_document() -> PaperDocument:
    """Build the shared canned document used across these tests."""

    return PaperDocument(
        title="Cached Reader Paper",
        sections={
            "Abstract": "This paper studies bounded caching of parsed documents.",
            "Results": "The deterministic parser improves predictable extraction.",
        },
        full_text=(
            "Cached Reader Paper\n\n"
            "Abstract\nThis paper studies bounded caching of parsed documents.\n"
            "Results\nThe deterministic parser improves predictable extraction."
        ),
        source_url=None,
    )


def _service(
    tmp_path: Path,
    database: Database,
    llm: Any,
    *,
    payloads: dict[str, bytes],
    with_cache: bool = True,
) -> tuple[ReaderService, _FakeParser, LocalArtifactStore]:
    parser = _FakeParser()
    store = LocalArtifactStore(tmp_path / "artifacts")
    cache: DocumentCache | None = None
    if with_cache:
        cache = DocumentCache(store=store, ingestion_repository=IngestionRepository(database))
    service = ReaderService(
        fetcher=_FakeFetcher(payloads),
        parser=parser,
        llm=llm,
        repository=ResearchRepository(database),
        llm_provider_name="counting-test",
        document_cache=cache,
    )
    return service, parser, store


async def test_first_read_parses_once_calls_llm_once_and_writes_artifacts(
    tmp_path: Path, database: Database
) -> None:
    llm = _CountingLLM()
    service, parser, store = _service(tmp_path, database, llm, payloads={URL: PAYLOAD_V1})

    result = await service.read_url(URL)

    sha = sha256_hex(PAYLOAD_V1)
    assert result.sha256 == sha
    assert result.from_cache is False
    assert parser.parse_calls == 1
    assert llm.calls == 1
    for artifact_type in ("pdf", "text", "sections"):
        artifact_file = (
            store.root / "papers" / result.paper_id / f"{sha}{ARTIFACT_SUFFIXES[artifact_type]}"
        )
        assert artifact_file.is_file(), artifact_type


async def test_second_read_of_same_bytes_skips_parser_and_llm(
    tmp_path: Path, database: Database
) -> None:
    llm = _CountingLLM()
    service, parser, _ = _service(tmp_path, database, llm, payloads={URL: PAYLOAD_V1})

    first = await service.read_url(URL)
    second = await service.read_url(URL)

    assert parser.parse_calls == 1
    assert llm.calls == 1
    assert second.from_cache is True
    assert second.sha256 == first.sha256 == sha256_hex(PAYLOAD_V1)
    assert second.card == first.card


async def test_force_refresh_calls_the_llm_again(tmp_path: Path, database: Database) -> None:
    llm = _CountingLLM()
    service, parser, _ = _service(tmp_path, database, llm, payloads={URL: PAYLOAD_V1})

    await service.read_url(URL)
    refreshed = await service.read_url(URL, force_refresh=True)

    assert llm.calls == 2
    assert parser.parse_calls == 2
    assert refreshed.from_cache is False


async def test_changed_bytes_add_artifacts_and_keep_originals(
    tmp_path: Path, database: Database
) -> None:
    llm = _CountingLLM()
    payloads = {URL: PAYLOAD_V1}
    service, parser, store = _service(tmp_path, database, llm, payloads=payloads)

    first = await service.read_url(URL)
    payloads[URL] = PAYLOAD_V2
    second = await service.read_url(URL)

    old_sha = sha256_hex(PAYLOAD_V1)
    new_sha = sha256_hex(PAYLOAD_V2)
    assert first.paper_id == second.paper_id
    assert second.sha256 == new_sha != old_sha
    assert llm.calls == 2
    assert parser.parse_calls == 2
    assert store.read(paper_id=second.paper_id, sha256=old_sha, artifact_type="pdf") == PAYLOAD_V1
    assert store.read(paper_id=second.paper_id, sha256=new_sha, artifact_type="pdf") == PAYLOAD_V2
    for sha in (old_sha, new_sha):
        for artifact_type in ("pdf", "text", "sections"):
            suffix = ARTIFACT_SUFFIXES[artifact_type]
            assert (store.root / "papers" / second.paper_id / f"{sha}{suffix}").is_file()


def _seed_paper(database: Database, slug: str) -> str:
    """Persist a real canonical paper and return its storage id.

    ``document_artifacts.paper_id`` is a foreign key, so an artifact can only be
    recorded against a paper that actually exists.
    """

    return ResearchRepository(database).upsert_merged_paper(
        Paper(
            id=f"url:{slug}",
            title=f"Cached Reader Fixture {slug}",
            abstract=None,
            authors=[],
            publication_year=None,
            venue=None,
            doi=None,
            url=f"https://papers.example/{slug}.pdf",
            citation_count=None,
            source="direct_pdf",
            external_ids={"url_sha256": slug},
        )
    )


async def test_document_cache_load_returns_none_without_artifacts(
    tmp_path: Path, database: Database
) -> None:
    cache, _ = _make_document_cache(tmp_path, database)

    missing = cache.load(paper_id="paper-none", sha256="a" * 64)

    assert missing is None


async def test_store_then_load_round_trips_document(tmp_path: Path, database: Database) -> None:
    cache, _ = _make_document_cache(tmp_path, database)
    document = _canned_document().model_copy(update={"source_url": URL})
    paper_id = _seed_paper(database, "roundtrip")

    sha = cache.store(
        paper_id=paper_id,
        content=b"%PDF-roundtrip",
        document=document,
        source_url=URL,
    )
    loaded = cache.load(paper_id=paper_id, sha256=sha)

    assert loaded is not None
    assert loaded.from_cache is True
    assert loaded.sha256 == sha
    assert loaded.document.title == document.title
    assert loaded.document.sections == document.sections
    assert loaded.document.full_text == document.full_text
    assert loaded.document.source_url == URL
    assert loaded.source_url == URL


async def test_corrupt_sections_artifact_returns_none_instead_of_raising(
    tmp_path: Path, database: Database
) -> None:
    cache, store = _make_document_cache(tmp_path, database)
    paper_id = _seed_paper(database, "corrupt")
    sha = cache.store(
        paper_id=paper_id,
        content=b"%PDF-corrupt",
        document=_canned_document(),
        source_url=URL,
    )
    sections_file = next((store.root / "papers").rglob(f"{sha}.sections.json"))
    sections_file.write_bytes(b"not json")

    assert cache.load(paper_id=paper_id, sha256=sha) is None


async def test_storing_same_content_twice_records_one_row_per_type(
    tmp_path: Path, database: Database
) -> None:
    cache, _ = _make_document_cache(tmp_path, database)
    paper_id = _seed_paper(database, "duplicate")

    first_sha = cache.store(
        paper_id=paper_id,
        content=b"%PDF-duplicate",
        document=_canned_document(),
        source_url=URL,
    )
    second_sha = cache.store(
        paper_id=paper_id,
        content=b"%PDF-duplicate",
        document=_canned_document(),
        source_url=URL,
    )

    ingestion = IngestionRepository(database)
    assert first_sha == second_sha
    assert ingestion.count_artifacts() == 3
    for artifact_type in ("pdf", "text", "sections"):
        stored = ingestion.get_artifact(paper_id, first_sha, artifact_type)
        assert stored is not None


async def test_service_without_document_cache_still_works_end_to_end(
    tmp_path: Path, database: Database
) -> None:
    llm = _CountingLLM()
    service, parser, _ = _service(
        tmp_path, database, llm, payloads={URL: PAYLOAD_V1}, with_cache=False
    )

    result = await service.read_url(URL)

    assert parser.parse_calls == 1
    assert llm.calls == 1
    assert result.from_cache is False
    assert result.sha256 == sha256_hex(PAYLOAD_V1)
    assert ResearchRepository(database).get_paper_card(result.paper_id) == result.card


async def test_llm_unavailable_error_still_propagates(tmp_path: Path, database: Database) -> None:
    llm = _UnavailableLLM()
    service, parser, _ = _service(tmp_path, database, llm, payloads={URL: PAYLOAD_V1})
    repository = ResearchRepository(database)

    with pytest.raises(LLMUnavailableError):
        await service.read_url(URL)

    assert llm.calls == 1
    assert parser.parse_calls == 1
    stored_papers = repository.get_papers_for_local_lexical_search("Cached Reader Paper")
    assert stored_papers
    assert all(repository.get_paper_card(paper.id) is None for paper in stored_papers)
