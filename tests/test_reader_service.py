from __future__ import annotations

from pathlib import Path

import httpx
import pymupdf
import pytest

from research_radar.errors import LLMUnavailableError
from research_radar.reader import DirectPDFFetcher, PDFParser, ReaderService
from research_radar.reader.llm import MockLLMProvider
from research_radar.storage import Database, ResearchRepository


def _pdf_bytes() -> bytes:
    document = pymupdf.open()
    try:
        page = document.new_page()
        text = """Useful Paper
Abstract
This paper has enough readable text to pass quality checks. It describes a bounded research reader.
Introduction
The context establishes why evidence support matters in paper analysis.
Method
We use a deterministic parser for reliable section extraction and analysis.
Results
The deterministic parser improves predictable extraction behavior.
Conclusion
The workflow is bounded and preserves evidence provenance.
"""
        assert page.insert_textbox(pymupdf.Rect(36, 36, 560, 806), text, fontsize=10) >= 0
        return document.tobytes()
    finally:
        document.close()


@pytest.fixture
def repository(tmp_path: Path) -> ResearchRepository:
    database = Database.create(f"sqlite:///{tmp_path / 'reader.db'}")
    database.initialize_schema()
    try:
        yield ResearchRepository(database)
    finally:
        database.dispose()


@pytest.mark.asyncio
async def test_reader_service_persists_validated_card(repository: ResearchRepository) -> None:
    payload = _pdf_bytes()

    async def resolver(hostname: str) -> list[str]:
        assert hostname == "papers.example"
        return ["8.8.8.8"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    fixture = {
        "paper_id": "untrusted-id",
        "problem": "Reliable paper extraction",
        "contributions": ["Bounded pipeline"],
        "main_claims": [
            {
                "claim": "Extraction is predictable",
                "source_section": "Results",
                "supporting_text": "deterministic parser improves predictable extraction",
            }
        ],
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ReaderService(
            fetcher=DirectPDFFetcher(client=client, resolver=resolver),
            parser=PDFParser(),
            llm=MockLLMProvider(fixture),
            repository=repository,
            llm_provider_name="mock-fixture",
        ).read_url("https://papers.example/paper.pdf")

    assert result.card.paper_id == result.paper_id
    assert result.card.main_claims[0].source_section == "Results"
    persisted = repository.get_paper_card(result.paper_id)
    assert persisted == result.card


@pytest.mark.asyncio
async def test_reader_service_never_persists_a_card_when_mock_is_unavailable(
    repository: ResearchRepository,
) -> None:
    payload = _pdf_bytes()

    async def resolver(hostname: str) -> list[str]:
        return ["8.8.8.8"]

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=payload))
    ) as client:
        service = ReaderService(
            fetcher=DirectPDFFetcher(client=client, resolver=resolver),
            parser=PDFParser(),
            llm=MockLLMProvider(),
            repository=repository,
            llm_provider_name="mock",
        )
        with pytest.raises(LLMUnavailableError):
            await service.read_url("https://papers.example/paper.pdf")

    assert repository.get_papers_for_local_lexical_search("Useful Paper")
    stored = repository.get_papers_for_local_lexical_search("Useful Paper")[0]
    assert repository.get_paper_card(stored.id) is None
