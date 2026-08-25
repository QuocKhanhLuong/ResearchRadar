"""Unit tests for Semantic Scholar enrichment endpoints and arXiv identity."""

from __future__ import annotations

import xml.etree.ElementTree as element_tree
from typing import Any

import httpx
import pytest

from research_radar.errors import ProviderUnavailableError
from research_radar.models.paper import Paper
from research_radar.providers.arxiv import ArxivProvider, _paper_from_entry
from research_radar.providers.semantic_scholar import SemanticScholarProvider

ATOM_ENTRY = "{http://www.w3.org/2005/Atom}entry"

S2_RECORD: dict[str, Any] = {
    "paperId": "S1",
    "title": "A shared scholarly work",
    "abstract": "Shared abstract.",
    "authors": [{"name": "Ada"}, {"name": ""}, {"name": "Grace"}],
    "year": 2021,
    "venue": "Venue",
    "url": "https://example.test/S1",
    "citationCount": 7,
    "externalIds": {"DOI": "10.1000/SHARED", "ArXiv": "2101.00001v3", "PMID": "42"},
}

ARXIV_XML = b"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns='http://www.w3.org/2005/Atom' xmlns:arxiv='http://arxiv.org/schemas/atom'>
  <entry>
    <id>http://arxiv.org/abs/2101.00001v3</id>
    <published>2021-01-01T00:00:00Z</published>
    <title>A shared scholarly work</title>
    <summary>Shared abstract.</summary>
    <author><name>Ada</name></author>
    <arxiv:doi>https://doi.org/10.1000/SHARED</arxiv:doi>
  </entry>
</feed>"""

ARXIV_XML_WITHOUT_DOI = b"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns='http://www.w3.org/2005/Atom'>
  <entry>
    <id>http://arxiv.org/abs/2101.00001v3</id>
    <title>A shared scholarly work</title>
    <author><name>Ada</name></author>
  </entry>
</feed>"""


def _client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler, timeout=httpx.Timeout(5.0))


def _ok_json(payload: Any) -> httpx.Response:
    return httpx.Response(200, json=payload)


async def _arxiv_papers(content: bytes) -> list[Paper]:
    async with _client(
        httpx.MockTransport(lambda request: httpx.Response(200, content=content))
    ) as client:
        return await ArxivProvider(client, minimum_interval_seconds=0).search("shared")


def _single_arxiv_entry(xml: bytes) -> element_tree.Element:
    root = element_tree.fromstring(xml)
    entries = root.findall(ATOM_ENTRY)
    assert len(entries) == 1
    return entries[0]


async def test_s2_search_normalizes_record_with_complete_identifiers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["limit"] == "10"
        assert request.url.params["fields"].startswith("paperId")
        return _ok_json({"data": [dict(S2_RECORD)]})

    async with _client(httpx.MockTransport(handler)) as client:
        papers = await SemanticScholarProvider(client).search("shared-work")

    assert len(papers) == 1
    paper = papers[0]
    assert paper.source == "semantic_scholar"
    assert paper.id == "semantic_scholar:S1"
    assert paper.external_ids["s2"] == "S1"
    assert paper.external_ids["semantic_scholar"] == "S1"
    assert paper.external_ids["doi"] == "10.1000/shared"
    assert paper.external_ids["arxiv"] == "2101.00001"
    assert paper.external_ids["pmid"] == "42"
    assert paper.doi == "10.1000/shared"
    assert paper.authors == ["Ada", "Grace"]


async def test_s2_references_reads_cited_paper_records() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _ok_json({"data": [{"citedPaper": dict(S2_RECORD)}]})

    async with _client(httpx.MockTransport(handler)) as client:
        papers = await SemanticScholarProvider(client).references("S1", limit=5)

    assert len(requests) == 1
    assert "/paper/S1/references" in str(requests[0].url)
    assert requests[0].url.params["limit"] == "5"
    assert [p.id for p in papers] == ["semantic_scholar:S1"]
    assert papers[0].source == "semantic_scholar"


async def test_s2_citations_reads_citing_paper_records() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _ok_json({"data": [{"citingPaper": dict(S2_RECORD)}]})

    async with _client(httpx.MockTransport(handler)) as client:
        papers = await SemanticScholarProvider(client).citations("S1")

    assert len(requests) == 1
    assert "/paper/S1/citations" in str(requests[0].url)
    assert [p.id for p in papers] == ["semantic_scholar:S1"]
    assert papers[0].source == "semantic_scholar"


async def test_s2_recommendations_reads_recommended_paper_records() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _ok_json({"recommendedPapers": [dict(S2_RECORD)]})

    async with _client(httpx.MockTransport(handler)) as client:
        papers = await SemanticScholarProvider(client).recommendations("S1")

    assert len(requests) == 1
    assert "/recommendations/v1/papers/forpaper/S1" in str(requests[0].url)
    assert [p.id for p in papers] == ["semantic_scholar:S1"]
    assert papers[0].source == "semantic_scholar"


async def test_s2_identifier_keeps_prefix_colon_and_encodes_slash() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "DOI:10.1%2Fabc" in str(request.url)
        return _ok_json({"recommendedPapers": []})

    async with _client(httpx.MockTransport(handler)) as client:
        papers = await SemanticScholarProvider(client).recommendations(" DOI:10.1/abc ")

    assert papers == []


async def test_s2_blank_identifier_raises_before_any_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _ok_json({})

    async with _client(httpx.MockTransport(handler)) as client:
        provider = SemanticScholarProvider(client)
        for method in (provider.references, provider.citations, provider.recommendations):
            with pytest.raises(ValueError):
                await method("   ")

    assert requests == []


@pytest.mark.parametrize("status", [429, 500])
async def test_s2_http_failures_raise_provider_unavailable(status: int) -> None:
    async with _client(
        httpx.MockTransport(lambda request, status=status: httpx.Response(status))
    ) as client:
        with pytest.raises(ProviderUnavailableError, match="Semantic Scholar"):
            await SemanticScholarProvider(client).references("S1")


async def test_s2_api_key_travels_in_header_never_in_url() -> None:
    secret = "secret-s2-key"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == secret
        assert secret not in str(request.url)
        return _ok_json({"recommendedPapers": [dict(S2_RECORD)]})

    async with _client(httpx.MockTransport(handler)) as client:
        provider = SemanticScholarProvider(client, api_key=secret)
        papers = await provider.recommendations("CorpusId:123")

    assert [p.id for p in papers] == ["semantic_scholar:S1"]


async def test_arxiv_search_strips_version_from_external_id() -> None:
    papers = await _arxiv_papers(ARXIV_XML)

    assert papers[0].id == "arxiv:2101.00001"
    assert papers[0].source == "arxiv"
    assert papers[0].external_ids["arxiv"] == "2101.00001"


async def test_arxiv_sets_pdf_url_without_version_suffix() -> None:
    papers = await _arxiv_papers(ARXIV_XML)

    assert papers[0].external_ids["pdf_url"] == "https://arxiv.org/pdf/2101.00001"


async def test_arxiv_entry_doi_is_normalized_into_paper_and_ids() -> None:
    papers = await _arxiv_papers(ARXIV_XML)

    assert papers[0].doi == "10.1000/shared"
    assert papers[0].external_ids["doi"] == "10.1000/shared"


async def test_arxiv_entry_without_doi_leaves_doi_unset() -> None:
    papers = await _arxiv_papers(ARXIV_XML_WITHOUT_DOI)

    assert papers[0].doi is None
    assert "doi" not in papers[0].external_ids


async def test_arxiv_malformed_xml_raises_provider_unavailable() -> None:
    with pytest.raises(ProviderUnavailableError, match="arXiv"):
        await _arxiv_papers(b"<feed xmlns='http://www.w3.org/2005/Atom'><entry>")


def test_cross_provider_identifiers_align_for_the_same_work() -> None:
    s2_paper = SemanticScholarProvider._paper_from_record(S2_RECORD)
    arxiv_paper = _paper_from_entry(_single_arxiv_entry(ARXIV_XML))

    assert s2_paper is not None and arxiv_paper is not None
    assert s2_paper.source != arxiv_paper.source
    assert s2_paper.id != arxiv_paper.id
    assert s2_paper.external_ids["arxiv"] == arxiv_paper.external_ids["arxiv"]
    assert s2_paper.external_ids["arxiv"] == "2101.00001"
    assert s2_paper.doi is not None and s2_paper.doi == arxiv_paper.doi
