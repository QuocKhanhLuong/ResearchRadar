from __future__ import annotations

import asyncio

import httpx
import pytest

from research_radar.errors import ProviderUnavailableError
from research_radar.providers.arxiv import ArxivProvider
from research_radar.providers.openalex import OpenAlexProvider, reconstruct_inverted_abstract
from research_radar.providers.semantic_scholar import SemanticScholarProvider


def _client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler, timeout=httpx.Timeout(5.0))


@pytest.mark.asyncio
async def test_openalex_normalizes_inverted_abstract_and_sparse_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["search"] == "medical imaging"
        assert request.url.params["per-page"] == "5"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "title": " A robust method ",
                        "abstract_inverted_index": {"robust": [1], "A": [0], "method": [2]},
                        "authorships": [{"author": {"display_name": "Ada"}}],
                        "publication_year": 2025,
                        "primary_location": {
                            "source": {"display_name": "Journal"},
                            "landing_page_url": "https://example.test/paper",
                        },
                        "doi": "https://doi.org/10.1000/Example",
                        "ids": {"pmid": "99", "ignored": {"nested": True}},
                        "cited_by_count": 4,
                    },
                    {"id": "https://openalex.org/W2"},
                ]
            },
        )

    async with _client(httpx.MockTransport(handler)) as client:
        papers = await OpenAlexProvider(client).search("medical imaging", 5)

    assert len(papers) == 1
    paper = papers[0]
    assert paper.id == "openalex:W1"
    assert paper.abstract == "A robust method"
    assert paper.doi == "10.1000/example"
    assert paper.external_ids == {"pmid": "99", "openalex": "W1", "doi": "10.1000/example"}


@pytest.mark.asyncio
async def test_openalex_keeps_api_key_out_of_request_urls_and_failure_messages() -> None:
    secret = "review-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        assert "api_key" not in request.url.params
        assert request.headers["authorization"] == f"Bearer {secret}"
        return httpx.Response(401, request=request)

    async with _client(httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderUnavailableError) as error:
            await OpenAlexProvider(client, api_key=secret).search("query")

    assert secret not in str(error.value)
    assert "HTTP 401" in str(error.value)


def test_openalex_abstract_reconstruction_tolerates_bad_shape() -> None:
    assert reconstruct_inverted_abstract({"a": [1, "bad"], "start": [0]}) == "start a"
    assert reconstruct_inverted_abstract([]) is None


@pytest.mark.asyncio
async def test_openalex_wraps_http_failure() -> None:
    async with _client(httpx.MockTransport(lambda request: httpx.Response(429))) as client:
        with pytest.raises(ProviderUnavailableError, match="OpenAlex"):
            await OpenAlexProvider(client).search("query")


ARXIV_XML = b"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns='http://www.w3.org/2005/Atom' xmlns:arxiv='http://arxiv.org/schemas/atom'>
  <entry>
    <id>http://arxiv.org/abs/2401.01234v2</id><updated>2024-01-02T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published><title>  Useful\nPaper </title>
    <summary> A useful summary. </summary><author><name>Ada</name></author>
    <arxiv:doi>10.1000/ARXIV</arxiv:doi><arxiv:journal_ref>Venue</arxiv:journal_ref>
  </entry>
</feed>"""


@pytest.mark.asyncio
async def test_arxiv_parses_atom_and_strips_identity_version() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["search_query"] == "all:vision"
        return httpx.Response(200, content=ARXIV_XML)

    async with _client(httpx.MockTransport(handler)) as client:
        papers = await ArxivProvider(client, minimum_interval_seconds=0).search("vision")

    assert papers[0].id == "arxiv:2401.01234"
    assert papers[0].title == "Useful Paper"
    assert papers[0].doi == "10.1000/arxiv"


def test_arxiv_legacy_identity_preserves_its_archive_prefix() -> None:
    from research_radar.providers.normalization import normalize_arxiv_id

    assert normalize_arxiv_id("https://arxiv.org/abs/hep-th/9901001v2") == "hep-th/9901001"
    assert normalize_arxiv_id("https://arxiv.org/abs/math/9901001v2") == "math/9901001"


@pytest.mark.asyncio
async def test_arxiv_serializes_requests_when_a_delay_is_configured() -> None:
    calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(asyncio.get_running_loop().time())
        return httpx.Response(200, content=b"<feed xmlns='http://www.w3.org/2005/Atom'/>")

    async with _client(httpx.MockTransport(handler)) as client:
        provider = ArxivProvider(client, minimum_interval_seconds=0.01)
        await asyncio.gather(provider.search("one"), provider.search("two"))

    assert len(calls) == 2
    assert calls[1] - calls[0] >= 0.008


@pytest.mark.asyncio
async def test_semantic_scholar_uses_explicit_fields_and_optional_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "paperId" in request.url.params["fields"]
        assert request.headers["x-api-key"] == "test-key"
        assert request.url.params["query"] == "state space"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "paperId": "S1",
                        "title": "State-space paper",
                        "abstract": None,
                        "authors": [{"name": "Grace"}],
                        "year": 2023,
                        "venue": "Conf",
                        "url": "https://example.test/S1",
                        "citationCount": 10,
                        "externalIds": {"DOI": "10.2/X", "ArXiv": "2401.2v3"},
                    }
                ]
            },
        )

    async with _client(httpx.MockTransport(handler)) as client:
        papers = await SemanticScholarProvider(client, api_key="test-key").search("state-space")

    assert papers[0].id == "semantic_scholar:S1"
    assert papers[0].external_ids["arxiv"] == "2401.2"
    assert papers[0].doi == "10.2/x"


@pytest.mark.asyncio
async def test_semantic_scholar_rejects_malformed_top_level_data() -> None:
    async with _client(httpx.MockTransport(lambda request: httpx.Response(200, json={}))) as client:
        with pytest.raises(ProviderUnavailableError, match="unexpected"):
            await SemanticScholarProvider(client).search("query")
