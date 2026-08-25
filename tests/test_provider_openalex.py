"""Focused tests for the hardened OpenAlex discovery provider."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from research_radar.errors import ProviderUnavailableError
from research_radar.providers.openalex import OpenAlexProvider


def _client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    """Create an async client wired to a mock transport."""

    return httpx.AsyncClient(transport=handler, timeout=httpx.Timeout(5.0))


def _records(prefix: str, count: int) -> list[dict[str, Any]]:
    """Build minimal valid OpenAlex work records with unique identifiers."""

    return [
        {
            "id": f"https://openalex.org/{prefix}{index}",
            "title": f"Paper {prefix}{index}",
            "doi": f"https://doi.org/10.1000/{prefix.lower()}{index}",
        }
        for index in range(1, count + 1)
    ]


async def test_normalizes_full_record_into_paper() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["search"] == "quantum sensing"
        assert "best_oa_location" in request.url.params["select"]
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "https://openalex.org/W123",
                        "title": "Discovery Paper",
                        "authorships": [
                            {"author": {"display_name": "Ada Lovelace"}},
                            {"author": {"display_name": "Grace Hopper"}},
                        ],
                        "publication_year": 2024,
                        "primary_location": {
                            "source": {"display_name": "Nature"},
                            "landing_page_url": "https://example.test/W123",
                        },
                        "doi": "https://doi.org/10.1000/discovery",
                        "ids": {"arxiv": "2401.00001v2"},
                        "cited_by_count": 42,
                    }
                ]
            },
        )

    async with _client(httpx.MockTransport(handler)) as client:
        papers = await OpenAlexProvider(client).search("quantum sensing")

    assert len(papers) == 1
    paper = papers[0]
    assert paper.title == "Discovery Paper"
    assert paper.doi == "10.1000/discovery"
    assert paper.external_ids["openalex"] == "W123"
    assert paper.external_ids["doi"] == "10.1000/discovery"
    assert paper.external_ids["arxiv"] == "2401.00001"
    assert paper.citation_count == 42
    assert paper.venue == "Nature"
    assert paper.publication_year == 2024
    assert paper.authors == ["Ada Lovelace", "Grace Hopper"]
    assert paper.source == "openalex"


async def test_reconstructs_inverted_abstract_in_word_order() -> None:
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "title": "Abstract Paper",
                "abstract_inverted_index": {"method": [2], "A": [0], "novel": [1]},
            }
        ]
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))

    async with _client(transport) as client:
        papers = await OpenAlexProvider(client).search("abstracts")

    assert papers[0].abstract == "A novel method"


async def test_paginates_until_limit_is_reached() -> None:
    seen_pages: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_pages.append(request.url.params["page"])
        assert request.url.params["per-page"] == "100"
        if request.url.params["page"] == "1":
            return httpx.Response(200, json={"results": _records("A", 100)})
        return httpx.Response(200, json={"results": _records("B", 3)})

    async with _client(httpx.MockTransport(handler)) as client:
        papers = await OpenAlexProvider(client).search("pagination", limit=103)

    assert len(papers) == 103
    assert seen_pages == ["1", "2"]
    assert papers[-1].external_ids["openalex"].startswith("B")


async def test_pagination_stops_when_page_is_short() -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json={"results": _records("S", 3)})

    async with _client(httpx.MockTransport(handler)) as client:
        papers = await OpenAlexProvider(client).search("short page", limit=50)

    assert request_count == 1
    assert len(papers) == 3


async def test_pagination_hard_cap_at_five_requests() -> None:
    pages_seen: list[str] = []
    sparse: dict[str, Any] = _records("C", 100)[0]
    filler = [{"title": "Missing identifier"}] * 99

    def handler(request: httpx.Request) -> httpx.Response:
        pages_seen.append(request.url.params["page"])
        return httpx.Response(200, json={"results": [sparse, *filler]})

    async with _client(httpx.MockTransport(handler)) as client:
        papers = await OpenAlexProvider(client).search("cap", limit=500)

    assert pages_seen == ["1", "2", "3", "4", "5"]
    assert len(papers) <= 200


async def test_pdf_url_preferred_from_best_oa_and_must_be_http() -> None:
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W10",
                "title": "Open Access Paper",
                "best_oa_location": {"pdf_url": "https://example.test/W10.pdf"},
                "primary_location": {"pdf_url": "https://mirror.example.test/W10.pdf"},
                "open_access": {"oa_url": "https://example.test/W10"},
            },
            {
                "id": "https://openalex.org/W11",
                "title": "Ftp Only Paper",
                "best_oa_location": {"pdf_url": "ftp://files.example.test/W11.pdf"},
                "open_access": {"oa_url": "https://example.test/W11"},
            },
            {
                "id": "https://openalex.org/W12",
                "title": "No Pdf Paper",
                "primary_location": {},
                "open_access": {},
            },
        ]
    }

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))

    async with _client(transport) as client:
        papers = await OpenAlexProvider(client).search("open access")

    by_id = {paper.id: paper for paper in papers}
    assert by_id["openalex:W10"].external_ids["pdf_url"] == "https://example.test/W10.pdf"
    assert by_id["openalex:W11"].external_ids["pdf_url"] == "https://example.test/W11"
    assert "pdf_url" not in by_id["openalex:W12"].external_ids
    assert by_id["openalex:W11"].url == "https://example.test/W11"


async def test_http_429_raises_rate_limit_error() -> None:
    async with _client(httpx.MockTransport(lambda request: httpx.Response(429))) as client:
        with pytest.raises(ProviderUnavailableError, match="rate limit"):
            await OpenAlexProvider(client).search("throttled")


async def test_http_500_raises_provider_unavailable() -> None:
    async with _client(httpx.MockTransport(lambda request: httpx.Response(500))) as client:
        with pytest.raises(ProviderUnavailableError, match="HTTP 500"):
            await OpenAlexProvider(client).search("broken")


async def test_timeout_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    async with _client(httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderUnavailableError, match="timed out"):
            await OpenAlexProvider(client).search("slow")


async def test_malformed_json_raises_provider_unavailable() -> None:
    async with _client(
        httpx.MockTransport(lambda request: httpx.Response(200, content=b"<not-json>"))
    ) as client:
        with pytest.raises(ProviderUnavailableError):
            await OpenAlexProvider(client).search("garbage")


async def test_non_list_results_raise_provider_unavailable() -> None:
    async with _client(
        httpx.MockTransport(lambda request: httpx.Response(200, json={"results": {"oops": True}}))
    ) as client:
        with pytest.raises(ProviderUnavailableError, match="unexpected response shape"):
            await OpenAlexProvider(client).search("shapeless")


async def test_records_missing_identity_are_skipped() -> None:
    payload = {
        "results": [
            {"id": "https://openalex.org/W40", "cited_by_count": 1},
            {"title": "No Identifier Paper"},
            {"id": "https://openalex.org/W41", "title": "Good Paper"},
        ]
    }

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))

    async with _client(transport) as client:
        papers = await OpenAlexProvider(client).search("sparse")

    assert [paper.id for paper in papers] == ["openalex:W41"]


async def test_api_key_stays_out_of_urls_and_errors() -> None:
    secret = "super-secret-key"
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        assert "api_key" not in request.url.params
        assert request.headers["authorization"] == f"Bearer {secret}"
        assert "researcher@example.test" in request.headers["user-agent"]
        return httpx.Response(500)

    async with _client(httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderUnavailableError) as error:
            await OpenAlexProvider(client, email="researcher@example.test", api_key=secret).search(
                "credentials"
            )

    assert seen_urls
    assert all(secret not in url for url in seen_urls)
    assert secret not in str(error.value)
