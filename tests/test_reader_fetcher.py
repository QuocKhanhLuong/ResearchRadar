from __future__ import annotations

from collections.abc import Sequence

import httpx
import pytest

from research_radar.reader.fetcher import (
    DirectPDFFetcher,
    PaperDownloadError,
    PDFDownloadLimits,
)


async def _public_resolver(hostname: str) -> Sequence[str]:
    assert hostname in {"papers.example", "cdn.example"}
    return ["8.8.8.8"]


@pytest.mark.asyncio
async def test_fetcher_streams_a_public_direct_pdf_with_injected_resolver() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200,
            headers={"content-length": "15"},
            content=b"%PDF-1.7\ncontent",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetcher = DirectPDFFetcher(client=client, resolver=_public_resolver)
        fetched = await fetcher.fetch("https://papers.example/article.pdf")

    assert fetched.content == b"%PDF-1.7\ncontent"
    assert fetched.source_url == "https://papers.example/article.pdf"
    assert requests == ["https://papers.example/article.pdf"]


@pytest.mark.asyncio
async def test_fetcher_validates_each_redirect_destination_before_connecting() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://cdn.example/final"})
        return httpx.Response(200, content=b"%PDF-1.7\nfinal")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetcher = DirectPDFFetcher(client=client, resolver=_public_resolver)
        fetched = await fetcher.fetch("https://papers.example/start")

    assert fetched.source_url == "https://cdn.example/final"
    assert requests == ["https://papers.example/start", "https://cdn.example/final"]


@pytest.mark.asyncio
async def test_fetcher_rejects_private_and_non_http_destinations_before_request() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, content=b"%PDF-1.7")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetcher = DirectPDFFetcher(client=client, resolver=_public_resolver)
        with pytest.raises(PaperDownloadError, match="non-public destination"):
            await fetcher.fetch("http://127.0.0.1/private.pdf")
        with pytest.raises(PaperDownloadError, match=r"HTTP\(S\)"):
            await fetcher.fetch("file:///tmp/private.pdf")

    assert not called


@pytest.mark.asyncio
async def test_fetcher_rejects_a_private_address_returned_by_the_injected_resolver() -> None:
    called = False

    async def private_resolver(hostname: str) -> Sequence[str]:
        assert hostname == "intranet.example"
        return ["10.0.0.4"]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, content=b"%PDF-1.7")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetcher = DirectPDFFetcher(client=client, resolver=private_resolver)
        with pytest.raises(PaperDownloadError, match="non-public destination"):
            await fetcher.fetch("https://intranet.example/paper.pdf")

    assert not called


@pytest.mark.asyncio
async def test_fetcher_rejects_a_redirect_to_a_private_destination() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private.pdf"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetcher = DirectPDFFetcher(client=client, resolver=_public_resolver)
        with pytest.raises(PaperDownloadError, match="non-public destination"):
            await fetcher.fetch("https://papers.example/start")


@pytest.mark.asyncio
async def test_fetcher_enforces_its_redirect_cap() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/next"})

    limits = PDFDownloadLimits(max_redirects=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetcher = DirectPDFFetcher(limits=limits, client=client, resolver=_public_resolver)
        with pytest.raises(PaperDownloadError, match="1-redirect limit"):
            await fetcher.fetch("https://papers.example/start")


@pytest.mark.asyncio
async def test_fetcher_enforces_declared_and_streamed_byte_limits() -> None:
    def declared_too_large(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-length": "999"}, content=b"%PDF-1.7")

    limits = PDFDownloadLimits(max_bytes=32)
    async with httpx.AsyncClient(transport=httpx.MockTransport(declared_too_large)) as client:
        fetcher = DirectPDFFetcher(limits=limits, client=client, resolver=_public_resolver)
        with pytest.raises(PaperDownloadError, match="byte download limit"):
            await fetcher.fetch("https://papers.example/large.pdf")

    def streamed_too_large(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"%PDF-1.7" + b"x" * 64)

    async with httpx.AsyncClient(transport=httpx.MockTransport(streamed_too_large)) as client:
        fetcher = DirectPDFFetcher(limits=limits, client=client, resolver=_public_resolver)
        with pytest.raises(PaperDownloadError, match="byte download limit"):
            await fetcher.fetch("https://papers.example/streamed.pdf")


@pytest.mark.asyncio
async def test_fetcher_rejects_non_pdf_payloads_and_maps_timeouts() -> None:
    def html_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not a PDF</html>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(html_handler)) as client:
        fetcher = DirectPDFFetcher(client=client, resolver=_public_resolver)
        with pytest.raises(PaperDownloadError, match="not a PDF"):
            await fetcher.fetch("https://papers.example/paper")

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler)) as client:
        fetcher = DirectPDFFetcher(client=client, resolver=_public_resolver)
        with pytest.raises(PaperDownloadError, match="timed out"):
            await fetcher.fetch("https://papers.example/slow.pdf")
