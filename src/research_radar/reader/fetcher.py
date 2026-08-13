"""Safe, bounded retrieval of direct public PDF URLs.

This module is intentionally limited to fetching bytes. PDF parsing remains in
``reader.parser`` and a future reader service composes the two boundaries.
"""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias
from urllib.parse import urljoin, urlsplit

import httpx

from research_radar.errors import PaperParseError

DEFAULT_MAX_PDF_DOWNLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_REDIRECTS = 3
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 30.0
DEFAULT_DOWNLOAD_CHUNK_SIZE = 64 * 1024

IPAddress: TypeAlias = ipaddress.IPv4Address | ipaddress.IPv6Address
ResolverResult: TypeAlias = Sequence[str | IPAddress]
AddressResolver: TypeAlias = Callable[[str], ResolverResult | Awaitable[ResolverResult]]


class PaperDownloadError(PaperParseError):
    """Raised when a direct PDF cannot be safely downloaded."""


@dataclass(frozen=True, slots=True)
class PDFDownloadLimits:
    """Resource and redirect limits for one direct-PDF download."""

    max_bytes: int = DEFAULT_MAX_PDF_DOWNLOAD_BYTES
    max_redirects: int = DEFAULT_MAX_REDIRECTS
    timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS
    chunk_size: int = DEFAULT_DOWNLOAD_CHUNK_SIZE

    def __post_init__(self) -> None:
        if self.max_bytes < 1:
            raise ValueError("max_bytes must be at least 1")
        if self.max_redirects < 0:
            raise ValueError("max_redirects cannot be negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")


@dataclass(frozen=True, slots=True)
class FetchedPDF:
    """Validated direct-PDF bytes and the final URL after safe redirects."""

    content: bytes
    source_url: str


class DirectPDFFetcher:
    """Fetch a direct PDF without following unvalidated redirects.

    Every hop is parsed, resolved, and rejected before connection if it targets
    a local, private, reserved, multicast, or otherwise non-public address.
    ``resolver`` is injectable so tests do not make DNS requests.
    """

    def __init__(
        self,
        *,
        limits: PDFDownloadLimits | None = None,
        client: httpx.AsyncClient | None = None,
        resolver: AddressResolver | None = None,
    ) -> None:
        self.limits = limits or PDFDownloadLimits()
        self._timeout = httpx.Timeout(self.limits.timeout_seconds)
        self._client = client or httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=False,
        )
        self._owns_client = client is None
        self._resolver = resolver or resolve_host_addresses

    async def fetch(self, url: str) -> FetchedPDF:
        """Fetch a direct public PDF, enforcing bounds before parsing it."""

        current_url = _validate_url(url)
        redirects = 0
        visited_urls: set[str] = set()

        while True:
            if current_url in visited_urls:
                raise PaperDownloadError("The PDF URL redirects in a loop.")
            visited_urls.add(current_url)
            await self._ensure_public_destination(current_url)

            try:
                async with self._client.stream(
                    "GET",
                    current_url,
                    headers={
                        "Accept": "application/pdf, application/octet-stream;q=0.9, */*;q=0.1"
                    },
                    follow_redirects=False,
                    timeout=self._timeout,
                ) as response:
                    if response.status_code in _REDIRECT_STATUS_CODES:
                        current_url = self._redirect_target(response, current_url, redirects)
                        redirects += 1
                        continue

                    if not response.is_success:
                        raise PaperDownloadError(
                            f"The PDF server returned HTTP {response.status_code}."
                        )

                    self._ensure_declared_size(response)
                    payload = await self._read_bounded_response(response)
            except PaperDownloadError:
                raise
            except httpx.TimeoutException as exc:
                raise PaperDownloadError("The PDF download timed out.") from exc
            except httpx.HTTPError as exc:
                raise PaperDownloadError("The PDF download failed due to a network error.") from exc

            if not payload.startswith(b"%PDF-"):
                raise PaperDownloadError("The downloaded file is not a PDF.")
            return FetchedPDF(content=payload, source_url=current_url)

    async def aclose(self) -> None:
        """Close only an HTTP client constructed by this fetcher."""

        if self._owns_client:
            await self._client.aclose()

    async def _ensure_public_destination(self, url: str) -> None:
        hostname = _hostname_from_url(url)
        normalized_hostname = hostname.rstrip(".").casefold()
        if normalized_hostname == "localhost" or normalized_hostname.endswith(".localhost"):
            raise PaperDownloadError("Local PDF destinations are not allowed.")
        if normalized_hostname.endswith(".local"):
            raise PaperDownloadError("Local PDF destinations are not allowed.")

        try:
            addresses: ResolverResult = [ipaddress.ip_address(hostname)]
        except ValueError:
            try:
                resolution = self._resolver(hostname)
                addresses = await resolution if inspect.isawaitable(resolution) else resolution
            except (OSError, socket.gaierror) as exc:
                raise PaperDownloadError("The PDF host could not be resolved.") from exc
            except PaperDownloadError:
                raise
            except Exception as exc:
                raise PaperDownloadError("The PDF host could not be resolved.") from exc

        if not addresses:
            raise PaperDownloadError("The PDF host did not resolve to an address.")

        for address in addresses:
            try:
                parsed_address = ipaddress.ip_address(address)
            except (TypeError, ValueError) as exc:
                raise PaperDownloadError("The PDF host resolved to an invalid address.") from exc
            if not parsed_address.is_global:
                raise PaperDownloadError("The PDF URL resolves to a non-public destination.")

    def _redirect_target(self, response: httpx.Response, current_url: str, redirects: int) -> str:
        if redirects >= self.limits.max_redirects:
            raise PaperDownloadError(
                f"The PDF URL exceeded the {self.limits.max_redirects}-redirect limit."
            )
        location = response.headers.get("location")
        if not location:
            raise PaperDownloadError("The PDF server sent a redirect without a destination.")
        return _validate_url(urljoin(current_url, location))

    def _ensure_declared_size(self, response: httpx.Response) -> None:
        content_length = response.headers.get("content-length")
        if content_length is None:
            return
        try:
            declared_size = int(content_length)
        except ValueError:
            return
        if declared_size > self.limits.max_bytes:
            raise PaperDownloadError(
                "The PDF exceeds the " f"{self.limits.max_bytes}-byte download limit."
            )

    async def _read_bounded_response(self, response: httpx.Response) -> bytes:
        payload = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=self.limits.chunk_size):
            if len(payload) + len(chunk) > self.limits.max_bytes:
                raise PaperDownloadError(
                    "The PDF exceeds the " f"{self.limits.max_bytes}-byte download limit."
                )
            payload.extend(chunk)
        if not payload:
            raise PaperDownloadError("The PDF download was empty.")
        return bytes(payload)


async def fetch_pdf(
    url: str,
    *,
    limits: PDFDownloadLimits | None = None,
    client: httpx.AsyncClient | None = None,
    resolver: AddressResolver | None = None,
) -> FetchedPDF:
    """Fetch one direct PDF and close an internally created client afterward."""

    fetcher = DirectPDFFetcher(limits=limits, client=client, resolver=resolver)
    try:
        return await fetcher.fetch(url)
    finally:
        await fetcher.aclose()


async def resolve_host_addresses(hostname: str) -> list[str]:
    """Resolve a hostname asynchronously through the event loop's DNS resolver."""

    loop = asyncio.get_running_loop()
    results = await loop.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    return sorted({result[4][0] for result in results})


_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})


def _validate_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError as exc:
        raise PaperDownloadError("The PDF URL has an invalid port.") from exc

    if parsed.scheme.casefold() not in {"http", "https"}:
        raise PaperDownloadError("Only public HTTP(S) PDF URLs are supported.")
    if not parsed.hostname:
        raise PaperDownloadError("The PDF URL must include a host.")
    if parsed.username is not None or parsed.password is not None:
        raise PaperDownloadError("PDF URLs with embedded credentials are not supported.")
    return parsed.geturl()


def _hostname_from_url(url: str) -> str:
    hostname = urlsplit(url).hostname
    if hostname is None:  # Defensive: URL validity was checked before this call.
        raise PaperDownloadError("The PDF URL must include a host.")
    return hostname
