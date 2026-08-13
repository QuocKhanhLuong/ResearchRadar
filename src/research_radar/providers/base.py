"""Provider-neutral scholarly discovery contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import httpx

from research_radar.models.paper import Paper

DEFAULT_HTTP_TIMEOUT_SECONDS = 20.0
DEFAULT_HTTP_CONNECT_TIMEOUT_SECONDS = 5.0


@runtime_checkable
class PaperProvider(Protocol):
    """A scholarly source that returns normalized papers for a text query."""

    name: str

    async def search(self, query: str, limit: int = 10) -> list[Paper]:
        """Return at most `limit` normalized papers or raise a provider error."""


def clamp_provider_limit(limit: int, *, maximum: int = 25) -> int:
    """Defensively bound requests independently of a user-facing service limit."""

    return max(1, min(limit, maximum))


def provider_timeout(timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS) -> httpx.Timeout:
    """Create an explicit bounded timeout for every scholarly HTTP request."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    return httpx.Timeout(
        timeout_seconds,
        connect=min(DEFAULT_HTTP_CONNECT_TIMEOUT_SECONDS, timeout_seconds),
    )


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Mapping[str, str | int] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: httpx.Timeout,
    max_attempts: int = 2,
) -> httpx.Response:
    """Issue a bounded GET with one retry for transient transport/rate failures.

    Request errors remain intentionally opaque to callers: adapters convert them
    into safe provider messages so URLs and request credentials never reach logs.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    for attempt in range(max_attempts):
        try:
            response = await client.get(url, params=params, headers=headers, timeout=timeout)
        except httpx.HTTPError:
            if attempt + 1 == max_attempts:
                raise
        else:
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return response
            if attempt + 1 == max_attempts:
                response.raise_for_status()
        await asyncio.sleep(0.2 * (attempt + 1))
    raise RuntimeError("Retry loop exited unexpectedly")  # pragma: no cover


def safe_provider_error(provider_name: str, error: Exception) -> str:
    """Describe a provider failure without leaking a URL, key, or response body."""

    if isinstance(error, httpx.TimeoutException):
        return f"{provider_name} request timed out."
    if isinstance(error, httpx.HTTPStatusError):
        return f"{provider_name} returned HTTP {error.response.status_code}."
    if isinstance(error, httpx.HTTPError):
        return f"{provider_name} could not be reached."
    return f"{provider_name} returned an invalid response."
