"""OpenAlex Works adapter with provider-local response handling."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx

from research_radar.errors import ProviderUnavailableError
from research_radar.models.paper import Paper
from research_radar.providers.base import (
    DEFAULT_HTTP_TIMEOUT_SECONDS,
    clamp_provider_limit,
    get_with_retry,
    provider_timeout,
    safe_provider_error,
)
from research_radar.providers.normalization import (
    integer_or_none,
    known_external_ids,
    normalize_doi,
    string_or_none,
)

logger = logging.getLogger(__name__)

_PAGE_SIZE = 100
_MAX_PAGES = 5
_MAX_RESULT_LIMIT = 200


class OpenAlexProvider:
    """Search the OpenAlex Works endpoint across bounded pages."""

    name = "openalex"
    base_url = "https://api.openalex.org/works"
    _select = (
        "id,title,abstract_inverted_index,authorships,publication_year,"
        "primary_location,best_oa_location,open_access,doi,ids,cited_by_count"
    )

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        email: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    ) -> None:
        self._client = client
        self._email = email
        self._api_key = api_key
        self._timeout = provider_timeout(timeout_seconds)

    async def search(self, query: str, limit: int = 10) -> list[Paper]:
        """Page through bounded OpenAlex results and adapt them into papers."""

        target = clamp_provider_limit(limit, maximum=_MAX_RESULT_LIMIT)
        per_page = min(target, _PAGE_SIZE)
        headers: dict[str, str] = {}
        if self._email:
            # Polite-pool convention: the email identifies us to OpenAlex. It is a
            # non-secret contact address, but it still never gets logged.
            headers["User-Agent"] = f"ResearchRadar/0.1 ({self._email})"
        if self._api_key:
            # Credential safety: the API key travels ONLY in the Authorization
            # header. It must never be placed in query params, written to a log
            # line, or embedded in an exception message.
            headers["Authorization"] = f"Bearer {self._api_key}"

        papers: list[Paper] = []
        requests_used = 0
        for page in range(1, _MAX_PAGES + 1):
            if len(papers) >= target:
                break
            params: dict[str, str | int] = {
                "search": query,
                "per-page": per_page,
                "page": page,
                "select": self._select,
            }
            results = await self._fetch_results(params, headers)
            requests_used += 1
            if not results:
                break
            for record in results:
                if not isinstance(record, Mapping):
                    continue
                paper = self._paper_from_record(record)
                if paper is not None:
                    papers.append(paper)
                    if len(papers) >= target:
                        break
            if len(results) < per_page:
                break
        logger.info(
            "OpenAlex returned %d normalized paper(s) from %d request(s).",
            len(papers),
            requests_used,
        )
        return papers[:target]

    async def _fetch_results(
        self,
        params: Mapping[str, str | int],
        headers: Mapping[str, str],
    ) -> list[Any]:
        """Fetch one results page, converting every failure into a provider error."""

        try:
            response = await get_with_retry(
                self._client,
                self.base_url,
                params=params,
                headers=headers,
                timeout=self._timeout,
            )
            payload = response.json()
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 429:
                raise ProviderUnavailableError(
                    "OpenAlex rate limit reached; try again shortly."
                ) from error
            raise ProviderUnavailableError(safe_provider_error("OpenAlex", error)) from error
        except (httpx.HTTPError, ValueError) as error:
            raise ProviderUnavailableError(safe_provider_error("OpenAlex", error)) from error

        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise ProviderUnavailableError("OpenAlex search returned an unexpected response shape.")
        return results

    @staticmethod
    def _first_http_url(values: list[object]) -> str | None:
        """Return the first non-empty http/https URL among candidate values."""

        for value in values:
            url = string_or_none(value)
            if url is None:
                continue
            if urlsplit(url).scheme.casefold() in {"http", "https"}:
                return url
        return None

    @classmethod
    def _pdf_url(cls, record: Mapping[str, Any]) -> str | None:
        """Resolve a PDF URL preferring best_oa_location, then primary, then OA URL."""

        best_oa = record.get("best_oa_location")
        primary_location = record.get("primary_location")
        open_access = record.get("open_access")
        return cls._first_http_url(
            [
                best_oa.get("pdf_url") if isinstance(best_oa, Mapping) else None,
                primary_location.get("pdf_url") if isinstance(primary_location, Mapping) else None,
                open_access.get("oa_url") if isinstance(open_access, Mapping) else None,
            ]
        )

    @classmethod
    def _paper_from_record(cls, record: Mapping[str, Any]) -> Paper | None:
        raw_id = string_or_none(record.get("id"))
        title = string_or_none(record.get("title"))
        if raw_id is None or title is None:
            return None
        openalex_id = raw_id.rstrip("/").rsplit("/", maxsplit=1)[-1]
        if not openalex_id:
            return None

        raw_ids = record.get("ids")
        ids = known_external_ids(raw_ids if isinstance(raw_ids, Mapping) else None)
        ids["openalex"] = openalex_id
        doi = normalize_doi(record.get("doi")) or ids.get("doi")
        if doi:
            ids["doi"] = doi
        pdf_url = cls._pdf_url(record)
        if pdf_url:
            ids["pdf_url"] = pdf_url

        primary_location = record.get("primary_location")
        location = primary_location if isinstance(primary_location, Mapping) else {}
        source = location.get("source")
        venue = string_or_none(source.get("display_name")) if isinstance(source, Mapping) else None
        canonical_url = string_or_none(location.get("landing_page_url"))
        if canonical_url is None:
            open_access = record.get("open_access")
            if isinstance(open_access, Mapping):
                canonical_url = string_or_none(open_access.get("oa_url"))
        if canonical_url is None and doi:
            canonical_url = f"https://doi.org/{doi}"
        if canonical_url is None:
            canonical_url = raw_id

        authorships = record.get("authorships")
        authors: list[str] = []
        if isinstance(authorships, list):
            for authorship in authorships:
                if not isinstance(authorship, Mapping):
                    continue
                author = authorship.get("author")
                if isinstance(author, Mapping):
                    name = string_or_none(author.get("display_name"))
                    if name:
                        authors.append(name)

        return Paper(
            id=f"openalex:{openalex_id}",
            title=title,
            abstract=reconstruct_inverted_abstract(record.get("abstract_inverted_index")),
            authors=authors,
            publication_year=integer_or_none(record.get("publication_year")),
            venue=venue,
            doi=doi,
            url=canonical_url,
            citation_count=integer_or_none(record.get("cited_by_count")),
            source=cls.name,
            external_ids=ids,
        )


def reconstruct_inverted_abstract(value: object) -> str | None:
    """Reconstruct OpenAlex's token-to-position inverted abstract representation."""

    if not isinstance(value, Mapping):
        return None
    positions: dict[int, str] = {}
    for token, raw_positions in value.items():
        if not isinstance(token, str) or not isinstance(raw_positions, list):
            continue
        for position in raw_positions:
            if isinstance(position, int) and position >= 0:
                positions[position] = token
    if not positions:
        return None
    return " ".join(positions[position] for position in sorted(positions))
