"""OpenAlex Works adapter with provider-local response handling."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

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


class OpenAlexProvider:
    """Search the OpenAlex Works endpoint and return normalized records."""

    name = "openalex"
    base_url = "https://api.openalex.org/works"
    _select = (
        "id,title,abstract_inverted_index,authorships,publication_year,"
        "primary_location,open_access,doi,ids,cited_by_count"
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
        """Request one bounded page and adapt usable records into ``Paper`` objects."""

        params: dict[str, str | int] = {
            "search": query,
            "per-page": clamp_provider_limit(limit, maximum=100),
            "select": self._select,
        }
        headers = {"User-Agent": f"ResearchRadar/0.1 ({self._email})"} if self._email else {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = await get_with_retry(
                self._client,
                self.base_url,
                params=params,
                headers=headers,
                timeout=self._timeout,
            )
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise ProviderUnavailableError(safe_provider_error("OpenAlex", error)) from error

        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise ProviderUnavailableError("OpenAlex search returned an unexpected response shape.")

        papers: list[Paper] = []
        for record in results:
            if not isinstance(record, Mapping):
                continue
            paper = self._paper_from_record(record)
            if paper is not None:
                papers.append(paper)
        logger.info("OpenAlex returned %d normalized paper(s).", len(papers))
        return papers

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
