"""Semantic Scholar Graph API adapter."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

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

_GRAPH_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"
_RECOMMENDATIONS_URL = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
_UNEXPECTED_SHAPE_MESSAGE = "Semantic Scholar returned an unexpected response shape."


class SemanticScholarProvider:
    """Use explicit Graph API fields and return provider-neutral records."""

    name = "semantic_scholar"
    base_url = f"{_GRAPH_PAPER_URL}/search"
    fields = "paperId,externalIds,title,abstract,authors,year,venue,url,citationCount"

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str | None = None,
        timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._timeout = provider_timeout(timeout_seconds)

    async def search(self, query: str, limit: int = 10) -> list[Paper]:
        """Search with a small explicit field projection and normalize its batch."""

        payload = await self._get_json(
            self.base_url,
            params={
                "query": query.replace("-", " "),
                "limit": clamp_provider_limit(limit, maximum=100),
                "fields": self.fields,
            },
        )
        return self._papers_from_payload(payload, list_key="data", record_key=None)

    async def references(self, paper_identifier: str, limit: int = 10) -> list[Paper]:
        """Return normalized papers cited by the identified paper."""

        identifier = self._normalized_identifier(paper_identifier)
        payload = await self._get_json(
            f"{_GRAPH_PAPER_URL}/{identifier}/references",
            params={
                "limit": clamp_provider_limit(limit, maximum=100),
                "fields": self.fields,
            },
        )
        return self._papers_from_payload(payload, list_key="data", record_key="citedPaper")

    async def citations(self, paper_identifier: str, limit: int = 10) -> list[Paper]:
        """Return normalized papers that cite the identified paper."""

        identifier = self._normalized_identifier(paper_identifier)
        payload = await self._get_json(
            f"{_GRAPH_PAPER_URL}/{identifier}/citations",
            params={
                "limit": clamp_provider_limit(limit, maximum=100),
                "fields": self.fields,
            },
        )
        return self._papers_from_payload(payload, list_key="data", record_key="citingPaper")

    async def recommendations(self, paper_identifier: str, limit: int = 10) -> list[Paper]:
        """Return normalized papers recommended for the identified paper."""

        identifier = self._normalized_identifier(paper_identifier)
        payload = await self._get_json(
            f"{_RECOMMENDATIONS_URL}/{identifier}",
            params={
                "limit": clamp_provider_limit(limit, maximum=100),
                "fields": self.fields,
            },
        )
        return self._papers_from_payload(payload, list_key="recommendedPapers", record_key=None)

    @staticmethod
    def _normalized_identifier(value: str) -> str:
        """Validate and URL-quote a caller-supplied Semantic Scholar paper identifier."""

        if not isinstance(value, str) or not value.strip():
            raise ValueError("paper_identifier must be a non-empty string")
        return quote(value.strip(), safe=":")

    async def _get_json(self, url: str, *, params: Mapping[str, str | int]) -> object:
        """Issue one bounded GET and decode its JSON body behind safe errors."""

        headers = {"x-api-key": self._api_key} if self._api_key else {}
        try:
            response = await get_with_retry(
                self._client,
                url,
                params=params,
                headers=headers,
                timeout=self._timeout,
            )
            return response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise ProviderUnavailableError(
                safe_provider_error("Semantic Scholar", error)
            ) from error

    def _papers_from_payload(
        self,
        payload: object,
        *,
        list_key: str,
        record_key: str | None,
    ) -> list[Paper]:
        """Normalize one bounded response body into provider-neutral papers."""

        papers: list[Paper] = []
        for record in _records_from_payload(payload, list_key=list_key, record_key=record_key):
            if (paper := self._paper_from_record(record)) is not None:
                papers.append(paper)
        logger.info("Semantic Scholar returned %d normalized paper(s).", len(papers))
        return papers

    @classmethod
    def _paper_from_record(cls, record: Mapping[str, Any]) -> Paper | None:
        paper_id = string_or_none(record.get("paperId"))
        title = string_or_none(record.get("title"))
        if paper_id is None or title is None:
            return None
        raw_ids = record.get("externalIds")
        ids = known_external_ids(raw_ids if isinstance(raw_ids, Mapping) else None)
        ids["s2"] = paper_id
        ids["semantic_scholar"] = paper_id
        doi = normalize_doi(record.get("doi")) or ids.get("doi")
        if doi:
            ids["doi"] = doi
        authors = record.get("authors")
        author_names: list[str] = []
        if isinstance(authors, list):
            for author in authors:
                if not isinstance(author, Mapping):
                    continue
                name = string_or_none(author.get("name"))
                if name:
                    author_names.append(name)
        return Paper(
            id=f"semantic_scholar:{paper_id}",
            title=title,
            abstract=string_or_none(record.get("abstract")),
            authors=author_names,
            publication_year=integer_or_none(record.get("year")),
            venue=string_or_none(record.get("venue")),
            doi=doi,
            url=string_or_none(record.get("url")),
            citation_count=integer_or_none(record.get("citationCount")),
            source=cls.name,
            external_ids=ids,
        )


def _records_from_payload(
    payload: object,
    *,
    list_key: str,
    record_key: str | None,
) -> list[Mapping[str, Any]]:
    """Extract candidate paper records from one bounded response body."""

    if not isinstance(payload, Mapping):
        raise ProviderUnavailableError(_UNEXPECTED_SHAPE_MESSAGE)
    listing = payload.get(list_key)
    if not isinstance(listing, list):
        raise ProviderUnavailableError(_UNEXPECTED_SHAPE_MESSAGE)
    records: list[Mapping[str, Any]] = []
    for item in listing:
        if record_key is None:
            if isinstance(item, Mapping):
                records.append(item)
            continue
        if isinstance(item, Mapping) and isinstance(nested := item.get(record_key), Mapping):
            records.append(nested)
    return records
