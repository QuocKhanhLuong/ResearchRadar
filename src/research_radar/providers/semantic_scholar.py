"""Semantic Scholar Graph API adapter."""

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


class SemanticScholarProvider:
    """Use explicit Graph API fields and return provider-neutral records."""

    name = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
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

        headers = {"x-api-key": self._api_key} if self._api_key else {}
        try:
            response = await get_with_retry(
                self._client,
                self.base_url,
                params={
                    "query": query.replace("-", " "),
                    "limit": clamp_provider_limit(limit, maximum=100),
                    "fields": self.fields,
                },
                headers=headers,
                timeout=self._timeout,
            )
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise ProviderUnavailableError(
                safe_provider_error("Semantic Scholar", error)
            ) from error
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ProviderUnavailableError(
                "Semantic Scholar returned an unexpected response shape."
            )
        papers: list[Paper] = []
        for item in data:
            if isinstance(item, Mapping) and (paper := self._paper_from_record(item)):
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
