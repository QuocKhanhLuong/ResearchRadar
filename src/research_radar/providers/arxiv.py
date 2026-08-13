"""arXiv Atom feed adapter with a small, polite request boundary."""

from __future__ import annotations

import asyncio
import logging
import time
import xml.etree.ElementTree as element_tree

import httpx

from research_radar.errors import ProviderUnavailableError
from research_radar.models.paper import Paper
from research_radar.providers.base import clamp_provider_limit
from research_radar.providers.normalization import normalize_arxiv_id, normalize_doi, string_or_none

logger = logging.getLogger(__name__)

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


class ArxivProvider:
    """Search arXiv's Atom endpoint and normalize viable entries."""

    name = "arxiv"
    base_url = "https://export.arxiv.org/api/query"

    def __init__(self, client: httpx.AsyncClient, *, minimum_interval_seconds: float = 3.0) -> None:
        self._client = client
        self._minimum_interval_seconds = minimum_interval_seconds
        self._request_lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def search(self, query: str, limit: int = 10) -> list[Paper]:
        """Issue a serialized bounded query and parse its Atom response."""

        await self._wait_for_request_slot()
        try:
            response = await self._client.get(
                self.base_url,
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": clamp_provider_limit(limit, maximum=100),
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
            )
            response.raise_for_status()
            root = element_tree.fromstring(response.content)
        except (httpx.HTTPError, element_tree.ParseError) as error:
            raise ProviderUnavailableError(f"arXiv search failed: {error}") from error

        papers: list[Paper] = []
        for entry in root.findall(f"{ATOM}entry"):
            paper = _paper_from_entry(entry)
            if paper is not None:
                papers.append(paper)
        logger.info("arXiv returned %d normalized paper(s).", len(papers))
        return papers

    async def _wait_for_request_slot(self) -> None:
        async with self._request_lock:
            delay = self._minimum_interval_seconds - (time.monotonic() - self._last_request_at)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_request_at = time.monotonic()


def _paper_from_entry(entry: element_tree.Element) -> Paper | None:
    entry_title = _collapse(entry.findtext(f"{ATOM}title"))
    entry_id = string_or_none(entry.findtext(f"{ATOM}id"))
    if entry_title == "Error" or entry_id is None or entry_title is None:
        return None
    arxiv_id = normalize_arxiv_id(entry_id)
    if arxiv_id is None:
        return None
    doi = normalize_doi(entry.findtext(f"{ARXIV}doi"))
    authors = [
        name
        for author in entry.findall(f"{ATOM}author")
        if (name := _collapse(author.findtext(f"{ATOM}name")))
    ]
    published = string_or_none(entry.findtext(f"{ATOM}published"))
    year = int(published[:4]) if published and published[:4].isdigit() else None
    return Paper(
        id=f"arxiv:{arxiv_id}",
        title=entry_title,
        abstract=_collapse(entry.findtext(f"{ATOM}summary")),
        authors=authors,
        publication_year=year,
        venue=_collapse(entry.findtext(f"{ARXIV}journal_ref")),
        doi=doi,
        url=entry_id,
        citation_count=None,
        source="arxiv",
        external_ids={
            key: value for key, value in {"arxiv": arxiv_id, "doi": doi}.items() if value
        },
    )


def _collapse(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    collapsed = " ".join(value.split())
    return collapsed or None
