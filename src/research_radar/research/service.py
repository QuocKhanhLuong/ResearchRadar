"""Public discovery facade that keeps command handlers out of provider code."""

from __future__ import annotations

from dataclasses import dataclass, field

from research_radar.research.dedup import deduplicate
from research_radar.research.ranker import RankedPaper, rank_papers
from research_radar.research.scout import ScoutService


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Deduplicated, ranked search results and any partial-provider warnings."""

    papers: list[RankedPaper]
    warnings: list[str] = field(default_factory=list)
    provider_counts: dict[str, int] = field(default_factory=dict)


class ResearchService:
    """Validate a user query, overfetch modestly, then deduplicate and rank it."""

    def __init__(self, scout: ScoutService, *, maximum_results: int = 10) -> None:
        self._scout = scout
        self._maximum_results = maximum_results

    async def search(self, query: str, limit: int = 5) -> SearchResult:
        """Search independently enabled providers without an LLM or persistence side effect."""

        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("Search query cannot be empty.")
        if not 1 <= limit <= self._maximum_results:
            raise ValueError(f"Result count must be between 1 and {self._maximum_results}.")
        raw = await self._scout.search(normalized_query, min(max(10, limit * 2), 25))
        deduplicated = deduplicate(raw.papers)
        ranked = rank_papers(normalized_query, deduplicated)
        return SearchResult(
            papers=ranked[:limit],
            warnings=raw.warnings,
            provider_counts=raw.provider_counts,
        )
