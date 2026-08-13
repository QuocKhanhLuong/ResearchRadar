"""Concurrent fan-out across independent scholarly providers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from research_radar.errors import ProviderUnavailableError
from research_radar.models.paper import Paper
from research_radar.providers.base import PaperProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScoutResult:
    """Raw normalized papers plus safe partial-provider warnings."""

    papers: list[Paper]
    warnings: list[str] = field(default_factory=list)
    provider_counts: dict[str, int] = field(default_factory=dict)


class ScoutService:
    """Query enabled providers concurrently without turning partial failure into absence."""

    def __init__(self, providers: list[PaperProvider]) -> None:
        self._providers = providers

    async def search(self, query: str, limit: int) -> ScoutResult:
        """Fan out safely and surface only a concise warning per unavailable provider."""

        if not self._providers:
            raise ProviderUnavailableError("No scholarly providers are configured.")
        outcomes = await asyncio.gather(
            *(provider.search(query, limit) for provider in self._providers),
            return_exceptions=True,
        )
        papers: list[Paper] = []
        warnings: list[str] = []
        counts: dict[str, int] = {}
        for provider, outcome in zip(self._providers, outcomes, strict=True):
            if isinstance(outcome, Exception):
                logger.warning("%s provider search failed: %s", provider.name, outcome)
                warnings.append(f"{provider.name} was unavailable; results may be partial.")
                continue
            papers.extend(outcome)
            counts[provider.name] = len(outcome)
        if not papers and len(warnings) == len(self._providers):
            raise ProviderUnavailableError("All scholarly providers were unavailable.")
        return ScoutResult(papers=papers, warnings=warnings, provider_counts=counts)
