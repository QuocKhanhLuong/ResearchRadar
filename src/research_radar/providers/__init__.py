"""Provider adapters that normalize scholarly APIs into domain models."""

from research_radar.providers.arxiv import ArxivProvider
from research_radar.providers.base import PaperProvider
from research_radar.providers.openalex import OpenAlexProvider
from research_radar.providers.semantic_scholar import SemanticScholarProvider

__all__ = ["ArxivProvider", "OpenAlexProvider", "PaperProvider", "SemanticScholarProvider"]
