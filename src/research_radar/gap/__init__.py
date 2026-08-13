"""Evidence-backed research gap analysis domain services."""

from research_radar.gap.corpus import ScopedCorpusService
from research_radar.gap.critic import CriticService
from research_radar.gap.miner import (
    ExplicitGapMiner,
    check_language_safety,
    enforce_language_safety,
)
from research_radar.gap.service import GapAnalysisResult, GapService

__all__ = [
    "CriticService",
    "ExplicitGapMiner",
    "GapAnalysisResult",
    "GapService",
    "ScopedCorpusService",
    "check_language_safety",
    "enforce_language_safety",
]
