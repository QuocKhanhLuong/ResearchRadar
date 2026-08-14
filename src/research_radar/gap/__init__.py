"""Evidence-backed research gap analysis domain services."""

from research_radar.gap.contradiction import ContradictionGapMiner
from research_radar.gap.corpus import ScopedCorpusService
from research_radar.gap.coverage import CoverageGapMiner
from research_radar.gap.critic import CriticService
from research_radar.gap.evaluation import EvaluationGapMiner
from research_radar.gap.method_transfer import MethodTransferGapMiner
from research_radar.gap.miner import (
    ExplicitGapMiner,
    check_language_safety,
    enforce_language_safety,
)
from research_radar.gap.service import GapAnalysisResult, GapService

__all__ = [
    "ContradictionGapMiner",
    "CoverageGapMiner",
    "CriticService",
    "EvaluationGapMiner",
    "ExplicitGapMiner",
    "GapAnalysisResult",
    "GapService",
    "MethodTransferGapMiner",
    "ScopedCorpusService",
    "check_language_safety",
    "enforce_language_safety",
]
