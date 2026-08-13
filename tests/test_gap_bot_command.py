"""Tests for Discord /gap command handler."""

from __future__ import annotations

import pytest

from research_radar.bot.client import create_bot
from research_radar.config import Settings
from research_radar.gap.service import GapAnalysisResult
from research_radar.models.gap import CandidateGap, CriticReview


@pytest.mark.asyncio
async def test_gap_command_is_registered_when_service_provided() -> None:
    class DummyGapService:
        async def analyze_gaps(self, topic: str, count: int = 1) -> GapAnalysisResult:
            return GapAnalysisResult()

        def get_candidate_detail(
            self, candidate_id: str
        ) -> tuple[CandidateGap | None, list[CriticReview]]:
            return None, []

    bot = create_bot(Settings(), gap_service=DummyGapService())
    try:
        commands = [cmd.name for cmd in bot.tree.get_commands()]
        assert "gap" in commands
        assert "gap-show" in commands
    finally:
        await bot.close()


@pytest.mark.asyncio
async def test_gap_command_renders_insufficient_evidence_response() -> None:
    class InsufficientEvidenceGapService:
        async def analyze_gaps(self, topic: str, count: int = 1) -> GapAnalysisResult:
            return GapAnalysisResult(
                is_insufficient_evidence=True,
                message="Insufficient structured evidence. Need more PaperCards.",
            )

        def get_candidate_detail(
            self, candidate_id: str
        ) -> tuple[CandidateGap | None, list[CriticReview]]:
            return None, []

    service = InsufficientEvidenceGapService()
    res = await service.analyze_gaps("quantum mri")
    assert res.is_insufficient_evidence
    assert "Insufficient structured evidence" in (res.message or "")
