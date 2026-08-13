"""Tests for language safety rules prohibiting absolute novelty claims."""

from __future__ import annotations

import pytest

from research_radar.gap.miner import (
    PROHIBITED_CLAIMS,
    check_language_safety,
    enforce_language_safety,
)


@pytest.mark.parametrize("phrase", PROHIBITED_CLAIMS)
def test_prohibited_claims_detected(phrase: str) -> None:
    text = f"This research topic has {phrase} before."
    assert not check_language_safety(text)


def test_enforce_language_safety_replaces_prohibited_phrases() -> None:
    text = "No one has studied scanner domain shift in 3D MRI."
    cleaned = enforce_language_safety(text)

    assert check_language_safety(cleaned)
    assert "No one has" not in cleaned
    assert "limited evidence was found for" in cleaned


def test_valid_conservative_language_passes() -> None:
    valid_text = (
        "Within the retrieved corpus, cross-scanner generalization "
        "appears as a repeated limitation."
    )
    assert check_language_safety(valid_text)
