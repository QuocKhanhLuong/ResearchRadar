"""Deterministic preparation and validation helpers for paper analysis.

The end-to-end URL/download/LLM/persistence workflow intentionally belongs in a
later reader service. These helpers keep that future workflow bounded and make
evidence checks independent from a particular LLM implementation.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Mapping

from research_radar.models import PaperCard, PaperDocument
from research_radar.reader.parser import canonical_section_name

logger = logging.getLogger(__name__)

DEFAULT_MAX_LLM_INPUT_CHARS = 24_000
DEFAULT_MAX_SECTION_CHARS = 6_000

_USEFUL_SECTION_ORDER = (
    "Abstract",
    "Introduction",
    "Related Work",
    "Method",
    "Experiments",
    "Results",
    "Discussion",
    "Limitations",
    "Conclusion",
)
_WHITESPACE = re.compile(r"\s+")


def select_useful_sections(
    document: PaperDocument,
    *,
    max_chars: int = DEFAULT_MAX_LLM_INPUT_CHARS,
    max_section_chars: int = DEFAULT_MAX_SECTION_CHARS,
) -> dict[str, str]:
    """Select a stable, bounded subset of useful paper text for an LLM.

    If a PDF has no recognized headings, a clearly labeled, bounded excerpt is
    returned. It is intentionally not treated as a valid evidence section by
    :func:`validate_card_evidence`.
    """

    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    if max_section_chars < 1:
        raise ValueError("max_section_chars must be at least 1")

    available_sections = _canonical_sections(document.sections)
    selected: dict[str, str] = {}
    remaining = max_chars
    for section_name in _USEFUL_SECTION_ORDER:
        content = available_sections.get(section_name)
        if not content or remaining <= 0:
            continue
        allocation = min(remaining, max_section_chars)
        excerpt = _truncate(content, allocation)
        if excerpt:
            selected[section_name] = excerpt
            # Reserve the whole allocation rather than using a few spare
            # characters left by word-boundary truncation for a meaningless
            # fragment of a later section.
            remaining -= allocation

    if selected:
        return selected

    fallback = _truncate(document.full_text, min(max_chars, max_section_chars))
    return {"Unsectioned excerpt": fallback} if fallback else {}


def format_selected_sections(selected_sections: Mapping[str, str]) -> str:
    """Format selected text with explicit labels for a structured LLM prompt."""

    return "\n\n".join(f"## {name}\n{text}" for name, text in selected_sections.items())


def validate_card_evidence(card: PaperCard, document: PaperDocument) -> PaperCard:
    """Clear claims whose declared section/text evidence cannot be verified.

    A claim can remain an unsourced model inference, but it must have both
    ``source_section`` and ``supporting_text`` cleared whenever its cited
    location is unknown or its quoted evidence does not occur in that section.
    This avoids presenting fabricated locations as research evidence.
    """

    card_copy = card.model_copy(deep=True)
    available_sections = _canonical_sections(document.sections)
    for claim in card_copy.main_claims:
        if claim.source_section is None:
            if claim.supporting_text is not None:
                claim.supporting_text = None
            continue

        canonical_name = canonical_section_name(claim.source_section)
        source_text = available_sections.get(canonical_name) if canonical_name else None
        if source_text is None:
            _clear_invalid_evidence(claim.claim, claim.source_section, "section was not detected")
            claim.source_section = None
            claim.supporting_text = None
            continue

        if claim.supporting_text is not None and _normalized_for_match(
            claim.supporting_text
        ) not in _normalized_for_match(source_text):
            _clear_invalid_evidence(
                claim.claim,
                claim.source_section,
                "supporting text was not found in the stated section",
            )
            claim.source_section = None
            claim.supporting_text = None
            continue

        claim.source_section = canonical_name

    return card_copy


def _canonical_sections(sections: Mapping[str, str]) -> dict[str, str]:
    canonical: dict[str, str] = {}
    for name, content in sections.items():
        normalized_name = canonical_section_name(name)
        if normalized_name and normalized_name not in canonical and content.strip():
            canonical[normalized_name] = content.strip()
    return canonical


def _truncate(value: str, limit: int) -> str:
    normalized = value.strip()
    if len(normalized) <= limit:
        return normalized
    boundary = normalized.rfind(" ", 0, max(1, limit - 1))
    if boundary > limit // 2:
        return f"{normalized[:boundary].rstrip()}…"
    return f"{normalized[: max(1, limit - 1)].rstrip()}…"


def _normalized_for_match(value: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _clear_invalid_evidence(claim: str, section: str, reason: str) -> None:
    logger.warning(
        "Clearing unverifiable PaperCard evidence claim",
        extra={"reason": reason, "section": section},
    )
