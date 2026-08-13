from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from research_radar.errors import PaperParseError
from research_radar.models import EvidenceClaim, PaperCard, PaperDocument
from research_radar.reader import (
    PDFParseLimits,
    PDFParser,
    detect_sections,
    select_useful_sections,
    validate_card_evidence,
)


def _pdf_bytes(*pages: str) -> bytes:
    document = pymupdf.open()
    try:
        for page_text in pages:
            page = document.new_page()
            result = page.insert_textbox(
                pymupdf.Rect(36, 36, 560, 806), page_text, fontsize=10, fontname="helv"
            )
            assert result >= 0
        return document.tobytes()
    finally:
        document.close()


_PAPER_TEXT = """A Useful Paper Title
Abstract
This abstract describes a reproducible paper reader for research workflows. It extracts
text from ordinary PDFs, limits resource usage, and makes unsupported scanned documents
an explicit failure rather than pretending that extracted text is reliable.

1 Introduction
The introduction gives enough context for the test fixture and contains several words
about reproducibility, paper analysis, and transparent evidence handling.

2 Methodology
Our method uses a deterministic PDF parser, bounded extraction, and section detection.

3 Results
The results show that the parser creates stable sections under normal input conditions.

Conclusion
The conclusion emphasizes bounded processing and reliable user-visible errors.
"""


def test_parser_extracts_bounded_pdf_text_sections_and_metadata_title() -> None:
    payload = _pdf_bytes(_PAPER_TEXT)

    paper = PDFParser().parse(payload, source_url="https://example.test/paper.pdf")

    assert paper.title == "A Useful Paper Title"
    assert paper.source_url == "https://example.test/paper.pdf"
    assert "Abstract" in paper.sections
    assert "Method" in paper.sections
    assert "Results" in paper.sections
    assert "Conclusion" in paper.sections
    assert "bounded extraction" in paper.full_text


def test_parser_accepts_a_local_path(tmp_path: Path) -> None:
    path = tmp_path / "reader-paper.pdf"
    path.write_bytes(_pdf_bytes(_PAPER_TEXT))

    paper = PDFParser().parse(path)

    assert paper.title == "A Useful Paper Title"


def test_parser_rejects_scanned_like_empty_text() -> None:
    with pytest.raises(PaperParseError, match="Too little readable text"):
        PDFParser().parse(_pdf_bytes(""))


def test_parser_enforces_byte_page_and_extracted_text_limits() -> None:
    payload = _pdf_bytes(_PAPER_TEXT, _PAPER_TEXT)

    with pytest.raises(PaperParseError, match="byte parsing limit"):
        PDFParser(PDFParseLimits(max_pdf_bytes=10)).parse(payload)
    with pytest.raises(PaperParseError, match="page parsing limit"):
        PDFParser(PDFParseLimits(max_pages=1)).parse(payload)
    with pytest.raises(PaperParseError, match="character parsing limit"):
        PDFParser(PDFParseLimits(max_text_chars=250, min_text_chars=100)).parse(payload)


def test_detect_sections_normalizes_common_numbered_heading_aliases() -> None:
    sections = detect_sections(
        """Abstract: A concise abstract.
1. Introduction
Intro text.
II. Methodology
Method text.
3 Results
Result text.
References
One citation."""
    )

    assert sections == {
        "Abstract": "A concise abstract.",
        "Introduction": "Intro text.",
        "Method": "Method text.",
        "Results": "Result text.",
        "References": "One citation.",
    }


def test_selected_sections_are_ordered_bounded_and_fall_back_safely() -> None:
    document = PaperDocument(
        title="P",
        full_text="fallback text " * 30,
        sections={
            "Conclusion": "conclusion " * 20,
            "Method": "method " * 20,
            "Abstract": "abstract " * 20,
        },
    )

    selected = select_useful_sections(document, max_chars=100, max_section_chars=60)

    assert list(selected) == ["Abstract", "Method"]
    assert sum(map(len, selected.values())) <= 100

    unsectioned = select_useful_sections(
        PaperDocument(title="P", full_text="fallback text " * 20), max_chars=50
    )
    assert list(unsectioned) == ["Unsectioned excerpt"]
    assert unsectioned["Unsectioned excerpt"].startswith("fallback text")
    assert len(unsectioned["Unsectioned excerpt"]) <= 50


def test_evidence_validation_preserves_only_verifiable_source_locations() -> None:
    document = PaperDocument(
        title="P",
        full_text="full text",
        sections={"Method": "We use a deterministic parser for PDF extraction."},
    )
    original = PaperCard(
        paper_id="paper-1",
        main_claims=[
            EvidenceClaim(
                claim="Good evidence",
                source_section="Methodology",
                supporting_text="deterministic parser",
            ),
            EvidenceClaim(
                claim="Unknown section",
                source_section="Discussion",
                supporting_text="not in the document",
            ),
            EvidenceClaim(
                claim="Incorrect quotation",
                source_section="Method",
                supporting_text="invented quotation",
            ),
        ],
    )

    validated = validate_card_evidence(original, document)

    assert validated.main_claims[0].source_section == "Method"
    assert validated.main_claims[0].supporting_text == "deterministic parser"
    assert validated.main_claims[1].source_section is None
    assert validated.main_claims[1].supporting_text is None
    assert validated.main_claims[2].source_section is None
    assert validated.main_claims[2].supporting_text is None
    assert original.main_claims[1].source_section == "Discussion"
