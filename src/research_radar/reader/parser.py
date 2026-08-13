"""Bounded, text-first PDF extraction for research papers.

This module deliberately accepts only local paths or already-fetched PDF bytes.
Network retrieval belongs at a higher boundary so parsing stays testable and is
never used as an arbitrary URL fetcher.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pymupdf

from research_radar.errors import PaperParseError
from research_radar.models import PaperDocument

_DEFAULT_MAX_PDF_BYTES: Final = 20 * 1024 * 1024
_DEFAULT_MAX_PAGES: Final = 100
_DEFAULT_MAX_TEXT_CHARS: Final = 250_000
_DEFAULT_MIN_TEXT_CHARS: Final = 200


@dataclass(frozen=True, slots=True)
class PDFParseLimits:
    """Hard limits applied before a document reaches later reader stages."""

    max_pdf_bytes: int = _DEFAULT_MAX_PDF_BYTES
    max_pages: int = _DEFAULT_MAX_PAGES
    max_text_chars: int = _DEFAULT_MAX_TEXT_CHARS
    min_text_chars: int = _DEFAULT_MIN_TEXT_CHARS

    def __post_init__(self) -> None:
        for name, value in (
            ("max_pdf_bytes", self.max_pdf_bytes),
            ("max_pages", self.max_pages),
            ("max_text_chars", self.max_text_chars),
            ("min_text_chars", self.min_text_chars),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.min_text_chars > self.max_text_chars:
            raise ValueError("min_text_chars cannot exceed max_text_chars")


_SECTION_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "Abstract": ("abstract",),
    "Introduction": ("introduction",),
    "Related Work": ("related work", "background", "literature review"),
    "Method": (
        "method",
        "methods",
        "methodology",
        "proposed method",
        "approach",
        "materials and methods",
        "material and methods",
    ),
    "Experiments": (
        "experiment",
        "experiments",
        "experimental setup",
        "experimental settings",
        "evaluation",
        "evaluations",
        "implementation details",
    ),
    "Results": ("result", "results", "findings"),
    "Discussion": ("discussion",),
    "Limitations": ("limitation", "limitations"),
    "Conclusion": ("conclusion", "conclusions", "concluding remarks"),
    "References": ("reference", "references", "bibliography"),
}

_SECTION_ORDER: Final[tuple[str, ...]] = tuple(_SECTION_ALIASES)
_ALIAS_TO_SECTION: Final[dict[str, str]] = {
    alias: section for section, aliases in _SECTION_ALIASES.items() for alias in aliases
}
_LEADING_SECTION_NUMBER = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*)|(?:[ivxlcdm]+))[\s.)\-:]+", re.IGNORECASE
)
_TRAILING_HEADING_PUNCTUATION = re.compile(r"[.:;\-–—]+$")
_INLINE_HEADING = re.compile(r"^(.{1,100}?)[\s]*[:\-–—][\s]+(.+)$")
_SPACE_RUN = re.compile(r"[ \t\f\v]+")


def canonical_section_name(name: str) -> str | None:
    """Return the canonical common-paper heading for *name*, when known.

    The same normalizer is shared by section extraction and later evidence
    validation so a model's ``Methodology`` can be mapped to detected
    ``Method`` without accepting unrelated free-form labels.
    """

    normalized = _normalize_heading(name)
    return _ALIAS_TO_SECTION.get(normalized)


class PDFParser:
    """Extract a text-based PDF into a bounded :class:`PaperDocument`."""

    def __init__(self, limits: PDFParseLimits | None = None) -> None:
        self.limits = limits or PDFParseLimits()

    def parse(
        self,
        source: bytes | bytearray | memoryview | str | Path,
        *,
        source_url: str | None = None,
    ) -> PaperDocument:
        """Parse a PDF path or bytes and reject unsafe/low-quality extraction.

        This is synchronous because PyMuPDF is blocking. A future async reader
        service should call it through ``asyncio.to_thread`` rather than doing
        extraction inside a Discord interaction callback.
        """

        payload, path_hint = self._read_source(source)
        self._ensure_byte_limit(payload)

        try:
            document = pymupdf.open(stream=payload, filetype="pdf")
        except (RuntimeError, ValueError, TypeError) as exc:
            raise PaperParseError("The supplied file is not a readable PDF.") from exc

        try:
            page_count = document.page_count
            if page_count < 1:
                raise PaperParseError("The PDF has no pages to extract.")
            if page_count > self.limits.max_pages:
                raise PaperParseError(
                    f"The PDF has {page_count} pages, exceeding the "
                    f"{self.limits.max_pages}-page parsing limit."
                )

            pages: list[str] = []
            text_char_count = 0
            for page_number in range(page_count):
                try:
                    page = document.load_page(page_number)
                    page_text = _normalize_page_text(page.get_text("text"))
                except (RuntimeError, ValueError) as exc:
                    raise PaperParseError(
                        f"Text extraction failed on PDF page {page_number + 1}."
                    ) from exc

                text_char_count += len(page_text)
                if text_char_count > self.limits.max_text_chars:
                    raise PaperParseError(
                        "The extracted PDF text exceeds the "
                        f"{self.limits.max_text_chars}-character parsing limit."
                    )
                if page_text:
                    pages.append(page_text)

            full_text = "\n\n".join(pages).strip()
            self._ensure_extraction_quality(full_text)
            sections = detect_sections(full_text)
            return PaperDocument(
                title=_choose_title(document.metadata or {}, full_text, path_hint),
                sections=sections,
                full_text=full_text,
                source_url=source_url,
            )
        finally:
            document.close()

    def _read_source(
        self, source: bytes | bytearray | memoryview | str | Path
    ) -> tuple[bytes, str | None]:
        if isinstance(source, (bytes, bytearray, memoryview)):
            return bytes(source), None

        path = Path(source)
        try:
            if not path.is_file():
                raise PaperParseError("The supplied PDF path does not point to a file.")
            size = path.stat().st_size
            if size > self.limits.max_pdf_bytes:
                raise PaperParseError(
                    f"The PDF is {size} bytes, exceeding the "
                    f"{self.limits.max_pdf_bytes}-byte parsing limit."
                )
            return path.read_bytes(), path.stem
        except OSError as exc:
            raise PaperParseError("The supplied PDF path could not be read.") from exc

    def _ensure_byte_limit(self, payload: bytes) -> None:
        size = len(payload)
        if size == 0:
            raise PaperParseError("The supplied PDF is empty.")
        if size > self.limits.max_pdf_bytes:
            raise PaperParseError(
                f"The PDF is {size} bytes, exceeding the "
                f"{self.limits.max_pdf_bytes}-byte parsing limit."
            )

    def _ensure_extraction_quality(self, text: str) -> None:
        text_length = len(text)
        alphanumeric_count = sum(character.isalnum() for character in text)
        minimum_alphanumeric = max(20, self.limits.min_text_chars // 5)
        if text_length < self.limits.min_text_chars or alphanumeric_count < minimum_alphanumeric:
            raise PaperParseError(
                "Too little readable text was extracted from this PDF; it may be scanned "
                "or image-only. OCR is not available in ResearchRadar V1."
            )


# ``PaperParser`` is a descriptive alias for callers that do not care about
# the underlying PDF library. Keep ``PDFParser`` as the concrete public name.
PaperParser = PDFParser


def parse_pdf(
    source: bytes | bytearray | memoryview | str | Path,
    *,
    source_url: str | None = None,
    limits: PDFParseLimits | None = None,
) -> PaperDocument:
    """Convenience function for one-off bounded PDF extraction."""

    return PDFParser(limits).parse(source, source_url=source_url)


def detect_sections(text: str) -> dict[str, str]:
    """Extract common paper sections using conservative line-based headings."""

    lines = [_normalize_line(line) for line in text.splitlines()]
    headings: list[tuple[int, str, str | None]] = []
    for index, line in enumerate(lines):
        if not line:
            continue
        section = canonical_section_name(line)
        inline_text: str | None = None
        if section is None:
            match = _INLINE_HEADING.match(line)
            if match:
                section = canonical_section_name(match.group(1))
                inline_text = _normalize_line(match.group(2)) if section else None
        if section is None:
            continue
        # A repeated running header should not cut a section into fragments.
        if any(existing_section == section for _, existing_section, _ in headings):
            continue
        headings.append((index, section, inline_text))

    sections: dict[str, str] = {}
    for heading_index, (start, section, inline_text) in enumerate(headings):
        end = headings[heading_index + 1][0] if heading_index + 1 < len(headings) else len(lines)
        section_lines = lines[start + 1 : end]
        if inline_text:
            section_lines.insert(0, inline_text)
        body = "\n".join(line for line in section_lines if line).strip()
        if body:
            sections[section] = body

    # Preserve the stable common-paper ordering even when headings appeared in
    # an unusual order in the extracted text.
    return {name: sections[name] for name in _SECTION_ORDER if name in sections}


def _normalize_page_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [_normalize_line(line) for line in value.split("\n")]
    return "\n".join(lines).strip()


def _normalize_line(value: str) -> str:
    return _SPACE_RUN.sub(" ", value).strip()


def _normalize_heading(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = _LEADING_SECTION_NUMBER.sub("", normalized)
    normalized = _TRAILING_HEADING_PUNCTUATION.sub("", normalized)
    return _SPACE_RUN.sub(" ", normalized).strip()


def _choose_title(metadata: dict[str, object], full_text: str, path_hint: str | None) -> str:
    metadata_title = _normalize_line(str(metadata.get("title") or ""))
    if metadata_title and metadata_title.casefold() not in {"untitled", "none"}:
        return metadata_title[:500]

    for line in full_text.splitlines():
        candidate = _normalize_line(line)
        if candidate and canonical_section_name(candidate) is None and len(candidate) <= 500:
            return candidate

    if path_hint:
        return path_hint.replace("_", " ").replace("-", " ")[:500]
    return "Untitled paper"
