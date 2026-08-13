"""PDF reader foundations independent from Discord and storage adapters."""

from research_radar.reader.fetcher import (
    DEFAULT_DOWNLOAD_CHUNK_SIZE,
    DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    DEFAULT_MAX_PDF_DOWNLOAD_BYTES,
    DEFAULT_MAX_REDIRECTS,
    DirectPDFFetcher,
    FetchedPDF,
    PaperDownloadError,
    PDFDownloadLimits,
    fetch_pdf,
)
from research_radar.reader.parser import (
    PaperParser,
    PDFParseLimits,
    PDFParser,
    canonical_section_name,
    detect_sections,
    parse_pdf,
)
from research_radar.reader.reader import (
    DEFAULT_MAX_LLM_INPUT_CHARS,
    DEFAULT_MAX_SECTION_CHARS,
    format_selected_sections,
    select_useful_sections,
    validate_card_evidence,
)
from research_radar.reader.service import ReaderService, ReadResult

__all__ = [
    "DEFAULT_MAX_LLM_INPUT_CHARS",
    "DEFAULT_MAX_SECTION_CHARS",
    "DEFAULT_DOWNLOAD_CHUNK_SIZE",
    "DEFAULT_DOWNLOAD_TIMEOUT_SECONDS",
    "DEFAULT_MAX_PDF_DOWNLOAD_BYTES",
    "DEFAULT_MAX_REDIRECTS",
    "DirectPDFFetcher",
    "FetchedPDF",
    "PDFDownloadLimits",
    "PDFParseLimits",
    "PDFParser",
    "PaperParser",
    "PaperDownloadError",
    "ReadResult",
    "ReaderService",
    "canonical_section_name",
    "detect_sections",
    "fetch_pdf",
    "format_selected_sections",
    "parse_pdf",
    "select_useful_sections",
    "validate_card_evidence",
]
