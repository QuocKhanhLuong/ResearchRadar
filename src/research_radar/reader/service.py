"""Bounded direct-PDF reading workflow, independent from Discord adapters."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass

from research_radar.errors import LLMUnavailableError
from research_radar.models import Paper, PaperCard, PaperDocument
from research_radar.reader.fetcher import DirectPDFFetcher
from research_radar.reader.llm import LLMMessage, LLMProvider
from research_radar.reader.parser import PDFParser
from research_radar.reader.reader import (
    format_selected_sections,
    select_useful_sections,
    validate_card_evidence,
)
from research_radar.storage import ResearchRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReadResult:
    """A persisted paper/card and the document used to derive it."""

    paper_id: str
    paper: Paper
    card: PaperCard
    document: PaperDocument
    selected_sections: dict[str, str]


class ReaderService:
    """Compose secure download, parsing, bounded inference, and persistence."""

    def __init__(
        self,
        *,
        fetcher: DirectPDFFetcher,
        parser: PDFParser,
        llm: LLMProvider,
        repository: ResearchRepository,
        llm_provider_name: str,
        llm_model: str | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._parser = parser
        self._llm = llm
        self._repository = repository
        self._llm_provider_name = llm_provider_name
        self._llm_model = llm_model

    async def read_url(self, url: str) -> ReadResult:
        """Analyze a direct public PDF URL, never inventing a mock-provider result."""

        fetched = await self._fetcher.fetch(url)
        document = await asyncio.to_thread(
            self._parser.parse,
            fetched.content,
            source_url=fetched.source_url,
        )
        selected_sections = select_useful_sections(document)
        if not selected_sections:
            raise ValueError("No useful text could be selected from the PDF.")

        paper = Paper(
            id=f"url:{hashlib.sha256(fetched.source_url.encode()).hexdigest()[:24]}",
            title=document.title,
            abstract=selected_sections.get("Abstract"),
            authors=[],
            publication_year=None,
            venue=None,
            doi=None,
            url=fetched.source_url,
            citation_count=None,
            source="direct_pdf",
            external_ids={"url_sha256": hashlib.sha256(fetched.source_url.encode()).hexdigest()},
        )
        paper_id = await asyncio.to_thread(self._repository.upsert_merged_paper, paper)
        messages = _analysis_messages(document, selected_sections, paper_id=paper_id)
        try:
            generated_card = await self._llm.generate_structured(messages, PaperCard)
        except LLMUnavailableError:
            logger.info("Paper reading halted because the configured LLM is unavailable.")
            raise
        card = generated_card.model_copy(update={"paper_id": paper_id})
        card = validate_card_evidence(card, document)
        await asyncio.to_thread(
            self._repository.upsert_paper_card,
            card,
            source_url=fetched.source_url,
            document_sha256=hashlib.sha256(fetched.content).hexdigest(),
            selected_sections=selected_sections,
            llm_provider=self._llm_provider_name,
            llm_model=self._llm_model,
        )
        return ReadResult(
            paper_id=paper_id,
            paper=paper,
            card=card,
            document=document,
            selected_sections=selected_sections,
        )


def _analysis_messages(
    document: PaperDocument,
    selected_sections: dict[str, str],
    *,
    paper_id: str,
) -> list[LLMMessage]:
    """Create an explicit evidence-bound prompt for one structured PaperCard."""

    return [
        LLMMessage(
            role="system",
            content=(
                "Extract a PaperCard from the supplied paper sections. Return only a JSON object "
                "matching the PaperCard schema. Include structured tasks, modalities, and "
                "evaluation_conditions with status 'observed', 'explicitly_absent', or 'unknown'. "
                "Default to 'unknown' if not explicitly stated in text. Do not invent evidence: "
                "use null source_section and supporting_text when unknown. source_section must be "
                "one of the supplied labels."
            ),
        ),
        LLMMessage(
            role="user",
            content=(
                f"Paper id (return this exact value in paper_id): {paper_id}\n"
                f"Paper title: {document.title}\n\n"
                f"Available evidence:\n{format_selected_sections(selected_sections)}"
            ),
        ),
    ]
