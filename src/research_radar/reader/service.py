"""Bounded direct-PDF reading workflow, independent from Discord adapters."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass

from research_radar.artifacts.base import sha256_hex
from research_radar.errors import LLMUnavailableError
from research_radar.models import Paper, PaperCard, PaperDocument
from research_radar.reader.cache import CachedDocument, DocumentCache
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
    from_cache: bool = False
    sha256: str = ""


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
        document_cache: DocumentCache | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._parser = parser
        self._llm = llm
        self._repository = repository
        self._llm_provider_name = llm_provider_name
        self._llm_model = llm_model
        self._document_cache = document_cache

    async def read_url(self, url: str, *, force_refresh: bool = False) -> ReadResult:
        """Analyze a direct public PDF URL, never inventing a mock-provider result.

        A repeat read of byte-identical content reuses the cached parsed
        document and any stored card for the same content digest, costing zero
        LLM requests unless ``force_refresh`` is set.
        """

        fetched = await self._fetcher.fetch(url)
        sha = sha256_hex(fetched.content)

        cached = await self._load_cached_document(
            fetched.source_url,
            sha256=sha,
            force_refresh=force_refresh,
        )
        if cached is not None:
            document = cached.document
        else:
            document = await asyncio.to_thread(
                self._parser.parse,
                fetched.content,
                source_url=fetched.source_url,
            )

        selected_sections = select_useful_sections(document)
        if not selected_sections:
            raise ValueError("No useful text could be selected from the PDF.")

        paper = _document_paper(document, fetched.source_url, selected_sections)
        paper_id = await asyncio.to_thread(self._repository.upsert_merged_paper, paper)

        if self._document_cache is not None and cached is None:
            await asyncio.to_thread(
                self._document_cache.store,
                paper_id=paper_id,
                content=fetched.content,
                document=document,
                source_url=fetched.source_url,
            )

        if not force_refresh:
            stored_record = await asyncio.to_thread(
                self._repository.get_paper_card_record, paper_id
            )
            if stored_record is not None and stored_record.document_sha256 == sha:
                return ReadResult(
                    paper_id=paper_id,
                    paper=paper,
                    card=stored_record.card,
                    document=document,
                    selected_sections=selected_sections,
                    from_cache=True,
                    sha256=sha,
                )

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
            document_sha256=sha,
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
            from_cache=cached is not None,
            sha256=sha,
        )

    async def _load_cached_document(
        self,
        source_url: str,
        *,
        sha256: str,
        force_refresh: bool,
    ) -> CachedDocument | None:
        """Resolve an already-known paper id for this URL, then load cached text.

        Artifacts are keyed by the storage-assigned paper id, so the id has to
        be known before the cache can be consulted. It is resolved with a pure
        lookup on the URL identity: a paper this reader has never seen simply
        misses the cache. Nothing is written to canonical storage here, so a
        read that fails before persistence leaves no placeholder row behind.
        """

        if self._document_cache is None or force_refresh:
            return None
        url_digest = hashlib.sha256(source_url.encode()).hexdigest()
        paper_id = await asyncio.to_thread(
            self._repository.find_paper_id_by_source,
            "url_sha256",
            url_digest,
        )
        if paper_id is None:
            return None
        return await asyncio.to_thread(
            self._document_cache.load,
            paper_id=paper_id,
            sha256=sha256,
        )


def _document_paper(
    document: PaperDocument,
    source_url: str,
    selected_sections: dict[str, str],
) -> Paper:
    """Build the canonical direct-PDF Paper record from one parsed document."""

    url_digest = hashlib.sha256(source_url.encode()).hexdigest()
    return Paper(
        id=f"url:{url_digest[:24]}",
        title=document.title,
        abstract=selected_sections.get("Abstract"),
        authors=[],
        publication_year=None,
        venue=None,
        doi=None,
        url=source_url,
        citation_count=None,
        source="direct_pdf",
        external_ids={"url_sha256": url_digest},
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
