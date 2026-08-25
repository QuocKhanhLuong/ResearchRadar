"""The personal research chat service: bounded evidence in, one answer out.

Pipeline per chat turn, in contracted order:

1. Normalize the query; empty text returns a usage hint with zero backend calls.
2. Route deterministically via :class:`~research_radar.chat.router.ChatRouter`.
3. Load advisory user memory when the route needs it (never raises).
4. Load explicit :class:`ProjectMemory` from SQLite for project turns.
5. Hybrid stored retrieval (lexical + optional semantic) when needed.
6. At most ONE live-discovery ingestion call when stored evidence is below the
   sufficiency threshold and the router allows it; returned ids are re-resolved
   from SQLite before they count as evidence.
7. Build the prompt (W8 module when present, minimal fallback otherwise) and
   make exactly one structured LLM call.
8. Validate cited ids against the packet; unknown ids are dropped.
9. Build the :class:`ChatResponse`.
10. Capture runs LAST, after a successful response, and only via
    :class:`~research_radar.memory.capture.MemoryCapturePolicy`.

Failure behaviour: LLM missing/failing yields a concise safe error with
``degraded=True`` (discovered-and-stored paper ids are still reported) and NO
memory write; memory, semantic, and ingestion outages each degrade to a
sanitized log line while the turn still returns. Every synchronous repository,
filesystem, embedding, or index call reachable from :meth:`ChatService.chat`
runs inside ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from research_radar.chat.evidence import (
    DiscoveryEvidenceItem,
    EvidencePacket,
    ProjectMemory,
    StoredEvidenceItem,
    validate_cited_ids,
)
from research_radar.chat.models import ChatMode, ChatRequest, ChatResponse
from research_radar.chat.prompt import build_chat_prompt
from research_radar.chat.router import ChatRouter, RouteDecision
from research_radar.memory.base import UserMemoryStore
from research_radar.memory.capture import MemoryCapturePolicy
from research_radar.memory.models import UserMemoryContext
from research_radar.models.paper_card import PaperCard
from research_radar.reader.llm.base import LLMMessage, LLMProvider
from research_radar.research.hybrid import HybridRetriever
from research_radar.research.ingestion import IngestionService
from research_radar.semantic.base import EmbeddingProvider, SemanticIndex
from research_radar.storage.repositories import ResearchRepository

logger = logging.getLogger(__name__)

_HARD_MAX_DISCOVERY_RESULTS = 12
_MAX_MEMORY_QUERY_CHARS = 400
_MAX_CARD_SUMMARY_CHARS = 280
_MAX_PROMPT_FIELD_CHARS = 400
_MAX_PROMPT_FACTS = 10
_MAX_CHAT_GAPS = 4
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_USAGE_HINT = (
    "Ask me to find, compare, or summarize research - for example "
    "'find recent papers on diffusion policy'. You can also tell me things "
    "worth remembering, like 'I prefer small open-source models'."
)

_DEGRADED_REPLY = "I couldn't synthesize an answer right now. Please try again soon."

_SYSTEM_RULES = (
    "You are ResearchRadar, a private single-user research assistant.\n"
    "RULES:\n"
    "1. Treat USER MEMORY as advisory personal context only - never as "
    "scientific support or a published result.\n"
    "2. Never present a user hypothesis or belief as an established finding.\n"
    "3. Never claim the literature contains no work on a topic; the corpus is "
    "partial.\n"
    "4. Cite only the paper ids and gap ids listed in the evidence sections.\n"
    "5. Distinguish stored full-text PaperCard evidence from abstract-only "
    "metadata, and say explicitly when an answer rests only on abstract- or "
    "discovery-level information.\n"
    "6. Say when the available evidence is insufficient rather than inventing "
    "an answer.\n"
    "7. When EXPLICIT PROJECT MEMORY and USER MEMORY conflict, EXPLICIT "
    "PROJECT MEMORY wins."
)


class ChatAnswer(BaseModel):
    """Structured reply consumed from one chat synthesis call."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1)
    referenced_paper_ids: list[str] = Field(default_factory=list)
    referenced_gap_ids: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ChatBudget:
    """Bounded retrieval, discovery, and synthesis limits for chat turns.

    ``max_discovery_results`` is additionally hard-clamped to 12 inside the
    service regardless of what a caller supplies. Seed this field from
    ``Settings.chat_live_discovery_limit`` at composition time so the configured
    setting participates in the clamp. ``auto_read_pdfs`` must remain 0 in this
    phase: mention chat never performs automatic full-PDF reads.
    """

    max_stored_evidence: int = 8
    max_discovery_results: int = 10  # hard-clamped to 12
    max_user_memory_facts: int = 8
    stored_sufficiency_threshold: int = 3
    auto_read_pdfs: int = 0  # must remain 0 in this phase

    def __post_init__(self) -> None:
        if self.auto_read_pdfs != 0:
            raise ValueError("ChatBudget.auto_read_pdfs must remain 0 in this phase.")


def _clip(text: str | None, max_chars: int) -> str | None:
    """Return whitespace-collapsed text clipped to ``max_chars``, or None."""

    if text is None:
        return None
    collapsed = " ".join(text.split())
    if not collapsed:
        return None
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def _tokenize(text: str) -> frozenset[str]:
    """Lowercase a text and split it into unique alphanumeric tokens."""

    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return frozenset(_TOKEN_RE.findall(normalized))


def _card_summary(card: PaperCard | None) -> str | None:
    """Derive one deterministic bounded summary line from a PaperCard."""

    if card is None:
        return None
    for claim in card.main_claims:
        clipped = _clip(claim.claim, _MAX_CARD_SUMMARY_CHARS)
        if clipped:
            return clipped
    for candidate in (card.problem, card.motivation):
        clipped = _clip(candidate, _MAX_CARD_SUMMARY_CHARS)
        if clipped:
            return clipped
    joined = "; ".join(filter(None, card.contributions[:2]))
    return _clip(joined, _MAX_CARD_SUMMARY_CHARS) or None


def _build_chat_messages(
    request: ChatRequest,
    decision: RouteDecision,
    packet: EvidencePacket,
) -> list[LLMMessage]:
    """Delegate prompt construction to the bounded chat prompt builder."""

    return build_chat_prompt(request, decision, packet)


class ChatService:
    """Orchestrate one bounded, evidence-grounded personal chat turn."""

    def __init__(
        self,
        *,
        repository: ResearchRepository,
        router: ChatRouter,
        user_memory: UserMemoryStore,
        capture_policy: MemoryCapturePolicy,
        llm_provider: LLMProvider | None = None,
        ingestion_service: IngestionService | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        semantic_index: SemanticIndex | None = None,
        budget: ChatBudget | None = None,
    ) -> None:
        """Wire the collaborators chat depends on; none are contacted here."""

        self._repository = repository
        self._router = router
        self._user_memory = user_memory
        self._capture_policy = capture_policy
        self._llm_provider = llm_provider
        self._ingestion_service = ingestion_service
        self._budget = budget or ChatBudget()
        self._hybrid = HybridRetriever(
            repository=repository,
            embedding_provider=embedding_provider,
            semantic_index=semantic_index,
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Run the ten-step pipeline for one chat turn."""

        query = " ".join((request.text or "").split())
        if not query:
            return ChatResponse(text=_USAGE_HINT, mode=ChatMode.CONVERSATIONAL)

        decision = await self._router.route(request)

        memory_context = await self._load_user_memory(decision, request)
        project = await self._load_project_memory(decision, request)

        stored: tuple[StoredEvidenceItem, ...] = ()
        gap_ids: tuple[str, ...] = ()
        if decision.needs_stored_research:
            stored, gap_ids = await asyncio.to_thread(
                self._collect_stored_evidence, decision.search_query
            )

        discovery, live_used = await self._maybe_run_live_discovery(decision, len(stored))
        packet = EvidencePacket(
            stored=stored,
            discovery=discovery,
            gap_ids=gap_ids,
            project=project,
            user_memory=memory_context,
            live_discovery_used=live_used,
        )
        mode = ChatMode.RESEARCH_LIVE if live_used else decision.mode

        messages = _build_chat_messages(request, decision, packet)
        answer = await self._generate_answer(messages)
        if answer is None:
            return self._degraded_response(mode, packet, memory_context)

        response = ChatResponse(
            text=answer.answer,
            mode=mode,
            paper_ids=_cited_in_namespace(answer.referenced_paper_ids, packet),
            gap_ids=_cited_in_namespace(
                answer.referenced_gap_ids, packet, allowed=packet.allowed_gap_ids
            ),
            used_user_memory=memory_context.available,
            live_discovery_used=live_used,
            evidence_scope=packet.evidence_scope,
            degraded=False,
        )
        await self._capture(request.text)
        return response

    async def _load_user_memory(
        self, decision: RouteDecision, request: ChatRequest
    ) -> UserMemoryContext:
        """Fetch advisory user memory when routed to; degrade on outage."""

        if not decision.needs_user_memory:
            return UserMemoryContext()
        # A project-scoped question often names nothing the memory graph knows
        # ("what should we try next on this project?"), so fold the resolved
        # project name into the retrieval query. Without it the user's own
        # recorded beliefs about that project are unreachable, and the
        # project-outranks-memory authority rule never gets to apply.
        query_parts = [request.text.strip()]
        project_hint = decision.project_hint or request.project_hint
        if project_hint:
            query_parts.append(project_hint.strip())
        query = " ".join(part for part in query_parts if part)[:_MAX_MEMORY_QUERY_CHARS]
        try:
            return await self._user_memory.get_context(
                query,
                limit=max(0, self._budget.max_user_memory_facts),
            )
        except Exception as exc:
            logger.warning(
                "User memory unavailable (%s); continuing without advisory context.",
                type(exc).__name__,
            )
            return UserMemoryContext(degraded=True)

    async def _load_project_memory(
        self, decision: RouteDecision, request: ChatRequest
    ) -> ProjectMemory | None:
        """Load explicit project state from SQLite for project turns."""

        hint = decision.project_hint or request.project_hint
        if not hint:
            return None
        try:
            project = await asyncio.to_thread(self._repository.get_project, hint)
        except Exception as exc:
            logger.warning(
                "Project memory unavailable (%s); continuing without it.",
                type(exc).__name__,
            )
            return None
        if project is None:
            logger.info("No stored project matched the given reference.")
            return None
        return ProjectMemory(
            project_id=project.id,
            name=project.name,
            constraints=tuple(project.constraints),
            hypotheses=tuple(project.hypotheses),
            rejected_ideas=tuple(project.rejected_ideas),
        )

    def _collect_stored_evidence(
        self, search_query: str
    ) -> tuple[tuple[StoredEvidenceItem, ...], tuple[str, ...]]:
        """Retrieve bounded stored papers and lexically relevant gaps.

        Runs entirely on a worker thread: hybrid retrieval performs SQLite
        lookups plus optional embedding/index work, and every id the retriever
        proposes is resolved against canonical storage again here before it can
        become evidence. Any storage failure degrades to empty evidence.
        """

        try:
            candidates = self._hybrid.retrieve(search_query)
            limit = max(0, self._budget.max_stored_evidence)
            stored: list[StoredEvidenceItem] = []
            for candidate in candidates[:limit]:
                paper = self._repository.get_paper(candidate.paper_id)
                if paper is None:
                    continue
                card = self._repository.get_paper_card(paper.id)
                stored.append(
                    StoredEvidenceItem(
                        paper_id=paper.id,
                        title=paper.title,
                        year=paper.publication_year,
                        venue=paper.venue,
                        abstract=paper.abstract,
                        has_paper_card=card is not None,
                        card_summary=_card_summary(card),
                    )
                )
            return tuple(stored), self._relevant_gap_ids(search_query)
        except Exception as exc:
            logger.warning(
                "Stored research retrieval degraded (%s); continuing without it.",
                type(exc).__name__,
            )
            return (), ()

    def _relevant_gap_ids(self, search_query: str) -> tuple[str, ...]:
        """Return a bounded set of lexically relevant candidate-gap ids."""

        query_tokens = _tokenize(search_query)
        if not query_tokens:
            return ()
        scored: list[tuple[int, str]] = []
        for gap in self._repository.list_candidates(limit=50):
            overlap = len(
                query_tokens
                & (
                    _tokenize(gap.title)
                    | _tokenize(gap.description)
                    | _tokenize(gap.research_question)
                )
            )
            if overlap > 0:
                scored.append((-overlap, gap.id))
        scored.sort()
        return tuple(gap_id for _, gap_id in scored[:_MAX_CHAT_GAPS])

    async def _maybe_run_live_discovery(
        self, decision: RouteDecision, stored_count: int
    ) -> tuple[tuple[DiscoveryEvidenceItem, ...], bool]:
        """Run at most one bounded ingestion call when stored evidence is thin.

        The discovery limit is clamped defensively to
        ``min(budget.max_discovery_results, 12)`` (with the configured setting
        seeded into the budget at composition time), and ``auto_read`` is always
        0: mention chat never reads full PDFs automatically. Returned canonical
        ids are re-resolved from SQLite before they count as evidence.

        Ingestion applies its ``limit`` PER PROVIDER and concatenates, so the
        request is divided by the provider count to keep the TOTAL discovered
        set inside the per-turn bound; the resolved evidence is truncated to
        the same bound as a second line of defence.
        """

        if (
            not decision.allows_live_discovery
            or not decision.search_query
            or self._ingestion_service is None
            or stored_count >= self._budget.stored_sufficiency_threshold
        ):
            return (), False
        limit = max(1, min(self._budget.max_discovery_results, _HARD_MAX_DISCOVERY_RESULTS))
        providers = max(1, getattr(self._ingestion_service, "provider_count", 1))
        per_provider = max(1, -(-limit // providers))
        try:
            result = await self._ingestion_service.ingest_research_topic(
                decision.search_query,
                limit=per_provider,
                auto_read=0,
            )
        except Exception as exc:
            logger.warning(
                "Live discovery failed (%s); continuing with stored evidence only.",
                type(exc).__name__,
            )
            return (), False
        items = await asyncio.to_thread(self._resolve_discovery_items, result.paper_ids)
        return items[:limit], bool(items)

    def _resolve_discovery_items(
        self, paper_ids: list[str]
    ) -> tuple[DiscoveryEvidenceItem, ...]:
        """Re-read every discovered id from SQLite; drop unresolvable ones."""

        items: list[DiscoveryEvidenceItem] = []
        for paper_id in paper_ids:
            paper = self._repository.get_paper(paper_id)
            if paper is None:
                logger.warning("A discovered id did not resolve in storage; dropped.")
                continue
            items.append(
                DiscoveryEvidenceItem(
                    paper_id=paper.id,
                    title=paper.title,
                    year=paper.publication_year,
                    venue=paper.venue,
                    abstract=paper.abstract,
                )
            )
        return tuple(items)

    async def _generate_answer(self, messages: list[LLMMessage]) -> ChatAnswer | None:
        """Make exactly one structured synthesis call; None means degraded."""

        provider = self._llm_provider
        if provider is None:
            logger.warning("Chat synthesis skipped: no language model is configured.")
            return None
        try:
            return await provider.generate_structured(messages, ChatAnswer)
        except Exception as exc:
            logger.warning(
                "Chat synthesis failed (%s); returning a safe reply.",
                type(exc).__name__,
            )
            return None

    def _degraded_response(
        self,
        mode: ChatMode,
        packet: EvidencePacket,
        memory_context: UserMemoryContext,
    ) -> ChatResponse:
        """Build the safe no-invention reply used when synthesis fails."""

        parts = [_DEGRADED_REPLY]
        if packet.live_discovery_used and packet.discovery:
            parts.append(
                f"Live discovery found and stored {len(packet.discovery)} new "
                "paper(s) on this topic; they are saved in your library for "
                "future questions."
            )
        return ChatResponse(
            text=" ".join(parts),
            mode=mode,
            paper_ids=tuple(item.paper_id for item in packet.discovery),
            used_user_memory=memory_context.available,
            live_discovery_used=packet.live_discovery_used,
            evidence_scope=packet.evidence_scope,
            degraded=True,
        )

    async def _capture(self, text: str) -> None:
        """Decide durability via the policy only; never fail the turn."""

        try:
            verdict = self._capture_policy.evaluate_user_message(text)
        except Exception as exc:
            logger.warning("Memory capture evaluation skipped (%s).", type(exc).__name__)
            return
        if not verdict.should_store:
            return
        try:
            await self._user_memory.add_episode(
                verdict.redacted_text,
                source_description="discord-chat",
                memory_class=verdict.memory_class,
            )
        except Exception as exc:
            logger.warning("Memory capture write skipped (%s).", type(exc).__name__)


def _cited_in_namespace(
    raw_ids: list[str],
    packet: EvidencePacket,
    *,
    allowed: set[str] | None = None,
) -> tuple[str, ...]:
    """Validate cited ids against the packet, then keep only the namespace."""

    validated = validate_cited_ids(raw_ids, packet)
    namespace = packet.allowed_paper_ids if allowed is None else allowed
    return tuple(cited_id for cited_id in validated if cited_id in namespace)
