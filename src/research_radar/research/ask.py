"""Research memory question answering engine (/ask V1)."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from research_radar.models.gap import CandidateGap
from research_radar.models.project import Project
from research_radar.reader.llm.base import LLMMessage, LLMProvider
from research_radar.storage.repositories import (
    ResearchRepository,
    StoredPaper,
    StoredPaperCard,
)

logger = logging.getLogger(__name__)


FORBIDDEN_PATTERNS = [
    r"\bno\s+one\s+has\s+studied\b",
    r"\bthis\s+is\s+the\s+first\b",
    r"\bno\s+research\s+exists\b",
    r"\bnobody\s+has\s+ever\b",
    r"\bfirst\s+paper\s+to\b",
]


class AskResponse(BaseModel):
    """The structured answer returned by AskService."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1)
    referenced_paper_ids: list[str] = Field(default_factory=list)
    referenced_gap_ids: list[str] = Field(default_factory=list)
    is_sufficient_evidence: bool = True


class AskLLMResponse(BaseModel):
    """Structured response model passed to LLMProvider.generate_structured."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1)
    referenced_paper_ids: list[str] = Field(default_factory=list)
    referenced_gap_ids: list[str] = Field(default_factory=list)
    is_sufficient_evidence: bool = True


@dataclass
class AskContext:
    """Retrieved evidence context bounding an /ask query."""

    query: str
    retrieved_papers: list[StoredPaper] = field(default_factory=list)
    retrieved_cards: list[StoredPaperCard] = field(default_factory=list)
    retrieved_gaps: list[CandidateGap] = field(default_factory=list)
    project: Project | None = None


def sanitize_llm_response(text: str) -> str:
    """Enforce guardrails against global literature absence language."""

    cleaned = text
    for pattern in FORBIDDEN_PATTERNS:
        cleaned = re.sub(
            pattern,
            "Within the papers currently stored in ResearchRadar",
            cleaned,
            flags=re.IGNORECASE,
        )
    return cleaned


def _tokenize(text: str) -> set[str]:
    norm = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return set(re.findall(r"\b[a-z0-9]+\b", norm))


class AskService:
    """Retrieve memory context and prompt vendor-neutral LLM for evidence-bounded Q&A."""

    def __init__(
        self,
        repository: ResearchRepository,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self._repository = repository
        self._llm_provider = llm_provider

    async def ask(
        self,
        question: str,
        *,
        project_id_or_name: str | None = None,
        max_evidence: int = 10,
    ) -> AskResponse:
        """Answer a research question scoped strictly to stored ResearchRadar memory."""

        q_tokens = _tokenize(question)
        if not q_tokens:
            return AskResponse(
                answer="Question contained no valid search terms.",
                is_sufficient_evidence=False,
            )

        project: Project | None = None
        if project_id_or_name:
            project = self._repository.get_project(project_id_or_name)

        # Lexical retrieval across Papers, PaperCards, and Gaps
        matched_papers: list[tuple[float, StoredPaper]] = []
        matched_cards: list[tuple[float, StoredPaperCard]] = []
        matched_gaps: list[tuple[float, CandidateGap]] = []

        # 1. Papers & Cards
        all_papers = self._repository.search_papers(query=question, limit=max_evidence * 2)
        for p in all_papers:
            p_text = f"{p.title} {p.abstract or ''} {' '.join(p.authors)}"
            p_toks = _tokenize(p_text)
            overlap = len(q_tokens & p_toks)
            if overlap > 0:
                matched_papers.append((float(overlap), p))

            card_row = self._repository.get_paper_card(p.id)
            if card_row is not None:
                c_text = (
                    f"{card_row.card.problem or ''} {' '.join(card_row.card.methods)} "
                    f"{' '.join(card_row.card.metrics)} {' '.join(card_row.card.limitations)}"
                )
                c_toks = _tokenize(c_text)
                c_overlap = len(q_tokens & c_toks)
                if c_overlap > 0:
                    matched_cards.append((float(c_overlap), card_row))

        # 2. Gaps
        all_gaps = self._repository.list_candidates(limit=50)
        for g in all_gaps:
            g_text = f"{g.title} {g.description} {g.research_question}"
            g_toks = _tokenize(g_text)
            g_overlap = len(q_tokens & g_toks)
            if g_overlap > 0:
                matched_gaps.append((float(g_overlap), g))

        # Sort and pick top items
        top_papers = [
            p for _, p in sorted(matched_papers, key=lambda x: x[0], reverse=True)[:max_evidence]
        ]
        top_cards = [
            c for _, c in sorted(matched_cards, key=lambda x: x[0], reverse=True)[:max_evidence]
        ]
        top_gaps = [
            g for _, g in sorted(matched_gaps, key=lambda x: x[0], reverse=True)[:max_evidence]
        ]

        ref_pids = list({p.id for p in top_papers} | {c.card.paper_id for c in top_cards})
        ref_gids = list({g.id for g in top_gaps})

        # Deterministic fallback when no LLM provider is configured or no evidence matches
        if not top_papers and not top_cards and not top_gaps and not project:
            return AskResponse(
                answer=(
                    "I found insufficient stored evidence to determine an answer to your question. "
                    "Within the papers currently stored in ResearchRadar, no matching content was "
                    "found."
                ),
                referenced_paper_ids=[],
                referenced_gap_ids=[],
                is_sufficient_evidence=False,
            )

        if self._llm_provider is None:
            # Deterministic fallback synthesis when LLM provider is not supplied
            lines: list[str] = [
                "Based on the analyzed project corpus currently stored in ResearchRadar:"
            ]
            if project:
                lines.append(f"• Project Scope: '{project.name}' (Goal: {project.goal or 'N/A'})")
            if top_papers:
                lines.append(
                    f"• Found {len(top_papers)} matching stored paper(s): "
                    + ", ".join(f"'{p.title}'" for p in top_papers[:3])
                )
            if top_gaps:
                lines.append(
                    f"• Found {len(top_gaps)} matching candidate gap(s): "
                    + ", ".join(f"'{g.title}'" for g in top_gaps[:3])
                )
            lines.append("Note: Connect an LLMProvider for automated synthesis.")
            return AskResponse(
                answer="\n".join(lines),
                referenced_paper_ids=ref_pids,
                referenced_gap_ids=ref_gids,
                is_sufficient_evidence=True,
            )

        # Build evidence text for LLM
        evidence_blocks: list[str] = []
        if project:
            evidence_blocks.append(
                f"--- Project Scope ---\nName: {project.name}\nGoal: {project.goal}\n"
                f"Hypotheses: {', '.join(project.hypotheses)}\n"
                f"Constraints: {', '.join(project.constraints)}"
            )

        for p in top_papers:
            evidence_blocks.append(
                f"--- Paper {p.id} ---\nTitle: {p.title}\nAbstract: {p.abstract or 'N/A'}"
            )

        for c in top_cards:
            evidence_blocks.append(
                f"--- PaperCard {c.card.paper_id} ---\nProblem: {c.card.problem or 'N/A'}\n"
                f"Methods: {', '.join(c.card.methods)}\n"
                f"Limitations: {', '.join(c.card.limitations)}"
            )

        for g in top_gaps:
            evidence_blocks.append(
                f"--- CandidateGap {g.id} ---\nTitle: {g.title}\n"
                f"Description: {g.description}\nRQ: {g.research_question}"
            )

        prompt_evidence = "\n\n".join(evidence_blocks)

        system_prompt = (
            "You are ResearchRadar, a private single-user research assistant.\n"
            "STRICT RULES:\n"
            "1. Answer ONLY using the supplied stored ResearchRadar evidence context below.\n"
            "2. Distinguish between direct evidence and plausible inferences.\n"
            "3. If evidence is insufficient, state clearly: "
            "'I found insufficient stored evidence to determine...'\n"
            "4. NEVER claim global literature absence (FORBIDDEN: 'No one has studied', "
            "'This is the first', 'No research exists').\n"
            "5. ALWAYS frame scope using allowed phrases like: "
            "'Within the papers currently stored in ResearchRadar...' or "
            "'Based on the analyzed project corpus...'"
        )

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(
                role="user",
                content=f"Question: {question}\n\nEvidence Context:\n{prompt_evidence}",
            ),
        ]

        try:
            llm_res = await self._llm_provider.generate_structured(messages, AskLLMResponse)
            clean_answer = sanitize_llm_response(llm_res.answer)
            return AskResponse(
                answer=clean_answer,
                referenced_paper_ids=llm_res.referenced_paper_ids or ref_pids,
                referenced_gap_ids=llm_res.referenced_gap_ids or ref_gids,
                is_sufficient_evidence=llm_res.is_sufficient_evidence,
            )
        except Exception as exc:
            logger.exception("LLM Q&A failed.")
            return AskResponse(
                answer=f"Error processing question with LLM: {exc}",
                referenced_paper_ids=ref_pids,
                referenced_gap_ids=ref_gids,
                is_sufficient_evidence=False,
            )
