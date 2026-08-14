"""Research memory question answering engine (/ask V1.1 Hardened)."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from research_radar.models.gap import CandidateGap, CriticReview
from research_radar.models.project import Project, ProjectGapLink, ProjectPaperLink
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
    """Expanded evidence context bounding an /ask query."""

    query: str
    project: Project | None = None
    retrieved_papers: list[StoredPaper] = field(default_factory=list)
    retrieved_cards: list[StoredPaperCard] = field(default_factory=list)
    retrieved_gaps: list[CandidateGap] = field(default_factory=list)
    critic_reviews: dict[str, CriticReview] = field(default_factory=dict)
    project_paper_links: list[ProjectPaperLink] = field(default_factory=list)
    project_gap_links: list[ProjectGapLink] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    rejected_ideas: list[str] = field(default_factory=list)


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

        proj_paper_links: list[ProjectPaperLink] = []
        proj_gap_links: list[ProjectGapLink] = []
        proj_paper_map: dict[str, ProjectPaperLink] = {}
        proj_gap_map: dict[str, ProjectGapLink] = {}

        if project is not None:
            proj_paper_links = self._repository.list_project_papers(project.id)
            proj_gap_links = self._repository.list_project_gaps(project.id)
            proj_paper_map = {link.paper_id: link for link in proj_paper_links}
            proj_gap_map = {link.candidate_id: link for link in proj_gap_links}

        # 1. Collect candidate papers (Project-linked + Global)
        candidate_papers_dict: dict[str, StoredPaper] = {}
        for pid in proj_paper_map:
            p = self._repository.get_paper(pid)
            if p is not None:
                candidate_papers_dict[p.id] = p

        global_papers = self._repository.search_papers(query=question, limit=max_evidence * 3)
        for p in global_papers:
            if p.id not in candidate_papers_dict:
                candidate_papers_dict[p.id] = p

        # 2. Collect candidate paper cards
        candidate_cards_dict: dict[str, StoredPaperCard] = {}
        for pid in candidate_papers_dict:
            card_row = self._repository.get_paper_card(pid)
            if card_row is not None:
                candidate_cards_dict[pid] = card_row

        # 3. Score & rank papers
        relation_bonus_map = {
            "seed": 30.0,
            "supporting": 30.0,
            "conflicting": 30.0,
            "relevant": 25.0,
            "background": 20.0,
        }

        scored_papers: list[tuple[float, StoredPaper]] = []
        for pid, paper in candidate_papers_dict.items():
            card_row = candidate_cards_dict.get(pid)
            c_text = ""
            if card_row is not None:
                c_text = (
                    f"{card_row.card.problem or ''} {' '.join(card_row.card.methods)} "
                    f"{' '.join(card_row.card.metrics)} {' '.join(card_row.card.limitations)}"
                )

            p_text = f"{paper.title} {paper.abstract or ''} {' '.join(paper.authors)} {c_text}"
            p_toks = _tokenize(p_text)
            overlap = len(q_tokens & p_toks)

            project_boost = 0.0
            if pid in proj_paper_map:
                rel = proj_paper_map[pid].relation
                project_boost = relation_bonus_map.get(rel, 20.0)

            total_score = project_boost + float(overlap)
            if total_score > 0 or pid in proj_paper_map:
                scored_papers.append((total_score, paper))

        scored_papers.sort(key=lambda x: x[0], reverse=True)
        top_papers = [p for _, p in scored_papers[:max_evidence]]
        top_cards = [candidate_cards_dict[p.id] for p in top_papers if p.id in candidate_cards_dict]

        # 4. Collect & score candidate gaps
        candidate_gaps_dict: dict[str, CandidateGap] = {}
        for gid in proj_gap_map:
            g = self._repository.get_candidate(gid)
            if g is not None:
                candidate_gaps_dict[g.id] = g

        global_gaps = self._repository.list_candidates(limit=50)
        for g in global_gaps:
            if g.id not in candidate_gaps_dict:
                candidate_gaps_dict[g.id] = g

        scored_gaps: list[tuple[float, CandidateGap]] = []
        for gid, gap in candidate_gaps_dict.items():
            g_text = f"{gap.title} {gap.description} {gap.research_question}"
            g_toks = _tokenize(g_text)
            g_overlap = len(q_tokens & g_toks)

            project_boost = 0.0
            if gid in proj_gap_map:
                st = proj_gap_map[gid].status
                if st in ("active", "interesting"):
                    project_boost = 25.0
                elif st in ("rejected", "resolved"):
                    project_boost = 5.0
                else:
                    project_boost = 20.0

            total_score = project_boost + float(g_overlap)
            if total_score > 0 or gid in proj_gap_map:
                scored_gaps.append((total_score, gap))

        scored_gaps.sort(key=lambda x: x[0], reverse=True)
        top_gaps = [g for _, g in scored_gaps[:max_evidence]]

        # 5. Fetch Critic Reviews for top gaps
        critic_reviews: dict[str, CriticReview] = {}
        for g in top_gaps:
            reviews = self._repository.list_critic_reviews(g.id)
            if reviews:
                critic_reviews[g.id] = reviews[-1]

        # 6. Assemble AskContext
        _ = AskContext(
            query=question,
            project=project,
            retrieved_papers=top_papers,
            retrieved_cards=top_cards,
            retrieved_gaps=top_gaps,
            critic_reviews=critic_reviews,
            project_paper_links=[
                proj_paper_map[p.id] for p in top_papers if p.id in proj_paper_map
            ],
            project_gap_links=[proj_gap_map[g.id] for g in top_gaps if g.id in proj_gap_map],
            hypotheses=list(project.hypotheses) if project else [],
            constraints=list(project.constraints) if project else [],
            rejected_ideas=list(project.rejected_ideas) if project else [],
        )

        # Source ID Safety sets
        allowed_paper_ids = {p.id for p in top_papers} | {c.card.paper_id for c in top_cards}
        allowed_gap_ids = {g.id for g in top_gaps}

        # Deterministic fallback when no evidence matches and no project provided
        if not top_papers and not top_cards and not top_gaps and not project:
            return AskResponse(
                answer=(
                    "I found insufficient stored evidence to determine an answer to your question. "
                    "Within the papers currently stored in ResearchRadar, no matching content "
                    "was found."
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
                if project.rejected_ideas:
                    lines.append(
                        f"• Project Rejected Ideas (History): {', '.join(project.rejected_ideas)}"
                    )
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

            val_pids = [pid for pid in sorted(allowed_paper_ids) if pid in allowed_paper_ids]
            val_gids = [gid for gid in sorted(allowed_gap_ids) if gid in allowed_gap_ids]

            return AskResponse(
                answer="\n".join(lines),
                referenced_paper_ids=val_pids,
                referenced_gap_ids=val_gids,
                is_sufficient_evidence=True,
            )

        # Build evidence packet for LLM
        evidence_blocks: list[str] = []
        if project:
            rej_text = ", ".join(project.rejected_ideas)
            proj_info = [
                "--- Project Memory ---",
                f"Name: {project.name}",
                f"Goal: {project.goal or 'N/A'}",
                f"Keywords: {', '.join(project.keywords)}",
                f"Hypotheses: {', '.join(project.hypotheses)}",
                f"Constraints: {', '.join(project.constraints)}",
                f"REJECTED IDEAS (Project History - Do NOT recommend as new): {rej_text}",
            ]
            evidence_blocks.append("\n".join(proj_info))

        for p in top_papers:
            p_block = [
                f"--- Paper {p.id} ---",
                f"Title: {p.title}",
                f"Abstract: {p.abstract or 'N/A'}",
            ]
            if p.id in proj_paper_map:
                link = proj_paper_map[p.id]
                p_block.append(f"Project Relation: {link.relation} (Note: {link.note or 'N/A'})")

            card_row = candidate_cards_dict.get(p.id)
            if card_row is not None:
                c = card_row.card
                if c.problem:
                    p_block.append(f"Problem: {c.problem}")
                if c.methods:
                    p_block.append(f"Methods: {', '.join(c.methods)}")
                if c.datasets:
                    p_block.append(f"Datasets: {', '.join(c.datasets)}")
                if c.metrics:
                    p_block.append(f"Metrics: {', '.join(c.metrics)}")
                if c.limitations:
                    p_block.append(f"Limitations: {', '.join(c.limitations)}")

            evidence_blocks.append("\n".join(p_block))

        for g in top_gaps:
            g_block = [
                f"--- CandidateGap {g.id} ---",
                f"Title: {g.title}",
                f"Type: {g.gap_type}",
                f"Review Status: {g.review_status}",
                f"Confidence: {g.confidence}",
                f"RQ: {g.research_question}",
                f"Description: {g.description}",
            ]
            if g.id in proj_gap_map:
                link = proj_gap_map[g.id]
                status_note = (
                    "PAST RESOLVED/REJECTED GAP"
                    if link.status in ("resolved", "rejected")
                    else "ACTIVE"
                )
                g_block.append(f"Project Gap Link Status: {link.status} ({status_note})")

            if g.id in critic_reviews:
                rev = critic_reviews[g.id]
                g_block.append(
                    f"Latest Critic Review (v{rev.review_version}): decision={rev.decision}, "
                    f"rationale={rev.rationale}, caveats={', '.join(rev.caveats)}"
                )

            evidence_blocks.append("\n".join(g_block))

        prompt_evidence = "\n\n".join(evidence_blocks)

        system_prompt = (
            "You are ResearchRadar, a private single-user research assistant.\n"
            "STRICT RULES:\n"
            "1. Answer ONLY using the supplied stored ResearchRadar evidence context below.\n"
            "2. Distinguish between:\n"
            "   - DIRECT EVIDENCE: supported by stored Paper/PaperCard/Gap/Critic evidence\n"
            "   - PROJECT MEMORY: goal, hypothesis, constraint, rejected idea\n"
            "   - INFERENCE: assistant synthesis based on stored evidence\n"
            "3. PROJECT REJECTED IDEAS are historical record. "
            "Do NOT recommend rejected ideas again as if they were new ideas.\n"
            "4. RESOLVED OR REJECTED GAPS are past status. "
            "Do NOT present a resolved or rejected gap as an active open gap.\n"
            "5. If evidence is insufficient, state clearly: "
            "'I found insufficient stored evidence to determine...'\n"
            "6. NEVER claim global literature absence (FORBIDDEN: 'No one has studied', "
            "'This is the first', 'No research exists').\n"
            "7. ALWAYS frame scope using allowed phrases like: "
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

            # Phase 3: Source ID Safety Validation (Discard hallucinated IDs)
            val_pids = [
                pid
                for pid in dict.fromkeys(llm_res.referenced_paper_ids)
                if pid in allowed_paper_ids
            ]
            val_gids = [
                gid
                for gid in dict.fromkeys(llm_res.referenced_gap_ids)
                if gid in allowed_gap_ids
            ]

            return AskResponse(
                answer=clean_answer,
                referenced_paper_ids=val_pids,
                referenced_gap_ids=val_gids,
                is_sufficient_evidence=llm_res.is_sufficient_evidence,
            )
        except Exception as exc:
            logger.exception("LLM Q&A failed.")
            return AskResponse(
                answer=f"Error processing question with LLM: {exc}",
                referenced_paper_ids=list(dict.fromkeys(allowed_paper_ids)),
                referenced_gap_ids=list(dict.fromkeys(allowed_gap_ids)),
                is_sufficient_evidence=False,
            )
