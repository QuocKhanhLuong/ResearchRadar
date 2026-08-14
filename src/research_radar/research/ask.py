"""Research memory question answering engine (/ask V1.2 Hardened)."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from research_radar.models.gap import CandidateGap, CriticReview
from research_radar.models.paper_card import PaperCard
from research_radar.models.project import Project, ProjectGapLink, ProjectPaperLink
from research_radar.reader.llm.base import LLMMessage, LLMProvider
from research_radar.storage.repositories import (
    ResearchRepository,
    StoredPaper,
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


@dataclass(frozen=True, slots=True)
class AskBudget:
    """Character and entity limits bounding the /ask evidence packet."""

    max_papers: int = 6
    max_gaps: int = 4
    max_claims_per_card: int = 3
    max_conditions_per_card: int = 3
    max_chars_per_evidence_item: int = 600
    max_total_context_chars: int = 8000


@dataclass
class AskContext:
    """Single source of truth for retrieved research memory and allowed IDs."""

    query: str
    project: Project | None = None
    retrieved_papers: list[StoredPaper] = field(default_factory=list)
    retrieved_cards: list[PaperCard] = field(default_factory=list)
    retrieved_gaps: list[CandidateGap] = field(default_factory=list)
    critic_reviews: dict[str, CriticReview] = field(default_factory=dict)
    project_paper_links: list[ProjectPaperLink] = field(default_factory=list)
    project_gap_links: list[ProjectGapLink] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    rejected_ideas: list[str] = field(default_factory=list)

    @property
    def allowed_paper_ids(self) -> set[str]:
        """All paper IDs permissible for LLM citation."""
        return {p.id for p in self.retrieved_papers} | {
            c.paper_id for c in self.retrieved_cards
        }

    @property
    def allowed_gap_ids(self) -> set[str]:
        """All gap IDs permissible for LLM citation."""
        return {g.id for g in self.retrieved_gaps}


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


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def format_evidence_packet(ctx: AskContext, budget: AskBudget) -> str:
    """Build a deterministic, bounded text packet directly from AskContext."""

    blocks: list[str] = []
    proj_paper_map = {link.paper_id: link for link in ctx.project_paper_links}
    proj_gap_map = {link.candidate_id: link for link in ctx.project_gap_links}
    card_map = {c.paper_id: c for c in ctx.retrieved_cards}

    # 1. Project Memory
    if ctx.project:
        p_lines = [
            "--- Project Memory ---",
            f"Name: {ctx.project.name}",
            f"Goal: {ctx.project.goal or 'N/A'}",
        ]
        if ctx.project.keywords:
            p_lines.append(f"Keywords: {', '.join(ctx.project.keywords)}")
        if ctx.hypotheses:
            p_lines.append(f"Hypotheses: {', '.join(ctx.hypotheses)}")
        if ctx.constraints:
            p_lines.append(f"Constraints: {', '.join(ctx.constraints)}")
        if ctx.rejected_ideas:
            rej_text = ", ".join(ctx.rejected_ideas)
            p_lines.append(
                f"REJECTED IDEAS (Project History - Do NOT recommend as new): {rej_text}"
            )
        blocks.append("\n".join(p_lines))

    # 2. Papers & Cards
    for p in ctx.retrieved_papers[: budget.max_papers]:
        p_lines = [
            f"--- Paper {p.id} ---",
            f"Title: {p.title}",
            f"Abstract: {_truncate_text(p.abstract or 'N/A', 300)}",
        ]
        if p.id in proj_paper_map:
            link = proj_paper_map[p.id]
            p_lines.append(f"Project Relation: {link.relation} (Note: {link.note or 'N/A'})")

        if p.id in card_map:
            c = card_map[p.id]
            if c.problem:
                p_lines.append(f"Problem: {_truncate_text(c.problem, 150)}")
            if c.tasks:
                task_strs = [
                    f"{t.value} [{t.status}]"
                    for t in c.tasks[: budget.max_conditions_per_card]
                ]
                p_lines.append(f"Tasks: {', '.join(task_strs)}")
            if c.modalities:
                mod_strs = [
                    f"{m.value} [{m.status}]"
                    for m in c.modalities[: budget.max_conditions_per_card]
                ]
                p_lines.append(f"Modalities: {', '.join(mod_strs)}")
            if c.evaluation_conditions:
                eval_strs = [
                    f"{e.value} [{e.status}]"
                    for e in c.evaluation_conditions[: budget.max_conditions_per_card]
                ]
                p_lines.append(f"Evaluation conditions: {', '.join(eval_strs)}")
            if c.methods:
                p_lines.append(f"Methods: {', '.join(c.methods[:4])}")
            if c.datasets:
                p_lines.append(f"Datasets: {', '.join(c.datasets[:3])}")
            if c.metrics:
                p_lines.append(f"Metrics: {', '.join(c.metrics[:3])}")
            if c.main_claims:
                claim_strs = [
                    f'"{_truncate_text(cl.claim, 100)}" [{cl.source_section or "Section"}]'
                    for cl in c.main_claims[: budget.max_claims_per_card]
                ]
                p_lines.append(f"Main claims: {'; '.join(claim_strs)}")
            if c.limitations:
                p_lines.append(f"Limitations: {', '.join(c.limitations[:3])}")
            if c.failure_cases:
                p_lines.append(f"Failure cases: {', '.join(c.failure_cases[:2])}")

        item_text = "\n".join(p_lines)
        blocks.append(_truncate_text(item_text, budget.max_chars_per_evidence_item))

    # 3. Gaps & Critic Reviews
    for g in ctx.retrieved_gaps[: budget.max_gaps]:
        g_lines = [
            f"--- CandidateGap {g.id} ---",
            f"Title: {g.title}",
            f"Type: {g.gap_type}",
            f"Review Status: {g.review_status}",
            f"Confidence: {g.confidence}",
            f"RQ: {_truncate_text(g.research_question, 200)}",
            f"Description: {_truncate_text(g.description, 200)}",
        ]
        if g.id in proj_gap_map:
            link = proj_gap_map[g.id]
            status_note = (
                "PAST RESOLVED/REJECTED GAP - Do NOT present as active open gap"
                if link.status in ("resolved", "rejected")
                else "ACTIVE PROJECT GAP"
            )
            g_lines.append(f"Project Gap Link Status: {link.status} ({status_note})")

        if g.id in ctx.critic_reviews:
            rev = ctx.critic_reviews[g.id]
            g_lines.append(
                f"Latest Critic Review (v{rev.review_version}): decision={rev.decision}, "
                f"rationale={_truncate_text(rev.rationale, 150)}, "
                f"caveats={', '.join(rev.caveats[:2])}"
            )

        item_text = "\n".join(g_lines)
        blocks.append(_truncate_text(item_text, budget.max_chars_per_evidence_item))

    full_packet = "\n\n".join(blocks)
    return _truncate_text(full_packet, budget.max_total_context_chars)


class AskService:
    """Retrieve memory context and prompt vendor-neutral LLM for evidence-bounded Q&A."""

    def __init__(
        self,
        repository: ResearchRepository,
        llm_provider: LLMProvider | None = None,
        budget: AskBudget | None = None,
    ) -> None:
        self._repository = repository
        self._llm_provider = llm_provider
        self._budget = budget or AskBudget()

    def build_ask_context(
        self,
        question: str,
        *,
        project_id_or_name: str | None = None,
        max_evidence: int = 10,
    ) -> AskContext:
        """Deterministically retrieve and assemble AskContext."""

        q_tokens = _tokenize(question)
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

        # 1. Collect candidate papers
        candidate_papers_dict: dict[str, StoredPaper] = {}
        for pid in proj_paper_map:
            p = self._repository.get_paper(pid)
            if p is not None:
                candidate_papers_dict[p.id] = p

        global_papers = self._repository.search_papers(query=question, limit=max_evidence * 3)
        for p in global_papers:
            if p.id not in candidate_papers_dict:
                candidate_papers_dict[p.id] = p

        # 2. Collect paper cards
        candidate_cards_dict: dict[str, PaperCard] = {}
        for pid in candidate_papers_dict:
            card = self._repository.get_paper_card(pid)
            if card is not None:
                candidate_cards_dict[pid] = card

        # 3. Score papers with relevance gate and bounded project bonus
        relation_bonus_map = {
            "seed": 8.0,
            "supporting": 8.0,
            "conflicting": 8.0,
            "relevant": 5.0,
            "background": 2.0,
        }

        scored_papers: list[tuple[float, StoredPaper]] = []
        for pid, paper in candidate_papers_dict.items():
            card = candidate_cards_dict.get(pid)
            c_tokens: set[str] = set()
            if card is not None:
                c_values: list[str] = [card.problem or "", card.motivation or ""]
                c_values.extend(card.methods)
                c_values.extend(card.datasets)
                c_values.extend(card.metrics)
                c_values.extend(card.limitations)
                c_values.extend(card.failure_cases)
                for t in card.tasks:
                    c_values.append(t.value)
                for m in card.modalities:
                    c_values.append(m.value)
                for ec in card.evaluation_conditions:
                    c_values.append(ec.value)
                for cl in card.main_claims:
                    c_values.append(cl.claim)
                c_tokens = _tokenize(" ".join(c_values))

            title_tokens = _tokenize(paper.title)
            abstract_tokens = _tokenize(paper.abstract or "")
            author_tokens = _tokenize(" ".join(paper.authors))

            title_hits = len(q_tokens & title_tokens)
            abstract_hits = len(q_tokens & abstract_tokens)
            card_hits = len(q_tokens & c_tokens)
            author_hits = len(q_tokens & author_tokens)

            lexical_score = (
                (3.0 * title_hits) + (2.0 * abstract_hits) + (1.0 * card_hits) + (1.0 * author_hits)
            )

            # Phase 1 Gate: If lexical_score == 0, exclude paper from evidence retrieval
            if lexical_score == 0.0:
                continue

            project_boost = 0.0
            if pid in proj_paper_map:
                rel = proj_paper_map[pid].relation
                project_boost = relation_bonus_map.get(rel, 2.0)

            total_score = (lexical_score * 2.0) + project_boost
            scored_papers.append((total_score, paper))

        scored_papers.sort(key=lambda x: x[0], reverse=True)
        top_papers = [p for _, p in scored_papers[: min(max_evidence, self._budget.max_papers)]]
        top_cards = [
            candidate_cards_dict[p.id] for p in top_papers if p.id in candidate_cards_dict
        ]

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
            g_title_tokens = _tokenize(gap.title)
            g_desc_tokens = _tokenize(gap.description)
            g_rq_tokens = _tokenize(gap.research_question)

            title_hits = len(q_tokens & g_title_tokens)
            desc_hits = len(q_tokens & g_desc_tokens)
            rq_hits = len(q_tokens & g_rq_tokens)

            lexical_score = (3.0 * title_hits) + (2.0 * desc_hits) + (2.0 * rq_hits)

            # Gate: If lexical_score == 0, exclude gap
            if lexical_score == 0.0:
                continue

            project_boost = 0.0
            if gid in proj_gap_map:
                st = proj_gap_map[gid].status
                if st in ("active", "interesting"):
                    project_boost = 4.0
                elif st in ("rejected", "resolved"):
                    project_boost = 0.0
                else:
                    project_boost = 2.0

            total_score = (lexical_score * 2.0) + project_boost
            scored_gaps.append((total_score, gap))

        scored_gaps.sort(key=lambda x: x[0], reverse=True)
        top_gaps = [g for _, g in scored_gaps[: min(max_evidence, self._budget.max_gaps)]]

        # 5. Fetch latest Critic Review for top gaps
        critic_reviews: dict[str, CriticReview] = {}
        for g in top_gaps:
            reviews = self._repository.list_critic_reviews(g.id)
            if reviews:
                critic_reviews[g.id] = reviews[-1]

        # 6. Assemble AskContext
        return AskContext(
            query=question,
            project=project,
            retrieved_papers=top_papers,
            retrieved_cards=top_cards,
            retrieved_gaps=top_gaps,
            critic_reviews=critic_reviews,
            project_paper_links=[
                proj_paper_map[p.id] for p in top_papers if p.id in proj_paper_map
            ],
            project_gap_links=[
                proj_gap_map[g.id] for g in top_gaps if g.id in proj_gap_map
            ],
            hypotheses=list(project.hypotheses) if project else [],
            constraints=list(project.constraints) if project else [],
            rejected_ideas=list(project.rejected_ideas) if project else [],
        )

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

        # Build AskContext as single source of truth
        ctx = self.build_ask_context(
            question,
            project_id_or_name=project_id_or_name,
            max_evidence=max_evidence,
        )

        allowed_paper_ids = ctx.allowed_paper_ids
        allowed_gap_ids = ctx.allowed_gap_ids

        # Check if memory has any evidence or project definition
        if not ctx.retrieved_papers and not ctx.retrieved_gaps and not ctx.project:
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
            # Deterministic fallback synthesis directly from AskContext
            lines: list[str] = [
                "Based on the analyzed project corpus currently stored in ResearchRadar:"
            ]
            if ctx.project:
                lines.append(
                    f"• Project Scope: '{ctx.project.name}' (Goal: {ctx.project.goal or 'N/A'})"
                )
                if ctx.rejected_ideas:
                    lines.append(
                        f"• Project Rejected Ideas (History): {', '.join(ctx.rejected_ideas)}"
                    )
            if ctx.retrieved_papers:
                lines.append(
                    f"• Found {len(ctx.retrieved_papers)} matching stored paper(s): "
                    + ", ".join(f"'{p.title}'" for p in ctx.retrieved_papers[:3])
                )
            if ctx.retrieved_gaps:
                lines.append(
                    f"• Found {len(ctx.retrieved_gaps)} matching candidate gap(s): "
                    + ", ".join(f"'{g.title}'" for g in ctx.retrieved_gaps[:3])
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

        # Build evidence packet from AskContext using budget
        prompt_evidence = format_evidence_packet(ctx, self._budget)

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

            # Phase 3: Source ID Safety Validation against AskContext allowed sets
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
        except Exception:
            logger.exception("LLM Q&A generation failed.")
            return AskResponse(
                answer="I couldn't synthesize an answer from the stored evidence right now.",
                referenced_paper_ids=list(dict.fromkeys(allowed_paper_ids)),
                referenced_gap_ids=list(dict.fromkeys(allowed_gap_ids)),
                is_sufficient_evidence=False,
            )
