"""Deterministic Method-Transfer Gap miner over analyzed PaperCards."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from research_radar.gap.miner import (
    enforce_language_safety,
    generate_gap_id,
)
from research_radar.models.gap import CandidateGap, EvidenceRef, GapProvenance
from research_radar.storage.repositories import ScopedCorpusResult


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")[:20] or "item"


HIGHLY_SPECIFIC_DETECTOR_METHODS = {
    "yolo",
    "faster-rcnn",
    "retinanet",
    "ssd",
    "mask-rcnn",
    "bounding-box detector head",
    "yolov8",
    "yolov5",
    "detr",
    "box head",
    "region proposal network",
    "rpn",
}

RECONSTRUCTION_TASKS = {
    "reconstruction",
    "mri reconstruction",
    "ct reconstruction",
    "image reconstruction",
}


@dataclass(frozen=True, slots=True)
class FeasibilityResult:
    """Result of method transfer feasibility assessment."""

    is_feasible: bool
    score: float
    reasoning: str


def assess_transfer_feasibility(
    method: str, source_context: str, target_context: str
) -> FeasibilityResult:
    """Determine deterministic feasibility of transferring method to target_context."""

    m_lower = method.lower()
    src_lower = source_context.lower()
    tgt_lower = target_context.lower()

    if src_lower == tgt_lower:
        return FeasibilityResult(
            is_feasible=False,
            score=0.0,
            reasoning="Source and target context are identical.",
        )

    # Rule 1: Object detection heads cannot transfer to reconstruction
    if any(det in m_lower for det in HIGHLY_SPECIFIC_DETECTOR_METHODS):
        if any(rec in tgt_lower for rec in RECONSTRUCTION_TASKS):
            return FeasibilityResult(
                is_feasible=False,
                score=0.0,
                reasoning="Detection box heads are incompatible with pixel/k-space reconstruction.",
            )

    score = 0.8
    return FeasibilityResult(
        is_feasible=True,
        score=score,
        reasoning=(
            f"Method '{method}' transfer from '{source_context}' to "
            f"'{target_context}' is plausible."
        ),
    )


class MethodTransferGapMiner:
    """Mine evidence-backed method transfer opportunities across scoped PaperCards."""

    def mine_transfer_gaps(
        self, topic: str, corpus: ScopedCorpusResult
    ) -> list[CandidateGap]:
        """Mine method transfer candidates with source & target representation."""

        cards = corpus.cards
        if len(cards) < 4:
            return []

        paper_title_map = {p.id: p.title for p in corpus.papers}

        method_source_papers: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        context_corpus_papers: dict[str, set[str]] = defaultdict(set)

        for stored_card in cards:
            card = stored_card.card
            pid = card.paper_id

            card_contexts: set[str] = set()
            for t in card.tasks:
                if t.status == "observed" and t.value.strip():
                    card_contexts.add(t.value.strip().lower())
            if card.problem and card.problem.strip():
                card_contexts.add(card.problem.strip().lower())

            for m in card.modalities:
                if m.status == "observed" and m.value.strip():
                    card_contexts.add(m.value.strip().lower())

            for c in card.evaluation_conditions:
                if c.status == "observed" and c.value.strip():
                    card_contexts.add(c.value.strip().lower())

            for ctx in card_contexts:
                context_corpus_papers[ctx].add(pid)
                for method in card.methods:
                    clean_m = " ".join(method.split())
                    if len(clean_m) >= 2:
                        method_source_papers[clean_m][ctx].add(pid)

        candidates: list[CandidateGap] = []
        now = _utc_now()

        for method, ctx_map in method_source_papers.items():
            for src_ctx, src_pids in ctx_map.items():
                if len(src_pids) < 2:
                    continue

                for tgt_ctx, tgt_corpus_pids in context_corpus_papers.items():
                    if src_ctx == tgt_ctx:
                        continue
                    if len(tgt_corpus_pids) < 2:
                        continue

                    used_in_target = ctx_map.get(tgt_ctx, set())
                    if used_in_target:
                        continue

                    feasibility = assess_transfer_feasibility(
                        method, src_ctx, tgt_ctx
                    )
                    if not feasibility.is_feasible:
                        continue

                    gap_id = generate_gap_id(
                        "method_transfer",
                        topic,
                        f"{_slug(method)}-{_slug(src_ctx)}-to-{_slug(tgt_ctx)}",
                    )

                    title = enforce_language_safety(
                        f"Potential method transfer: {method} → {tgt_ctx}"
                    )
                    desc_text = (
                        f"Within the retrieved corpus of {topic}, method '{method}' has "
                        f"evidence in source context '{src_ctx}' ({len(src_pids)} paper(s)), "
                        f"while limited evidence was retrieved for target context '{tgt_ctx}' "
                        f"despite its representation across {len(tgt_corpus_pids)} paper(s)."
                    )
                    description = enforce_language_safety(desc_text)
                    rq_text = (
                        f"Within the retrieved corpus of {topic}, can {method}, which shows "
                        f"evidence in '{src_ctx}', improve performance or robustness in "
                        f"'{tgt_ctx}'?"
                    )
                    research_question = enforce_language_safety(rq_text)

                    supporting_evidence: list[EvidenceRef] = []
                    for pid in sorted(src_pids)[:2]:
                        paper_title = paper_title_map.get(pid, f"Paper {pid}")
                        supporting_evidence.append(
                            EvidenceRef(
                                paper_id=pid,
                                paper_title=paper_title,
                                evidence_kind="supporting",
                                claim_or_field="methods",
                                supporting_text=(
                                    f"Method '{method}' demonstrated in source context '{src_ctx}' "
                                    f"in {paper_title}"
                                ),
                            )
                        )

                    for pid in sorted(tgt_corpus_pids)[:2]:
                        paper_title = paper_title_map.get(pid, f"Paper {pid}")
                        supporting_evidence.append(
                            EvidenceRef(
                                paper_id=pid,
                                paper_title=paper_title,
                                evidence_kind="supporting",
                                claim_or_field="tasks",
                                supporting_text=(
                                    f"Target context '{tgt_ctx}' represented in {paper_title}"
                                ),
                            )
                        )

                    provenance = GapProvenance(
                        retrievals=[],
                        corpus_paper_ids=list(corpus.corpus_paper_ids),
                        corpus_description=(
                            f"Method-transfer analysis for '{topic}' across {len(cards)} cards"
                        ),
                        supporting_evidence=supporting_evidence,
                        conflicting_evidence=[],
                    )

                    caveats = [
                        "This is a transfer hypothesis, not a verified novelty claim.",
                        (
                            f"Method '{method}' is established in '{src_ctx}', but no combination "
                            f"with '{tgt_ctx}' was retrieved in the scoped corpus."
                        ),
                        (
                            f"Analysis is bounded by the retrieved corpus of "
                            f"{len(corpus.corpus_paper_ids)} stored papers."
                        ),
                    ]

                    confidence = round(min(0.8, feasibility.score * 0.8), 2)

                    candidates.append(
                        CandidateGap(
                            id=gap_id,
                            title=title,
                            description=description,
                            gap_type="method_transfer",
                            research_question=research_question,
                            supporting_papers=sorted(src_pids | tgt_corpus_pids),
                            conflicting_papers=[],
                            evidence_count=len(supporting_evidence),
                            novelty_score=None,
                            evidence_score=0.6,
                            importance_score=None,
                            feasibility_score=feasibility.score,
                            confidence=confidence,
                            search_scope=f"Method Transfer Scoper over {len(cards)} PaperCards",
                            caveats=caveats,
                            provenance=provenance,
                            review_status="candidate",
                            created_at=now,
                        )
                    )

        return candidates
