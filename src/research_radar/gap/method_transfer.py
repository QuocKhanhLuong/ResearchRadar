"""Deterministic Method-Transfer Gap miner over analyzed PaperCards with typed contexts."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

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


TransferDimension = Literal["task", "modality", "dataset", "evaluation_condition"]


@dataclass(frozen=True, slots=True)
class TransferContext:
    """Typed context entry for method transfer reasoning."""

    dimension: TransferDimension
    value: str
    paper_id: str
    source_section: str | None = None
    supporting_text: str | None = None


MethodClass = Literal[
    "generic_regularizer",
    "representation_method",
    "architecture_component",
    "task_specific_head",
    "loss_objective",
    "training_strategy",
    "unknown",
]


def infer_method_class(method: str) -> MethodClass:
    """Infer method class using inspectable rules."""

    m = method.lower().strip()

    # 1. Task-specific heads
    if any(
        kw in m
        for kw in (
            "head",
            "yolo",
            "faster-rcnn",
            "retinanet",
            "ssd",
            "mask-rcnn",
            "rpn",
            "box",
            "detector",
            "classification head",
            "segmentation decoder",
        )
    ):
        return "task_specific_head"

    # 2. Generic regularizers
    if any(
        kw in m
        for kw in (
            "spectral regularization",
            "regularizer",
            "regularization",
            "dropout",
            "weight decay",
            "batch normalization",
            "data augmentation",
            "augmentation",
        )
    ):
        return "generic_regularizer"

    # 3. Representation methods
    if any(
        kw in m
        for kw in (
            "contrastive learning",
            "domain adaptation",
            "self-supervised",
            "masked autoencoder",
            "representation learning",
            "simclr",
            "byol",
        )
    ):
        return "representation_method"

    # 4. Architecture components
    if any(
        kw in m
        for kw in (
            "u-net",
            "unet",
            "transformer",
            "attention",
            "resnet",
            "encoder",
            "decoder",
            "lora",
            "vit",
        )
    ):
        return "architecture_component"

    # 5. Loss objectives
    if any(kw in m for kw in ("loss", "objective", "focal", "dice loss", "cross-entropy")):
        return "loss_objective"

    # 6. Training strategies
    if any(
        kw in m
        for kw in (
            "distillation",
            "curriculum",
            "meta-learning",
            "adversarial training",
        )
    ):
        return "training_strategy"

    return "unknown"


@dataclass(frozen=True, slots=True)
class FeasibilityResult:
    """Result of method transfer feasibility assessment."""

    is_feasible: bool
    score: float
    reasoning: str
    method_class: MethodClass


def assess_transfer_feasibility(
    method: str,
    source_dimension: TransferDimension,
    source_context: str,
    target_dimension: TransferDimension,
    target_context: str,
) -> FeasibilityResult:
    """Determine feasibility using method_class x transfer_dimension heuristics."""

    if source_dimension != target_dimension:
        return FeasibilityResult(
            is_feasible=False,
            score=0.0,
            reasoning=(
                f"Incompatible transfer dimensions ({source_dimension} vs {target_dimension})."
            ),
            method_class="unknown",
        )

    if source_context.lower() == target_context.lower():
        return FeasibilityResult(
            is_feasible=False,
            score=0.0,
            reasoning="Source and target context are identical.",
            method_class="unknown",
        )

    m_class = infer_method_class(method)

    if m_class == "task_specific_head":
        if source_dimension == "task":
            return FeasibilityResult(
                is_feasible=False,
                score=0.0,
                reasoning=f"Task-specific head '{method}' cannot be transferred cross-task.",
                method_class=m_class,
            )
        return FeasibilityResult(
            is_feasible=False,
            score=0.0,
            reasoning=f"Task-specific head '{method}' has restricted transferability.",
            method_class=m_class,
        )

    if m_class == "generic_regularizer":
        return FeasibilityResult(
            is_feasible=True,
            score=0.8,
            reasoning=(
                f"Generic regularizer '{method}' transfer across {source_dimension}s is plausible."
            ),
            method_class=m_class,
        )

    if m_class == "representation_method":
        score = 0.8 if source_dimension in ("task", "modality", "dataset") else 0.75
        return FeasibilityResult(
            is_feasible=True,
            score=score,
            reasoning=(
                f"Representation method '{method}' transfer across {source_dimension}s "
                f"is plausible."
            ),
            method_class=m_class,
        )

    if m_class == "architecture_component":
        score = 0.75 if source_dimension in ("task", "modality") else 0.7
        return FeasibilityResult(
            is_feasible=True,
            score=score,
            reasoning=(
                f"Architecture component '{method}' transfer across {source_dimension}s "
                f"is plausible."
            ),
            method_class=m_class,
        )

    if m_class == "loss_objective":
        score = 0.7 if source_dimension == "task" else 0.5
        return FeasibilityResult(
            is_feasible=True,
            score=score,
            reasoning=(
                f"Loss objective '{method}' transfer across {source_dimension}s is plausible."
            ),
            method_class=m_class,
        )

    if m_class == "training_strategy":
        score = 0.75 if source_dimension in ("task", "dataset") else 0.65
        return FeasibilityResult(
            is_feasible=True,
            score=score,
            reasoning=(
                f"Training strategy '{method}' transfer across {source_dimension}s is plausible."
            ),
            method_class=m_class,
        )

    # Unknown method class: lower confidence, never assume strong feasibility
    return FeasibilityResult(
        is_feasible=True,
        score=0.5,
        reasoning=(
            f"Method '{method}' has unknown class; transfer assigned lower baseline confidence."
        ),
        method_class="unknown",
    )


class MethodTransferGapMiner:
    """Mine evidence-backed method transfer opportunities across scoped PaperCards."""

    def mine_transfer_gaps(self, topic: str, corpus: ScopedCorpusResult) -> list[CandidateGap]:
        """Mine method transfer candidates with typed contexts and extraction safety."""

        cards = corpus.cards
        if len(cards) < 4:
            return []

        paper_title_map = {p.id: p.title for p in corpus.papers}
        total_cards = len(cards)

        # Calculate extraction completeness per dimension
        dimension_observed_counts: dict[TransferDimension, int] = defaultdict(int)
        for stored in cards:
            card = stored.card
            if (
                any(t.status in ("observed", "explicitly_absent") for t in card.tasks)
                or card.problem
            ):
                dimension_observed_counts["task"] += 1
            if any(m.status in ("observed", "explicitly_absent") for m in card.modalities):
                dimension_observed_counts["modality"] += 1
            if card.datasets:
                dimension_observed_counts["dataset"] += 1
            if any(
                c.status in ("observed", "explicitly_absent") for c in card.evaluation_conditions
            ):
                dimension_observed_counts["evaluation_condition"] += 1

        extraction_coverage = {
            dim: (dimension_observed_counts[dim] / total_cards) if total_cards > 0 else 0.0
            for dim in ("task", "modality", "dataset", "evaluation_condition")
        }

        # Build typed contexts: dimension -> value -> list[TransferContext]
        # method -> dimension -> value -> set[paper_id]
        method_contexts: dict[str, dict[TransferDimension, dict[str, list[TransferContext]]]] = (
            defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
        # dimension -> value -> list[TransferContext]
        corpus_contexts: dict[TransferDimension, dict[str, list[TransferContext]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for stored in cards:
            card = stored.card
            pid = card.paper_id

            card_typed_contexts: list[TransferContext] = []

            # 1. Tasks
            for t in card.tasks:
                if t.status == "observed" and t.value.strip():
                    val = t.value.strip().lower()
                    card_typed_contexts.append(
                        TransferContext(
                            dimension="task",
                            value=val,
                            paper_id=pid,
                            source_section=t.source_section,
                            supporting_text=t.supporting_text,
                        )
                    )
            if card.problem and card.problem.strip():
                val = card.problem.strip().lower()
                card_typed_contexts.append(
                    TransferContext(
                        dimension="task",
                        value=val,
                        paper_id=pid,
                        source_section="problem",
                        supporting_text=card.problem,
                    )
                )

            # 2. Modalities
            for m in card.modalities:
                if m.status == "observed" and m.value.strip():
                    val = m.value.strip().lower()
                    card_typed_contexts.append(
                        TransferContext(
                            dimension="modality",
                            value=val,
                            paper_id=pid,
                            source_section=m.source_section,
                            supporting_text=m.supporting_text,
                        )
                    )

            # 3. Datasets
            for d in card.datasets:
                if d.strip():
                    val = d.strip().lower()
                    card_typed_contexts.append(
                        TransferContext(
                            dimension="dataset",
                            value=val,
                            paper_id=pid,
                            source_section="datasets",
                            supporting_text=d,
                        )
                    )

            # 4. Evaluation Conditions
            for c in card.evaluation_conditions:
                if c.status == "observed" and c.value.strip():
                    val = c.value.strip().lower()
                    card_typed_contexts.append(
                        TransferContext(
                            dimension="evaluation_condition",
                            value=val,
                            paper_id=pid,
                            source_section=c.source_section,
                            supporting_text=c.supporting_text,
                        )
                    )

            for ctx in card_typed_contexts:
                corpus_contexts[ctx.dimension][ctx.value].append(ctx)
                for method in card.methods:
                    clean_m = " ".join(method.split())
                    if len(clean_m) >= 2:
                        method_contexts[clean_m][ctx.dimension][ctx.value].append(ctx)

        candidates: list[CandidateGap] = []
        now = _utc_now()

        for method, dim_map in method_contexts.items():
            for dim, val_map in dim_map.items():
                # Safety Check: Target dimension extraction coverage must be >= 50%
                if extraction_coverage.get(dim, 0.0) < 0.5:
                    continue

                for src_val, src_ctxs in val_map.items():
                    src_pids = {c.paper_id for c in src_ctxs}
                    if len(src_pids) < 2:
                        continue

                    # Look for target context in SAME dimension
                    for tgt_val, tgt_ctxs in corpus_contexts[dim].items():
                        if src_val == tgt_val:
                            continue
                        tgt_pids = {c.paper_id for c in tgt_ctxs}
                        if len(tgt_pids) < 2:
                            continue

                        # Check if method is already observed in target context
                        if tgt_val in val_map:
                            continue

                        feasibility = assess_transfer_feasibility(
                            method, dim, src_val, dim, tgt_val
                        )
                        if not feasibility.is_feasible:
                            continue

                        gap_id = generate_gap_id(
                            "method_transfer",
                            topic,
                            f"{_slug(method)}-{_slug(dim)}-{_slug(src_val)}-to-{_slug(tgt_val)}",
                        )

                        title = enforce_language_safety(
                            f"Potential method transfer: {method} → {tgt_val}"
                        )
                        desc_text = (
                            f"Within the retrieved corpus of {topic}, method '{method}' has "
                            f"evidence in source {dim} '{src_val}' ({len(src_pids)} paper(s)), "
                            f"while limited evidence was retrieved for target {dim} '{tgt_val}' "
                            f"despite its representation across {len(tgt_pids)} paper(s)."
                        )
                        description = enforce_language_safety(desc_text)
                        rq_text = (
                            f"Within the retrieved corpus of {topic}, can {method}, which shows "
                            f"evidence in source {dim} '{src_val}', improve performance or "
                            f"robustness in target {dim} '{tgt_val}'?"
                        )
                        research_question = enforce_language_safety(rq_text)

                        # Clean EvidenceRefs without quote fabrication or leakage
                        supporting_evidence: list[EvidenceRef] = []
                        # Add method evidence ref (from source cards)
                        # Do NOT reuse task/modality/dataset context evidence as method evidence
                        for pid in sorted(list(src_pids))[:2]:
                            paper_title = paper_title_map.get(pid, f"Paper {pid}")
                            supporting_evidence.append(
                                EvidenceRef(
                                    paper_id=pid,
                                    paper_title=paper_title,
                                    evidence_kind="supporting",
                                    claim_or_field="methods",
                                    source_section=None,
                                    supporting_text=None,
                                )
                            )

                        # Add target context evidence ref with correct plural field names
                        dim_field = {
                            "task": "tasks",
                            "modality": "modalities",
                            "dataset": "datasets",
                            "evaluation_condition": "evaluation_conditions",
                        }.get(dim, dim)

                        for ctx in tgt_ctxs[:2]:
                            paper_title = paper_title_map.get(ctx.paper_id, f"Paper {ctx.paper_id}")
                            supporting_evidence.append(
                                EvidenceRef(
                                    paper_id=ctx.paper_id,
                                    paper_title=paper_title,
                                    evidence_kind="supporting",
                                    claim_or_field=dim_field,
                                    source_section=ctx.source_section,
                                    supporting_text=ctx.supporting_text,
                                )
                            )

                        corpus_desc = (
                            f"Method-transfer [{method}:{dim}:{src_val}->{tgt_val}] "
                            f"analysis for '{topic}' across {total_cards} cards"
                        )
                        provenance = GapProvenance(
                            retrievals=[],
                            corpus_paper_ids=list(corpus.corpus_paper_ids),
                            corpus_description=corpus_desc,
                            supporting_evidence=supporting_evidence,
                            conflicting_evidence=[],
                        )

                        caveats = [
                            "This is a transfer hypothesis, not a verified novelty claim.",
                            (
                                f"Method '{method}' is established in source {dim} '{src_val}', "
                                f"but no combination with target {dim} '{tgt_val}' was retrieved "
                                f"in the scoped corpus."
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
                                supporting_papers=sorted(src_pids | tgt_pids),
                                conflicting_papers=[],
                                evidence_count=len(supporting_evidence),
                                novelty_score=None,
                                evidence_score=0.6,
                                importance_score=None,
                                feasibility_score=feasibility.score,
                                confidence=confidence,
                                search_scope=(
                                    f"Method Transfer Scoper over {total_cards} PaperCards"
                                ),
                                caveats=caveats,
                                provenance=provenance,
                                review_status="candidate",
                                created_at=now,
                            )
                        )

        return candidates
