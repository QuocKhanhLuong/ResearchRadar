"""Deterministic contradiction gap miner over analyzed PaperCard claims and structured context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from research_radar.gap.miner import (
    _tokenize,
    enforce_language_safety,
    generate_gap_id,
)
from research_radar.models.gap import CandidateGap, EvidenceRef, GapProvenance
from research_radar.models.paper_card import PaperCard
from research_radar.storage.repositories import ScopedCorpusResult


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")[:20] or "item"


# Polarity directional phrases
POSITIVE_PATTERNS = (
    r"\b(improves?|improving|improved|enhances?|enhancing|enhanced|outperforms?|outperformed|boosts?|superior|better)\b",
    r"\b(reduces?|lowers?)\s+(error|loss|uncertainty|latency|cost)\b",
)

NEGATIVE_PATTERNS = (
    r"\b(degrades?|degrading|degraded|worsens?|worsening|underperforms?|underperformed|hurts?|inferior|worse|fails\s+to\s+improve)\b",
    r"\b(increases?|higher)\s+(error|loss|uncertainty|latency|cost)\b",
)

TASK_CATEGORIES = {"reconstruction", "segmentation", "classification", "detection", "registration"}
MODALITY_CATEGORIES = {"mri", "ct", "xray", "ultrasound", "pet", "spect"}

ACCURACY_METRICS = {"dice", "hd95", "iou", "psnr", "ssim", "mae", "rmse", "auc", "accuracy", "f1"}
COMPUTATION_METRICS = {"latency", "fps", "flops", "memory", "runtime", "throughput", "time"}

CONTRADICTION_GENERIC_TERMS = {
    "model", "models", "performance", "results", "result", "method", "methods",
    "paper", "papers", "approach", "approaches", "study", "studies", "data",
    "analysis", "evaluation", "evaluations", "system", "systems", "proposed",
    "algorithm", "algorithms", "technique", "techniques", "experiment",
    "experiments", "accuracy", "score", "scores", "using", "used", "based",
    "show", "shows", "shown", "table", "figure", "fig", "section", "compare",
    "compared", "findings", "finding", "work", "works", "test", "testing",
    "tested", "value", "values", "high", "low", "new", "our", "their", "baseline",
    "reported", "observed", "evaluated", "quality", "effect", "effects",
}


@dataclass(frozen=True, slots=True)
class ContextCompatibilityResult:
    """Structured compatibility assessment between two PaperCards."""

    status: Literal["compatible", "partially_compatible", "incompatible"]
    score: float
    rationale: str
    is_different_metric: bool = False
    is_different_condition: bool = False
    is_different_dataset: bool = False


def _detect_claim_polarity(claim_text: str) -> str:
    """Classify claim polarity as 'positive', 'negative', or 'neutral'."""

    text = claim_text.lower()
    is_pos = any(re.search(p, text) for p in POSITIVE_PATTERNS)
    is_neg = any(re.search(p, text) for p in NEGATIVE_PATTERNS)

    if is_pos and not is_neg:
        return "positive"
    if is_neg and not is_pos:
        return "negative"
    return "neutral"


def assess_context_compatibility(
    card_a: PaperCard, card_b: PaperCard
) -> ContextCompatibilityResult:
    """Assess context compatibility across task, modality, metric, condition, and dataset."""

    # 1. Task check (hard mismatch check)
    tasks_a = {t.value.lower() for t in card_a.tasks if t.status == "observed"}
    if card_a.problem:
        tasks_a.add(card_a.problem.lower())

    tasks_b = {t.value.lower() for t in card_b.tasks if t.status == "observed"}
    if card_b.problem:
        tasks_b.add(card_b.problem.lower())

    terms_a = set(re.findall(r"\b[a-zA-Z0-9]{2,}\b", " ".join(tasks_a))) & TASK_CATEGORIES
    terms_b = set(re.findall(r"\b[a-zA-Z0-9]{2,}\b", " ".join(tasks_b))) & TASK_CATEGORIES

    if terms_a and terms_b and not (terms_a & terms_b):
        return ContextCompatibilityResult(
            status="incompatible",
            score=0.0,
            rationale="Hard task mismatch (e.g. segmentation vs reconstruction).",
        )

    # 2. Modality check (hard mismatch check)
    mods_a = {m.value.lower() for m in card_a.modalities if m.status == "observed"}
    mods_b = {m.value.lower() for m in card_b.modalities if m.status == "observed"}

    m_terms_a = set(re.findall(r"\b[a-zA-Z0-9]{2,}\b", " ".join(mods_a))) & MODALITY_CATEGORIES
    m_terms_b = set(re.findall(r"\b[a-zA-Z0-9]{2,}\b", " ".join(mods_b))) & MODALITY_CATEGORIES

    if m_terms_a and m_terms_b and not (m_terms_a & m_terms_b):
        return ContextCompatibilityResult(
            status="incompatible", score=0.0, rationale="Hard modality mismatch (e.g. MRI vs CT)."
        )

    # 3. Metric / Measured Quantity check
    metrics_a = {m.lower() for m in card_a.metrics}
    metrics_b = {m.lower() for m in card_b.metrics}

    has_acc_a = bool(metrics_a & ACCURACY_METRICS)
    has_acc_b = bool(metrics_b & ACCURACY_METRICS)
    has_comp_a = bool(metrics_a & COMPUTATION_METRICS)
    has_comp_b = bool(metrics_b & COMPUTATION_METRICS)

    if (has_acc_a and not has_comp_a and has_comp_b and not has_acc_b) or (
        has_comp_a and not has_acc_a and has_acc_b and not has_comp_b
    ):
        return ContextCompatibilityResult(
            status="incompatible",
            score=0.0,
            rationale="Metric mismatch: accuracy/quality metric vs latency/throughput metric.",
            is_different_metric=True,
        )

    # 4. Evaluation condition & dataset checks
    conds_a = {c.value.lower() for c in card_a.evaluation_conditions if c.status == "observed"}
    conds_b = {c.value.lower() for c in card_b.evaluation_conditions if c.status == "observed"}

    datasets_a = {d.lower() for d in card_a.datasets}
    datasets_b = {d.lower() for d in card_b.datasets}

    is_diff_cond = bool(conds_a and conds_b and not (conds_a & conds_b))
    is_diff_ds = bool(datasets_a and datasets_b and not (datasets_a & datasets_b))

    score = 1.0
    if not tasks_a or not tasks_b:
        score -= 0.15
    if not mods_a or not mods_b:
        score -= 0.15

    if is_diff_cond or is_diff_ds:
        score -= 0.2
        return ContextCompatibilityResult(
            status="partially_compatible",
            score=max(0.4, round(score, 2)),
            rationale="Differing evaluation conditions or datasets.",
            is_different_condition=is_diff_cond,
            is_different_dataset=is_diff_ds,
        )

    return ContextCompatibilityResult(
        status="compatible",
        score=max(0.6, round(score, 2)),
        rationale="Compatible task, modality, and evaluation context.",
    )


def calculate_context_compatibility(card_a: PaperCard, card_b: PaperCard) -> float:
    """Calculate compatibility score between two PaperCards (0.0 to 1.0)."""

    return assess_context_compatibility(card_a, card_b).score


def _extract_domain_tokens(text: str) -> set[str]:
    """Tokenize text and remove both standard stopwords and contradiction generic terms."""

    tokens = _tokenize(text)
    return {t for t in tokens if t not in CONTRADICTION_GENERIC_TERMS and len(t) >= 3}


class ContradictionGapMiner:
    """Mine conflicting claims across context-compatible PaperCards."""

    def mine_contradiction_gaps(
        self, topic: str, corpus: ScopedCorpusResult
    ) -> list[CandidateGap]:
        """Detect pairwise contradictory claims across analyzed PaperCards."""

        cards = corpus.cards
        if len(cards) < 2:
            return []

        paper_title_map = {p.id: p.title for p in corpus.papers}
        candidates: list[CandidateGap] = []
        now = _utc_now()
        seen_pairs: set[str] = set()

        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                stored_a = cards[i]
                stored_b = cards[j]
                card_a = stored_a.card
                card_b = stored_b.card

                if card_a.paper_id == card_b.paper_id:
                    continue

                compatibility = assess_context_compatibility(card_a, card_b)
                if compatibility.status == "incompatible" or compatibility.score < 0.4:
                    continue

                claims_a: list[tuple[str, str | None, str | None]] = []
                for c in card_a.main_claims:
                    claims_a.append((c.claim, c.source_section, c.supporting_text))
                for contrib in card_a.contributions:
                    claims_a.append((contrib, "contributions", contrib))

                claims_b: list[tuple[str, str | None, str | None]] = []
                for c in card_b.main_claims:
                    claims_b.append((c.claim, c.source_section, c.supporting_text))
                for contrib in card_b.contributions:
                    claims_b.append((contrib, "contributions", contrib))

                for text_a, sec_a, supp_a in claims_a:
                    pol_a = _detect_claim_polarity(text_a)
                    if pol_a == "neutral":
                        continue

                    tokens_a = _extract_domain_tokens(text_a)

                    for text_b, sec_b, supp_b in claims_b:
                        pol_b = _detect_claim_polarity(text_b)
                        if pol_b == "neutral" or pol_a == pol_b:
                            continue  # Must be opposite polarity!

                        tokens_b = _extract_domain_tokens(text_b)
                        overlap_tokens = tokens_a & tokens_b

                        # Must share core domain concepts (at least 1 specific non-generic term)
                        if len(overlap_tokens) < 1:
                            continue

                        concept_phrase = " ".join(sorted(overlap_tokens)[:3])
                        id_a_min = min(card_a.paper_id, card_b.paper_id)
                        id_b_max = max(card_a.paper_id, card_b.paper_id)
                        pair_key = f"{id_a_min}:{id_b_max}:{concept_phrase}"
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)

                        is_context_conditioned = (
                            compatibility.status == "partially_compatible"
                            or compatibility.is_different_condition
                            or compatibility.is_different_dataset
                        )

                        gap_id = generate_gap_id(
                            "contradiction",
                            topic,
                            f"{_slug(concept_phrase)}-{_slug(card_a.paper_id)}-{_slug(card_b.paper_id)}",
                        )

                        if is_context_conditioned:
                            title = enforce_language_safety(
                                f"Context-conditioned disagreement on {concept_phrase} in {topic}"
                            )
                            rq_text = (
                                f"Within the retrieved corpus of {topic}, under which experimental "
                                f"conditions (e.g., across datasets or evaluation settings) does "
                                f"{concept_phrase} improve or degrade performance?"
                            )
                            research_question = enforce_language_safety(rq_text)
                        else:
                            title = enforce_language_safety(
                                f"Conflicting evidence on {concept_phrase} in {topic}"
                            )
                            rq_text = (
                                f"Within the retrieved corpus of {topic}, why do paper "
                                f"'{card_a.paper_id}' and paper '{card_b.paper_id}' report "
                                f"opposing outcomes for {concept_phrase} under similar conditions?"
                            )
                            research_question = enforce_language_safety(rq_text)

                        if is_context_conditioned:
                            desc_text = (
                                f"Within the retrieved corpus of {topic}, paper "
                                f"'{card_a.paper_id}' and paper '{card_b.paper_id}' observed "
                                f"opposing outcomes under differing reported datasets or "
                                f"evaluation conditions."
                            )
                        else:
                            desc_text = (
                                f"Within the retrieved corpus of {topic}, paper "
                                f"'{card_a.paper_id}' and paper '{card_b.paper_id}' report "
                                f"contradictory findings regarding '{concept_phrase}' under "
                                f"similar reported conditions."
                            )
                        description = enforce_language_safety(desc_text)

                        title_a = paper_title_map.get(card_a.paper_id, f"Paper {card_a.paper_id}")
                        title_b = paper_title_map.get(card_b.paper_id, f"Paper {card_b.paper_id}")

                        ref_a = EvidenceRef(
                            paper_id=card_a.paper_id,
                            paper_title=title_a,
                            evidence_kind="supporting",
                            claim_or_field="main_claims",
                            source_section=sec_a,
                            supporting_text=supp_a or text_a,
                        )
                        ref_b = EvidenceRef(
                            paper_id=card_b.paper_id,
                            paper_title=title_b,
                            evidence_kind="conflicting",
                            claim_or_field="main_claims",
                            source_section=sec_b,
                            supporting_text=supp_b or text_b,
                        )

                        kind_str = (
                            "context_conditioned_disagreement"
                            if is_context_conditioned
                            else "direct_contradiction"
                        )
                        provenance = GapProvenance(
                            retrievals=[],
                            corpus_paper_ids=list(corpus.corpus_paper_ids),
                            corpus_description=(
                                f"Contradiction [{kind_str}] for '{topic}' across "
                                f"{len(cards)} cards"
                            ),
                            supporting_evidence=[ref_a],
                            conflicting_evidence=[ref_b],
                        )

                        caveats = [
                            "This is a candidate research question, not a verified novelty claim.",
                            (
                                "Within the retrieved corpus, these findings appear inconsistent "
                                "under reported conditions."
                            ),
                        ]
                        if is_context_conditioned:
                            caveats.append(
                                "This represents a context-conditioned disagreement (e.g. "
                                "differing datasets or evaluation settings) rather than a "
                                "direct protocol contradiction."
                            )
                        caveats.append(
                            f"Analysis is bounded by the retrieved corpus of "
                            f"{len(corpus.corpus_paper_ids)} stored papers."
                        )

                        confidence = round(min(0.85, compatibility.score * 0.75), 2)

                        candidates.append(
                            CandidateGap(
                                id=gap_id,
                                title=title,
                                description=description,
                                gap_type="contradiction",
                                research_question=research_question,
                                supporting_papers=[card_a.paper_id],
                                conflicting_papers=[card_b.paper_id],
                                evidence_count=2,
                                novelty_score=None,
                                evidence_score=0.6,
                                importance_score=None,
                                feasibility_score=None,
                                confidence=confidence,
                                search_scope=f"Contradiction Matrix over {len(cards)} PaperCards",
                                caveats=caveats,
                                provenance=provenance,
                                review_status="candidate",
                                created_at=now,
                            )
                        )

        return candidates
