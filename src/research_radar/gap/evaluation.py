"""Evaluation gap miner for detecting underrepresented evaluation conditions."""

from __future__ import annotations

import re
from collections import defaultdict
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
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")[:20] or "cond"


STANDARD_EVALUATION_CONDITIONS = (
    "domain shift",
    "scanner shift",
    "noise robustness",
    "calibration",
    "out-of-distribution robustness",
    "cross-dataset validation",
    "computational cost and latency",
)


class EvaluationGapMiner:
    """Mine underrepresented or missing evaluation conditions across scoped PaperCards."""

    def mine_evaluation_gaps(
        self, topic: str, corpus: ScopedCorpusResult
    ) -> list[CandidateGap]:
        """Identify evaluation conditions rarely or never evaluated in the scoped corpus."""

        total_cards_count = len(corpus.cards)
        if total_cards_count < 2:
            return []

        paper_title_map = {p.id: p.title for p in corpus.papers}

        # Collect methods and evaluation conditions per card
        methods_by_paper: dict[str, list[str]] = defaultdict(list)
        eval_conditions_by_paper: dict[str, list[str]] = defaultdict(list)
        cards_with_extracted_conditions: set[str] = set()

        for stored_card in corpus.cards:
            card = stored_card.card
            pid = card.paper_id

            if card.methods:
                methods_by_paper[pid].extend(card.methods)

            # Check structured evaluation_conditions
            if card.evaluation_conditions:
                for item in card.evaluation_conditions:
                    if item.status == "observed" and item.value.strip():
                        eval_conditions_by_paper[pid].append(item.value.strip().lower())
                        cards_with_extracted_conditions.add(pid)
                    elif item.status in {"observed", "explicitly_absent"}:
                        cards_with_extracted_conditions.add(pid)

            # Also check attributable text in metrics, limitations, failure_cases
            attributable_text = " ".join(
                card.metrics + card.limitations + card.failure_cases
            ).lower()

            if attributable_text.strip():
                cards_with_extracted_conditions.add(pid)

            for cond in STANDARD_EVALUATION_CONDITIONS:
                if cond in attributable_text:
                    eval_conditions_by_paper[pid].append(cond)

        # Check extraction coverage threshold (>= 50% of cards must have extracted condition data)
        extraction_coverage = len(cards_with_extracted_conditions) / max(1, total_cards_count)
        if extraction_coverage < 0.5:
            # Unknown-heavy corpus: insufficient extraction coverage to infer evaluation gaps
            return []

        # Count method paper frequencies
        method_counts: dict[str, set[str]] = defaultdict(set)
        for pid, m_list in methods_by_paper.items():
            for m in m_list:
                clean_m = " ".join(m.split())
                if len(clean_m) >= 2:
                    method_counts[clean_m].add(pid)

        candidates: list[CandidateGap] = []
        now = _utc_now()

        for method, pids in method_counts.items():
            if len(pids) < 2:
                continue

            # Check which standard evaluation conditions are observed for this method family
            observed_conditions: set[str] = set()
            for pid in pids:
                for cond_val in eval_conditions_by_paper[pid]:
                    for std_cond in STANDARD_EVALUATION_CONDITIONS:
                        if std_cond in cond_val or cond_val in std_cond:
                            observed_conditions.add(std_cond)

            for cond in STANDARD_EVALUATION_CONDITIONS:
                if cond in observed_conditions:
                    continue

                # Target condition has 0 observed evidence for this method family!
                gap_id = generate_gap_id(
                    "evaluation", topic, f"{_slug(method)}-{_slug(cond)}"
                )

                title = enforce_language_safety(
                    f"Underrepresented evaluation condition in {topic}: {cond} for {method}"
                )
                description = enforce_language_safety(
                    f"Within the retrieved corpus of {topic}, evaluation condition '{cond}' "
                    f"was not observed for method family '{method}' across {len(pids)} "
                    f"analyzed paper(s)."
                )
                research_question = enforce_language_safety(
                    f"Within the retrieved corpus of {topic}, how does {method} perform "
                    f"under evaluation condition '{cond}'?"
                )

                supporting_evidence: list[EvidenceRef] = []
                for pid in sorted(pids)[:3]:
                    paper_title = paper_title_map.get(pid, f"Paper {pid}")
                    supporting_evidence.append(
                        EvidenceRef(
                            paper_id=pid,
                            paper_title=paper_title,
                            evidence_kind="supporting",
                            claim_or_field="methods",
                            supporting_text=f"Method '{method}' used in {paper_title}",
                        )
                    )

                provenance = GapProvenance(
                    retrievals=[],
                    corpus_paper_ids=list(corpus.corpus_paper_ids),
                    corpus_description=(
                        f"Evaluation analysis for '{topic}' across "
                        f"{total_cards_count} PaperCards"
                    ),
                    supporting_evidence=supporting_evidence,
                    conflicting_evidence=[],
                )

                caveats = [
                    "This is a candidate research question, not a verified novelty claim.",
                    (
                        f"Evaluation condition '{cond}' was not observed in the scoped "
                        f"corpus for method '{method}'."
                    ),
                    (
                        f"Analysis is bounded by the retrieved corpus of "
                        f"{total_cards_count} stored papers."
                    ),
                ]

                candidates.append(
                    CandidateGap(
                        id=gap_id,
                        title=title,
                        description=description,
                        gap_type="evaluation",
                        research_question=research_question,
                        supporting_papers=sorted(pids),
                        conflicting_papers=[],
                        evidence_count=len(supporting_evidence),
                        novelty_score=None,
                        evidence_score=0.5,
                        importance_score=None,
                        feasibility_score=None,
                        confidence=0.45,
                        search_scope=f"Evaluation Analysis over {total_cards_count} PaperCards",
                        caveats=caveats,
                        provenance=provenance,
                        review_status="candidate",
                        created_at=now,
                    )
                )

        return candidates
