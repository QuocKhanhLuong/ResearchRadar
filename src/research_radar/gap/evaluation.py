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

        if len(corpus.cards) < 2:
            return []

        paper_title_map = {p.id: p.title for p in corpus.papers}

        # Collect methods and explicit evaluation conditions per card
        methods_by_paper: dict[str, list[str]] = defaultdict(list)
        eval_conditions_by_paper: dict[str, list[str]] = defaultdict(list)

        for stored_card in corpus.cards:
            card = stored_card.card
            pid = card.paper_id

            if card.methods:
                methods_by_paper[pid].extend(card.methods)

            # Look for evaluation conditions in metrics, limitations, and failure cases
            all_text = " ".join(
                card.metrics + card.limitations + card.failure_cases
            ).lower()

            for cond in STANDARD_EVALUATION_CONDITIONS:
                if cond in all_text:
                    eval_conditions_by_paper[pid].append(cond)

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

            # Check which standard evaluation conditions are missing for this method family
            evaluated_conditions: set[str] = set()
            for pid in pids:
                evaluated_conditions.update(eval_conditions_by_paper[pid])

            for cond in STANDARD_EVALUATION_CONDITIONS:
                if cond in evaluated_conditions:
                    continue

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
                            supporting_text=(
                                f"Method '{method}' evaluated without explicit '{cond}' "
                                f"testing in {paper_title}"
                            ),
                        )
                    )

                provenance = GapProvenance(
                    retrievals=[],
                    corpus_paper_ids=list(corpus.corpus_paper_ids),
                    corpus_description=(
                        f"Evaluation analysis for '{topic}' across "
                        f"{len(corpus.cards)} PaperCards"
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
                        f"{len(corpus.corpus_paper_ids)} stored papers."
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
                        search_scope=f"Evaluation Analysis over {len(corpus.cards)} PaperCards",
                        caveats=caveats,
                        provenance=provenance,
                        review_status="candidate",
                        created_at=now,
                    )
                )

        return candidates
