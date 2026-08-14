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
from research_radar.models.paper_card import StructuredEvidence
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
        """Identify evaluation conditions with explicit unevaluated evidence in scoped corpus."""

        total_cards_count = len(corpus.cards)
        if total_cards_count < 2:
            return []

        paper_title_map = {p.id: p.title for p in corpus.papers}

        # Collect methods and explicit evaluation_conditions per card
        methods_by_paper: dict[str, list[str]] = defaultdict(list)
        eval_conditions_by_paper: dict[str, dict[str, StructuredEvidence]] = defaultdict(dict)

        for stored_card in corpus.cards:
            card = stored_card.card
            pid = card.paper_id

            if card.methods:
                methods_by_paper[pid].extend(card.methods)

            # PaperCard.evaluation_conditions is the sole source of truth
            for item in card.evaluation_conditions:
                if not item.value.strip():
                    continue
                val_clean = item.value.strip().lower()
                matched_cond = None
                for std_cond in STANDARD_EVALUATION_CONDITIONS:
                    if std_cond in val_clean or val_clean in std_cond:
                        matched_cond = std_cond
                        break
                cond_key = matched_cond or val_clean
                eval_conditions_by_paper[pid][cond_key] = item

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

            for cond in STANDARD_EVALUATION_CONDITIONS:
                observed_pids: list[str] = []
                explicitly_absent_pids: list[tuple[str, StructuredEvidence]] = []
                unknown_pids: list[str] = []

                for pid in pids:
                    ev = eval_conditions_by_paper[pid].get(cond)
                    if ev is None or ev.status == "unknown":
                        unknown_pids.append(pid)
                    elif ev.status == "observed":
                        observed_pids.append(pid)
                    elif ev.status == "explicitly_absent":
                        explicitly_absent_pids.append((pid, ev))

                # Rule 1: If condition was observed in ANY paper for this method => Not a gap
                if observed_pids:
                    continue

                # Rule 2: Candidate allowed ONLY when target condition has explicitly_absent
                # evidence. If all papers are unknown => insufficient evidence, no candidate.
                if not explicitly_absent_pids:
                    continue

                gap_id = generate_gap_id(
                    "evaluation", topic, f"{_slug(method)}-{_slug(cond)}"
                )

                title = enforce_language_safety(
                    f"Underrepresented evaluation condition in {topic}: {cond} for {method}"
                )
                description = enforce_language_safety(
                    f"Within the retrieved corpus of {topic}, evaluation condition '{cond}' "
                    f"was explicitly unevaluated for method family '{method}' across "
                    f"{len(pids)} analyzed paper(s)."
                )
                research_question = enforce_language_safety(
                    f"Within the retrieved corpus of {topic}, how does {method} perform "
                    f"under evaluation condition '{cond}'?"
                )

                supporting_evidence: list[EvidenceRef] = []
                for pid, ev in explicitly_absent_pids[:3]:
                    paper_title = paper_title_map.get(pid, f"Paper {pid}")
                    supporting_evidence.append(
                        EvidenceRef(
                            paper_id=pid,
                            paper_title=paper_title,
                            evidence_kind="supporting",
                            claim_or_field="evaluation_conditions",
                            source_section=ev.source_section,
                            supporting_text=ev.supporting_text,
                        )
                    )

                for pid in sorted(pids):
                    if len(supporting_evidence) >= 3:
                        break
                    if pid not in [ref.paper_id for ref in supporting_evidence]:
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
                        f"Evaluation condition '{cond}' was explicitly noted as unobserved "
                        f"in the scoped corpus for method '{method}'."
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
