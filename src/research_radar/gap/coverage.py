"""Deterministic coverage gap miner over attributable PaperCard matrix dimensions."""

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
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")[:20] or "item"


class CoverageGapMiner:
    """Mine sparse matrix combinations across attributable method x dimension matrices."""

    def mine_coverage_gaps(
        self, topic: str, corpus: ScopedCorpusResult
    ) -> list[CandidateGap]:
        """Build coverage matrices and identify sparse combinations with zero observed evidence."""

        if len(corpus.cards) < 2:
            return []

        total_cards_count = len(corpus.cards)
        paper_title_map = {p.id: p.title for p in corpus.papers}

        # Collect methods per paper card
        card_methods: dict[str, list[str]] = defaultdict(list)

        # Collect distinct dimension items with observed status
        card_tasks: dict[str, list[str]] = defaultdict(list)
        card_datasets: dict[str, list[str]] = defaultdict(list)
        card_modalities: dict[str, list[str]] = defaultdict(list)
        card_metrics: dict[str, list[str]] = defaultdict(list)
        card_conditions: dict[str, list[str]] = defaultdict(list)

        cards_with_tasks: set[str] = set()
        cards_with_datasets: set[str] = set()
        cards_with_modalities: set[str] = set()
        cards_with_metrics: set[str] = set()
        cards_with_conditions: set[str] = set()

        for stored_card in corpus.cards:
            card = stored_card.card
            pid = card.paper_id

            if card.methods:
                card_methods[pid].extend(card.methods)

            # Tasks
            if card.tasks:
                for item in card.tasks:
                    if item.status == "observed" and item.value.strip():
                        card_tasks[pid].append(item.value.strip())
                        cards_with_tasks.add(pid)
            elif card.problem:
                card_tasks[pid].append(card.problem.strip())
                cards_with_tasks.add(pid)

            # Datasets
            if card.datasets:
                for d in card.datasets:
                    if d.strip():
                        card_datasets[pid].append(d.strip())
                        cards_with_datasets.add(pid)

            # Modalities
            if card.modalities:
                for item in card.modalities:
                    if item.status == "observed" and item.value.strip():
                        card_modalities[pid].append(item.value.strip())
                        cards_with_modalities.add(pid)

            # Metrics (distinct from evaluation conditions!)
            if card.metrics:
                for m in card.metrics:
                    if m.strip():
                        card_metrics[pid].append(m.strip())
                        cards_with_metrics.add(pid)

            # Evaluation Conditions (distinct from metrics!)
            if card.evaluation_conditions:
                for item in card.evaluation_conditions:
                    if item.status == "observed" and item.value.strip():
                        card_conditions[pid].append(item.value.strip())
                        cards_with_conditions.add(pid)

        # Count method frequencies across independent papers
        method_counts: dict[str, set[str]] = defaultdict(set)
        for pid, methods in card_methods.items():
            for m in methods:
                clean_m = " ".join(m.split())
                if len(clean_m) >= 2:
                    method_counts[clean_m].add(pid)

        dimension_configs = [
            ("task", card_tasks, cards_with_tasks),
            ("dataset", card_datasets, cards_with_datasets),
            ("modality", card_modalities, cards_with_modalities),
            ("metric", card_metrics, cards_with_metrics),
            ("evaluation condition", card_conditions, cards_with_conditions),
        ]

        candidates: list[CandidateGap] = []
        now = _utc_now()

        for dim_label, card_dim_map, cards_with_dim in dimension_configs:
            # Check extraction coverage threshold (>= 50% of cards must have extracted values)
            extraction_coverage = len(cards_with_dim) / max(1, total_cards_count)
            if extraction_coverage < 0.5:
                # Insufficient extraction coverage for this dimension in scoped corpus
                continue

            dim_counts: dict[str, set[str]] = defaultdict(set)
            for pid, items in card_dim_map.items():
                for item in items:
                    clean_item = " ".join(item.split())
                    if len(clean_item) >= 2:
                        dim_counts[clean_item].add(pid)

            # Find sparse pairs: method (>=2 papers) and dimension item (>=2 papers)
            for method, m_pids in method_counts.items():
                if len(m_pids) < 2:
                    continue

                for dim_item, d_pids in dim_counts.items():
                    if len(d_pids) < 2:
                        continue

                    # Check joint occurrence of observed evidence in scoped corpus
                    joint_pids = m_pids & d_pids
                    if len(joint_pids) > 0:
                        # Observed evidence exists!
                        continue

                    # Zero observed evidence WITH sufficient extraction coverage
                    gap_id = generate_gap_id(
                        "coverage", topic, f"{_slug(method)}-{_slug(dim_item)}"
                    )

                    title = enforce_language_safety(
                        f"Unobserved coverage combination: {method} × {dim_item}"
                    )
                    description = enforce_language_safety(
                        f"Within the retrieved corpus of {topic}, no evidence was observed "
                        f"combining method '{method}' (supported by {len(m_pids)} paper(s)) "
                        f"with {dim_label} '{dim_item}' (supported by {len(d_pids)} paper(s))."
                    )
                    research_question = enforce_language_safety(
                        f"Within the retrieved corpus of {topic}, how can {method} be "
                        f"applied to or evaluated on {dim_label} '{dim_item}'?"
                    )

                    supporting_evidence: list[EvidenceRef] = []
                    for pid in sorted(m_pids)[:2]:
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
                    for pid in sorted(d_pids)[:2]:
                        paper_title = paper_title_map.get(pid, f"Paper {pid}")
                        supporting_evidence.append(
                            EvidenceRef(
                                paper_id=pid,
                                paper_title=paper_title,
                                evidence_kind="supporting",
                                claim_or_field=dim_label,
                                supporting_text=(
                                    f"{dim_label.capitalize()} '{dim_item}' used in {paper_title}"
                                ),
                            )
                        )

                    provenance = GapProvenance(
                        retrievals=[],
                        corpus_paper_ids=list(corpus.corpus_paper_ids),
                        corpus_description=(
                            f"Coverage matrix for '{topic}' over {total_cards_count} PaperCards"
                        ),
                        supporting_evidence=supporting_evidence,
                        conflicting_evidence=[],
                    )

                    caveats = [
                        "This is a candidate research question, not a verified novelty claim.",
                        (
                            "Zero observed count means no evidence was found in the retrieved "
                            "corpus, not that no research exists."
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
                            gap_type="coverage",
                            research_question=research_question,
                            supporting_papers=sorted(m_pids | d_pids),
                            conflicting_papers=[],
                            evidence_count=len(supporting_evidence),
                            novelty_score=None,
                            evidence_score=0.5,
                            importance_score=None,
                            feasibility_score=None,
                            confidence=0.4,
                            search_scope=f"Coverage Matrix over {total_cards_count} PaperCards",
                            caveats=caveats,
                            provenance=provenance,
                            review_status="candidate",
                            created_at=now,
                        )
                    )

        return candidates
