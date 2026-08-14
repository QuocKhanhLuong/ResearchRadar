"""Deterministic contradiction gap miner over analyzed PaperCard claims and structured context."""

from __future__ import annotations

import re
from datetime import UTC, datetime

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


def calculate_context_compatibility(card_a: PaperCard, card_b: PaperCard) -> float:
    """Calculate context compatibility score between two PaperCards (0.0 to 1.0).

    Returns 0.0 for hard context incompatibilities (e.g. task/modality mismatch).
    """

    # 1. Task check
    tasks_a = {t.value.lower() for t in card_a.tasks if t.status == "observed"}
    if card_a.problem:
        tasks_a.add(card_a.problem.lower())

    tasks_b = {t.value.lower() for t in card_b.tasks if t.status == "observed"}
    if card_b.problem:
        tasks_b.add(card_b.problem.lower())

    terms_a = set(re.findall(r"\b[a-zA-Z0-9]{2,}\b", " ".join(tasks_a))) & TASK_CATEGORIES
    terms_b = set(re.findall(r"\b[a-zA-Z0-9]{2,}\b", " ".join(tasks_b))) & TASK_CATEGORIES

    if terms_a and terms_b and not (terms_a & terms_b):
        return 0.0  # Hard task mismatch

    # 2. Modality check
    mods_a = {m.value.lower() for m in card_a.modalities if m.status == "observed"}
    mods_b = {m.value.lower() for m in card_b.modalities if m.status == "observed"}

    m_terms_a = set(re.findall(r"\b[a-zA-Z0-9]{2,}\b", " ".join(mods_a))) & MODALITY_CATEGORIES
    m_terms_b = set(re.findall(r"\b[a-zA-Z0-9]{2,}\b", " ".join(mods_b))) & MODALITY_CATEGORIES

    if m_terms_a and m_terms_b and not (m_terms_a & m_terms_b):
        return 0.0  # Hard modality mismatch

    # 3. Soft compatibility & missing context penalties
    score = 1.0
    if not tasks_a or not tasks_b:
        score -= 0.15
    if not mods_a or not mods_b:
        score -= 0.15
    if not card_a.datasets or not card_b.datasets:
        score -= 0.1

    return max(0.2, round(score, 2))


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

                comp_score = calculate_context_compatibility(card_a, card_b)
                if comp_score < 0.5:
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

                    tokens_a = _tokenize(text_a)

                    for text_b, sec_b, supp_b in claims_b:
                        pol_b = _detect_claim_polarity(text_b)
                        if pol_b == "neutral" or pol_a == pol_b:
                            continue  # Must be opposite polarity!

                        tokens_b = _tokenize(text_b)
                        overlap_tokens = tokens_a & tokens_b

                        # Must share core research concepts (at least 2 non-stopwords)
                        if len(overlap_tokens) < 2:
                            continue

                        concept_phrase = " ".join(sorted(overlap_tokens)[:3])
                        id_a_min = min(card_a.paper_id, card_b.paper_id)
                        id_b_max = max(card_a.paper_id, card_b.paper_id)
                        pair_key = f"{id_a_min}:{id_b_max}:{concept_phrase}"
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)

                        gap_id = generate_gap_id(
                            "contradiction",
                            topic,
                            f"{_slug(concept_phrase)}-{_slug(card_a.paper_id)}-{_slug(card_b.paper_id)}",
                        )

                        title = enforce_language_safety(
                            f"Conflicting evidence on {concept_phrase} in {topic}"
                        )
                        description = enforce_language_safety(
                            f"Within the retrieved corpus of {topic}, paper '{card_a.paper_id}' "
                            f"and paper '{card_b.paper_id}' report contradictory findings "
                            f"regarding '{concept_phrase}' under similar reported conditions."
                        )
                        research_question = enforce_language_safety(
                            f"Within the retrieved corpus of {topic}, under which conditions does "
                            f"{concept_phrase} improve or degrade performance?"
                        )

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

                        provenance = GapProvenance(
                            retrievals=[],
                            corpus_paper_ids=list(corpus.corpus_paper_ids),
                            corpus_description=(
                                f"Contradiction analysis for '{topic}' across {len(cards)} cards"
                            ),
                            supporting_evidence=[ref_a],
                            conflicting_evidence=[ref_b],
                        )

                        caveats = [
                            "This is a candidate research question, not a verified novelty claim.",
                            (
                                "Within the retrieved corpus, these findings appear inconsistent "
                                "under similar reported conditions."
                            ),
                            (
                                "These results may reflect differences in datasets or "
                                "evaluation protocols."
                            ),
                            (
                                f"Analysis is bounded by the retrieved corpus of "
                                f"{len(corpus.corpus_paper_ids)} stored papers."
                            ),
                        ]

                        confidence = round(min(0.8, comp_score * 0.7), 2)

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
