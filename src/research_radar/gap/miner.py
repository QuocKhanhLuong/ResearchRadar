"""Explicit gap miner using attributable limitations from PaperCards."""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime
from uuid import uuid4

from research_radar.models.gap import (
    CandidateGap,
    EvidenceRef,
    GapProvenance,
)
from research_radar.storage.repositories import ScopedCorpusResult, StoredPaper

PROHIBITED_CLAIMS = (
    "no one has",
    "nobody has",
    "never been studied",
    "does not exist",
    "first ever",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _tokenize(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text).lower()
    words = re.findall(r"\b[a-z0-9]{2,}\b", normalized)
    return {w for w in words if w not in STOPWORDS}


def check_language_safety(text: str) -> bool:
    """Return True if text is free of forbidden absolute novelty claims."""

    lowered = text.lower()
    return not any(phrase in lowered for phrase in PROHIBITED_CLAIMS)


def enforce_language_safety(text: str) -> str:
    """Clean absolute novelty claims from generated text."""

    result = text
    for phrase in PROHIBITED_CLAIMS:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        result = pattern.sub("limited evidence was found for", result)
    return result


class ExplicitGapMiner:
    """Extract attributable explicit limitations from scoped PaperCards and group them."""

    def mine_gaps(
        self, topic: str, corpus: ScopedCorpusResult
    ) -> list[CandidateGap]:
        """Group repeated author-stated limitations across independent papers."""

        if not corpus.cards:
            return []

        paper_map: dict[str, StoredPaper] = {
            paper.id: paper for paper in corpus.papers
        }

        # Step 1: Collect attributable evidence references
        extracted_evidence: list[EvidenceRef] = []
        for stored_card in corpus.cards:
            card = stored_card.card
            paper = paper_map.get(card.paper_id)
            paper_title = paper.title if paper else f"Paper {card.paper_id}"

            for idx, text in enumerate(card.limitations):
                if text.strip():
                    extracted_evidence.append(
                        EvidenceRef(
                            paper_id=card.paper_id,
                            paper_title=paper_title,
                            evidence_kind="supporting",
                            claim_or_field=f"limitations[{idx}]",
                            source_section="limitations",
                            supporting_text=text.strip(),
                            source_url=stored_card.source_url,
                        )
                    )

            for idx, text in enumerate(card.future_work):
                if text.strip():
                    extracted_evidence.append(
                        EvidenceRef(
                            paper_id=card.paper_id,
                            paper_title=paper_title,
                            evidence_kind="supporting",
                            claim_or_field=f"future_work[{idx}]",
                            source_section="future_work",
                            supporting_text=text.strip(),
                            source_url=stored_card.source_url,
                        )
                    )

            for idx, text in enumerate(card.failure_cases):
                if text.strip():
                    extracted_evidence.append(
                        EvidenceRef(
                            paper_id=card.paper_id,
                            paper_title=paper_title,
                            evidence_kind="supporting",
                            claim_or_field=f"failure_cases[{idx}]",
                            source_section="failure_cases",
                            supporting_text=text.strip(),
                            source_url=stored_card.source_url,
                        )
                    )

        if not extracted_evidence:
            return []

        # Step 2: Group comparable statements deterministically
        groups: list[list[EvidenceRef]] = []
        for ref in extracted_evidence:
            ref_tokens = _tokenize(ref.supporting_text or "")
            if not ref_tokens:
                continue

            merged = False
            for group in groups:
                # Check overlap with existing group
                group_tokens: set[str] = set()
                for item in group:
                    group_tokens.update(_tokenize(item.supporting_text or ""))

                overlap = ref_tokens & group_tokens
                # Conservative overlap threshold (>= 2 shared terms or high Jaccard)
                jaccard = len(overlap) / max(1, len(ref_tokens | group_tokens))
                if len(overlap) >= 2 or jaccard >= 0.35:
                    group.append(ref)
                    merged = True
                    break

            if not merged:
                groups.append([ref])

        # Step 3: Filter groups with minimum support (>= 2 independent papers)
        candidates: list[CandidateGap] = []
        now = _utc_now()

        for group in groups:
            paper_ids = list(dict.fromkeys(ref.paper_id for ref in group))
            if len(paper_ids) < 2:
                # 1 paper: weak lead only, do not surface as candidate gap in V2A
                continue

            # Key terms for candidate generation
            combined_tokens: set[str] = set()
            for ref in group:
                combined_tokens.update(_tokenize(ref.supporting_text or ""))
            top_terms = sorted(combined_tokens)[:4]
            theme_summary = " ".join(top_terms) if top_terms else "observed limitation"

            evidence_count = len(group)
            evidence_score = min(1.0, round(0.4 + 0.15 * len(paper_ids) + 0.05 * evidence_count, 2))
            confidence = min(1.0, round(0.3 + 0.1 * len(paper_ids), 2))

            title = enforce_language_safety(
                f"Repeated limitation in {topic}: {theme_summary}"
            )
            description = enforce_language_safety(
                f"Within the retrieved corpus, {theme_summary} appears as a repeated limitation "
                f"across {len(paper_ids)} independent papers."
            )
            research_question = enforce_language_safety(
                f"Within the retrieved corpus of {topic}, how can methods address {theme_summary}?"
            )

            corpus_desc = f"Scoped corpus of {len(corpus.cards)} PaperCards matching '{topic}'"
            provenance = GapProvenance(
                retrievals=[],
                corpus_paper_ids=list(corpus.corpus_paper_ids),
                corpus_description=corpus_desc,
                supporting_evidence=group,
                conflicting_evidence=[],
            )

            caveats = [
                "This is a candidate research question, not a verified novelty claim.",
                (
                    f"Analysis is bounded by the retrieved corpus of "
                    f"{len(corpus.corpus_paper_ids)} stored papers."
                ),
            ]
            if corpus.missing_cards_paper_ids:
                caveats.append(
                    f"{len(corpus.missing_cards_paper_ids)} matching stored papers lack an "
                    f"analyzed PaperCard."
                )

            scope_str = (
                f"{len(corpus.cards)} stored PaperCards "
                f"({corpus.total_matching_papers} matching papers)"
            )

            candidates.append(
                CandidateGap(
                    id=str(uuid4()),
                    title=title,
                    description=description,
                    gap_type="explicit",
                    research_question=research_question,
                    supporting_papers=paper_ids,
                    conflicting_papers=[],
                    evidence_count=evidence_count,
                    novelty_score=None,
                    evidence_score=evidence_score,
                    importance_score=None,
                    feasibility_score=None,
                    confidence=confidence,
                    search_scope=scope_str,
                    caveats=caveats,
                    provenance=provenance,
                    review_status="candidate",
                    created_at=now,
                )
            )

        return candidates
