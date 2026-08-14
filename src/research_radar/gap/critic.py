"""Bounded Critic verification pass over candidate research gaps."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Protocol

from research_radar.errors import ProviderUnavailableError
from research_radar.models.gap import CandidateGap, CriticReview, EvidenceRef, RetrievalRecord
from research_radar.models.paper import Paper
from research_radar.models.paper_card import PaperCard
from research_radar.research.scout import ScoutResult
from research_radar.storage.repositories import StoredPaperCard


class ScoutSearchProtocol(Protocol):
    """Minimum scout interface required by CriticService."""

    async def search(self, query: str, limit: int) -> ScoutResult: ...


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _check_context_compatibility_and_rejection(
    card: PaperCard, candidate: CandidateGap, paper_title: str
) -> tuple[bool, EvidenceRef | None]:
    """Check if a stored PaperCard contains context-compatible evidence resolving candidate."""

    cand_text = f"{candidate.title} {candidate.research_question}".lower()
    candidate_keywords = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", cand_text))

    # Extract tasks from candidate and card
    card_tasks = {
        t.value.lower()
        for t in card.tasks
        if t.status == "observed" and t.value.strip()
    }
    if card.problem:
        card_tasks.add(card.problem.lower())

    # Extract modalities from candidate and card
    card_modalities = {
        m.value.lower()
        for m in card.modalities
        if m.status == "observed" and m.value.strip()
    }

    # Task compatibility check (e.g. reconstruction vs segmentation vs classification)
    known_task_terms = {
        "reconstruction", "segmentation", "classification", "detection", "registration"
    }
    cand_task_terms = candidate_keywords & known_task_terms
    card_text = " ".join(card_tasks)
    card_task_terms = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", card_text)) & known_task_terms
    if cand_task_terms and card_task_terms and not (cand_task_terms & card_task_terms):
        # Task mismatch -> context incompatible!
        return False, None

    # Modality compatibility check (e.g. mri vs ct vs xray vs ultrasound)
    known_modality_terms = {"mri", "ct", "xray", "ultrasound", "pet", "spect"}
    cand_mod_terms = candidate_keywords & known_modality_terms
    mod_text = " ".join(card_modalities)
    card_mod_terms = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", mod_text)) & known_modality_terms
    if cand_mod_terms and card_mod_terms and not (cand_mod_terms & card_mod_terms):
        # Modality mismatch -> context incompatible!
        return False, None

    # Check for direct resolution in main_claims, contributions, methods
    for claim in card.main_claims:
        claim_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", claim.claim.lower()))
        if len(candidate_keywords & claim_words) >= 3:
            return True, EvidenceRef(
                paper_id=card.paper_id,
                paper_title=paper_title,
                evidence_kind="conflicting",
                claim_or_field="main_claims",
                supporting_text=claim.claim,
            )

    for contrib in card.contributions:
        contrib_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", contrib.lower()))
        if len(candidate_keywords & contrib_words) >= 3:
            return True, EvidenceRef(
                paper_id=card.paper_id,
                paper_title=paper_title,
                evidence_kind="conflicting",
                claim_or_field="contributions",
                supporting_text=contrib,
            )

    return False, None


class CriticService:
    """Bounded verifier that tests candidate gaps against fresh multi-provider searches."""

    def __init__(self, scout: ScoutSearchProtocol) -> None:
        self._scout = scout

    def derive_queries(self, candidate: CandidateGap) -> list[str]:
        """Derive 3 to 5 bounded alternative verification queries."""

        queries: list[str] = []

        # 1. Title phrase
        clean_title = re.sub(
            r"^(Repeated limitation in|Candidate gap in)\s*",
            "",
            candidate.title,
            flags=re.IGNORECASE,
        )
        queries.append(clean_title.strip())

        # 2. Key terms from research question
        stop_words = {"within", "retrieved", "corpus", "how", "can", "address", "methods"}
        rq_words = [
            w for w in re.findall(r"\b[a-zA-Z0-9]{3,}\b", candidate.research_question)
            if w.lower() not in stop_words
        ]
        if rq_words:
            queries.append(" ".join(rq_words[:4]))

        # 3. Topic/limitation + review/benchmark
        if rq_words:
            queries.append(f"{' '.join(rq_words[:3])} benchmark review")

        # 4. Competing terminology / solution search
        if rq_words:
            queries.append(f"{' '.join(rq_words[:2])} robustness solution")

        # Deduplicate while preserving order, max 4 queries
        seen: set[str] = set()
        deduped: list[str] = []
        for q in queries:
            normalized = " ".join(q.split())
            if normalized and normalized.lower() not in seen:
                seen.add(normalized.lower())
                deduped.append(normalized)

        return deduped[:4]

    async def review_candidate(
        self,
        candidate: CandidateGap,
        review_version: int = 1,
        *,
        memory_cards: tuple[StoredPaperCard, ...] = (),
    ) -> tuple[CriticReview, CandidateGap]:
        """Execute fresh bounded searches, evaluate overlap, and record audit rationale."""

        queries = self.derive_queries(candidate)
        retrieval_records: list[RetrievalRecord] = []
        fresh_papers: list[Paper] = []
        provider_warnings: list[str] = []
        all_failed = True

        now = _utc_now()

        for query in queries:
            try:
                scout_result = await self._scout.search(query, limit=5)
                all_failed = False
                paper_ids = [p.id for p in scout_result.papers]
                fresh_papers.extend(scout_result.papers)
                if scout_result.warnings:
                    provider_warnings.extend(scout_result.warnings)

                successful = list(scout_result.provider_counts.keys())
                failed = [w.split()[0] for w in scout_result.warnings if w.split()]

                retrieval_records.append(
                    RetrievalRecord(
                        query=query,
                        query_purpose="critic",
                        sources_searched=list(dict.fromkeys(successful + failed)),
                        successful_sources=successful,
                        failed_sources=failed,
                        retrieved_at=now,
                        retrieved_paper_ids=paper_ids,
                        result_count=len(scout_result.papers),
                    )
                )
            except ProviderUnavailableError as error:
                provider_warnings.append(str(error))
                retrieval_records.append(
                    RetrievalRecord(
                        query=query,
                        query_purpose="critic",
                        sources_searched=[],
                        successful_sources=[],
                        failed_sources=[str(error)],
                        retrieved_at=now,
                        retrieved_paper_ids=[],
                        result_count=0,
                    )
                )

        dedup_warnings = list(dict.fromkeys(provider_warnings))
        new_paper_ids = list(
            dict.fromkeys(p.id for p in fresh_papers if p.id not in candidate.supporting_papers)
        )

        # Check for potential overlap in fresh literature (metadata overlap => downgrade only!)
        overlapping_paper_ids: list[str] = []
        cand_text = f"{candidate.title} {candidate.research_question}".lower()
        candidate_keywords = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", cand_text))

        for paper in fresh_papers:
            if paper.id in candidate.supporting_papers:
                continue
            paper_text = f"{paper.title} {paper.abstract or ''}".lower()
            paper_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", paper_text))
            overlap = candidate_keywords & paper_words
            if len(overlap) >= 3:
                overlapping_paper_ids.append(paper.id)

        overlapping_paper_ids = list(dict.fromkeys(overlapping_paper_ids))

        # Check for context-compatible grounded rejection from stored PaperCards
        invalidating_paper_id: str | None = None
        invalidating_evidence_ref: EvidenceRef | None = None

        for stored_card in memory_cards:
            if (
                stored_card.card.paper_id in candidate.supporting_papers
                or stored_card.card.paper_id in candidate.conflicting_papers
            ):
                continue

            card = stored_card.card
            paper_title = f"Paper {card.paper_id}"

            is_rejected, ev_ref = _check_context_compatibility_and_rejection(
                card, candidate, paper_title
            )

            if is_rejected and ev_ref:
                invalidating_paper_id = card.paper_id
                invalidating_evidence_ref = ev_ref
                break

        # Decision matrix
        decision: str
        rationale: str
        critic_caveats: list[str] = []

        if invalidating_paper_id and invalidating_evidence_ref:
            decision = "rejected"
            rationale = (
                f"Ground evidence in stored PaperCard for paper '{invalidating_paper_id}' "
                f"directly addresses or resolves this candidate research question."
            )
            critic_caveats.append(
                f"Resolved by ground evidence in analyzed paper '{invalidating_paper_id}'."
            )
        elif all_failed:
            decision = "downgraded"
            rationale = (
                "Verification incomplete: all scholarly providers were unavailable during Critic."
            )
            critic_caveats.append(
                "Scholarly providers were unavailable during Critic verification."
            )
        elif overlapping_paper_ids:
            decision = "downgraded"
            rationale = (
                f"Fresh searches found {len(overlapping_paper_ids)} potentially overlapping works "
                f"that have not yet been deeply analyzed."
            )
            critic_caveats.append(
                f"{len(overlapping_paper_ids)} fresh papers retrieved by Critic "
                f"require deep analysis for potential overlap."
            )
        elif dedup_warnings:
            decision = "downgraded"
            rationale = (
                "No strong overlapping work found, but partial provider failures occurred."
            )
            critic_caveats.extend(dedup_warnings)
        else:
            decision = "preserved"
            rationale = (
                "No strong overlapping evidence found in bounded fresh searches."
            )

        # Retain original caveats and add critic caveats
        combined_caveats = list(dict.fromkeys(candidate.caveats + critic_caveats))

        # Confidence calculation
        old_conf = candidate.confidence if candidate.confidence is not None else 0.5
        if decision == "preserved":
            new_conf = round(min(1.0, old_conf + 0.15), 2)
        elif decision == "downgraded":
            new_conf = round(max(0.1, old_conf - 0.15), 2)
        else:
            new_conf = 0.0

        updated_provenance = candidate.provenance.model_copy(deep=True)
        updated_provenance.retrievals.extend(retrieval_records)
        if invalidating_evidence_ref:
            updated_provenance.conflicting_evidence.append(invalidating_evidence_ref)

        updated_candidate = candidate.model_copy(
            update={
                "review_status": decision,
                "confidence": new_conf,
                "caveats": combined_caveats,
                "provenance": updated_provenance,
            }
        )

        review = CriticReview(
            candidate_id=candidate.id,
            review_version=review_version,
            queries_used=queries,
            retrieval_records=retrieval_records,
            new_paper_ids=new_paper_ids,
            overlapping_paper_ids=overlapping_paper_ids,
            decision=decision,  # type: ignore[arg-type]
            rationale=rationale,
            caveats=critic_caveats,
            created_at=now,
        )

        return review, updated_candidate
