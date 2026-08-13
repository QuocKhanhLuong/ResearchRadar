# Gap Engine V2: Evidence-Backed Candidate Validation

## Status and boundary

This is a **future V2 design**, not an implemented ResearchRadar feature. V1
must remain a single-process, SQLite-backed research-ingestion and evidence
memory system. A gap engine is only useful once that foundation has reliable,
inspectable inputs; it must never be approximated by a generic LLM prompt such
as “find research gaps in these papers.”

The V2 engine remains normal Python services called from the existing
application composition root. It does not require a multi-agent framework,
vector database, worker queue, microservice, or new user/tenant model.

## Preconditions before implementation

V2 work should start only after V1 can demonstrate all of the following:

- Normalized, deduplicated `Paper` records with stable storage IDs and
  provider-source provenance.
- Persisted, validated `PaperCard` records whose evidence locations refer to
  actual selected document sections; absent evidence remains null/empty.
- A `ScoutService` that can run fresh, bounded, multi-provider searches and
  retain partial-failure warnings rather than treating a failure as no results.
- Repository operations for deterministic local retrieval and for reading the
  source IDs, watch/discovery provenance, PaperCards, and document-analysis
  provenance needed by a review.
- A shared task/domain taxonomy and explicit query scope. A sparse or absent
  field must mean “unknown in this retrieved corpus,” never “not studied.”

If these conditions are not met, improve V1 ingestion and evidence quality
instead of producing gap candidates.

## Future service boundary

```text
persisted Papers + PaperCards + discovery provenance
                         │
                         ▼
             GapMinerService (candidate signals)
                         │
                         ▼
              CandidateGap with provenance
                         │
                         ▼
               CriticService (fresh re-search)
                         │
                         ▼
             preserve | downgrade | reject
```

`GapMinerService` may use deterministic grouping/counting first and a bounded
LLM call only to structure already supplied evidence. It cannot cite a paper or
claim that is not in the candidate input. `CriticService` is a separate,
bounded verifier: it generates alternative searches, calls the existing
`ScoutService`, deduplicates the retrieved corpus, and records whether it found
overlapping or conflicting work. It is not an autonomous agent loop.

New V2 persistence, if and when needed, belongs behind repositories (for
example, candidate, provenance, evidence-link, and critic-review records) in
the same SQLite database. Raw provider payloads and unverified model output do
not become evidence records.

## Candidate and provenance contracts

The exact Pydantic/table shape can evolve, but a candidate must contain the
following concepts. Scores are bounded, documented heuristic assessments, not
measurements of objective novelty or truth.

```python
class EvidenceRef(BaseModel):
    paper_id: str
    paper_title: str
    evidence_kind: Literal["supporting", "conflicting", "context"]
    claim_or_field: str                 # e.g. "limitations" or "main_claims[1]"
    source_section: str | None
    supporting_text: str | None
    source_url: str | None


class RetrievalRecord(BaseModel):
    query: str                           # exact submitted query
    query_purpose: str                   # discovery, verification, or critic
    sources_searched: list[str]          # enabled providers attempted
    successful_sources: list[str]
    failed_sources: list[str]            # safe provider warning summaries
    retrieved_at: datetime                # UTC
    retrieved_paper_ids: list[str]       # post-dedup corpus membership
    result_count: int


class GapProvenance(BaseModel):
    retrievals: list[RetrievalRecord]    # all initial and critic searches
    corpus_paper_ids: list[str]          # exact deduplicated corpus used
    corpus_description: str              # task, time/filter, and inclusion scope
    supporting_evidence: list[EvidenceRef]
    conflicting_evidence: list[EvidenceRef]


class CandidateGap(BaseModel):
    title: str
    description: str
    gap_type: Literal[
        "explicit", "coverage", "contradiction", "evaluation", "method_transfer"
    ]
    research_question: str
    supporting_papers: list[str]
    conflicting_papers: list[str]
    evidence_count: int
    novelty_score: float | None
    evidence_score: float | None
    importance_score: float | None
    feasibility_score: float | None
    confidence: float | None
    search_scope: str
    caveats: list[str]
    provenance: GapProvenance
    review_status: Literal["candidate", "preserved", "downgraded", "rejected"]
```

`evidence_count` is derived from the deduplicated evidence references, not a
count of search hits. `supporting_papers` and `conflicting_papers` reference
the canonical stored paper IDs, while `EvidenceRef` links each conclusion back
to its exact PaperCard field/section where possible. A candidate with only
metadata, no source text, or a failed/partial retrieval can still be recorded
as a weak lead, but its caveats and confidence must make that limitation clear.

Each candidate must preserve, at minimum:

- every exact query used and its purpose;
- all sources attempted, successful, and unavailable;
- UTC retrieval times and the deduplicated retrieved-paper set;
- the inclusion/exclusion scope and any date, domain, or result-limit filters;
- supporting evidence and conflicting evidence with paper identity and source
  location; and
- the Critic’s alternative queries, new papers, outcome, and rationale.

This provenance makes a candidate reproducible enough to inspect later. It
does **not** make the retrieved corpus exhaustive.

## Candidate-signal workflows

All workflows create hypotheses, not claims of novelty. They first use the
persisted corpus, then require fresh discovery and Critic review before an
operator sees a candidate as preserved.

### 1. Explicit gap

1. Select PaperCards in a stated scope and collect `limitations`, `future_work`,
   and `failure_cases` with their paper IDs.
2. Group comparable statements using a documented taxonomy or conservative
   normalized matching; retain each original statement as evidence.
3. Form a question such as “Under what conditions is X insufficient for Y?”
   rather than converting an author’s suggestion into a fact.
4. Run fresh searches combining the task, method, limitation terms, synonyms,
   and likely competing terminology. Send the result to the Critic.

Repeated author-stated limitations are useful signals, not proof that no remedy
exists.

### 2. Coverage gap

1. Define a sparse matrix with an explicit scope, for example
   `method × task`, `method × dataset`, `method × modality`, or
   `method × evaluation condition`.
2. Populate cells only from parsed, attributable PaperCard evidence. Keep
   `unknown`, `not extracted`, and `observed zero` distinct.
3. Flag an under-covered cell only when its row/column comparison and minimum
   evidence thresholds are documented.
4. Search the sparse combination directly and through synonyms before retaining
   it as a candidate.

A blank cell is a retrieval/extraction observation, not evidence of an empty
research area.

### 3. Contradiction gap

1. Select apparently opposed, attributable `main_claims` about a comparable
   intervention and outcome.
2. Record task, dataset, modality, metric, and evaluation conditions alongside
   both claims. If these differ materially, the initial finding is a context
   mismatch, not a contradiction.
3. Search for replication, comparative, negative-result, and review literature
   using both claim directions and alternative terminology.
4. Preserve a contradiction candidate only if the evidence remains comparable;
   otherwise downgrade it to a context-dependent finding or reject it.

### 4. Evaluation gap

1. Count reported metrics and evaluation conditions from evidence-backed cards,
   including robustness settings such as scanner shift, domain shift, noise,
   and calibration when those fields become reliably extractable.
2. Compare the condition to the same scoped task/method corpus, not to all
   literature.
3. Search directly for the apparently rare condition and common aliases before
   scoring it.

For example, “noise robustness appears rarely evaluated in this retrieved
corpus” is permitted; “nobody evaluates noise robustness” is not.

### 5. Method-transfer gap

1. Define related source and target domains before counting, including the
   intended task, modality, and transfer assumptions.
2. Establish that a method family is supported by attributable evidence in
   domain A and appears limited in the scoped corpus for domain B.
3. Search method aliases, source-domain terminology, target-domain terminology,
   and known adaptation vocabulary for existing transfers.
4. State the resulting question as a feasibility hypothesis, including reasons
   a transfer could fail (data geometry, labels, cost, safety, or evaluation
   mismatch).

Frequency differences alone never establish a novel or practical transfer.

## Critic invalidation loop

Every `CandidateGap` passes through `CriticService` before it is surfaced as a
preserved opportunity.

```text
candidate + initial provenance
  -> derive bounded alternative queries
     (synonyms, broader/narrower terms, competing methods, negative claim,
      review/benchmark/replication terms, and domain-specific aliases)
  -> ScoutService fresh retrieval with normal timeout/failure handling
  -> normalize + deduplicate + attach new corpus/provenance
  -> compare new evidence with candidate assumptions
  -> preserve, downgrade, or reject with an auditable rationale
```

The Critic must have fixed limits for query count, per-provider result count,
and elapsed/retry budget. It must record unavailable providers and partial
searches; it cannot convert a failed source into supporting absence. A result
that finds prior work, a close baseline, an incompatible scope, or a credible
conflicting outcome should lower confidence or reject the candidate. A clean
result may only preserve the qualified hypothesis—it cannot prove global
novelty.

The review should be append-only or versioned: retain the original candidate,
initial corpus, fresh corpus, and decision instead of overwriting the evidence
trail. LLMs, if used to draft alternative queries or summarize comparison
results, receive only compact labeled evidence and must return structured,
validated output. They do not decide truth autonomously.

## Claim language and presentation rules

Candidate text and Discord/UI presentation must separate observed evidence
from inference and name the retrieval boundary. Preferred forms include:

- “Within the retrieved corpus, I found limited evidence for …”
- “The reviewed papers frequently report …; fresh searches may still be
  incomplete because …”
- “This is a candidate research question, not a verified novelty claim.”
- “The apparent disagreement may depend on dataset, metric, or evaluation
  condition.”

Never state “No one has studied X,” “X does not exist,” or an equivalent
comprehensive novelty claim. Surface provider failures, missing cards, sparse
evidence, and scope restrictions as caveats next to the candidate rather than
hiding them in logs.

## Incremental implementation gate

When V2 begins, add the smallest exercised components in this order:

1. Provenance/evidence-link persistence and deterministic explicit-gap signals.
2. Bounded Critic re-search with fake-provider tests and outcome persistence.
3. One matrix-based workflow with explicit unknown-vs-zero handling.
4. Contradiction, evaluation, and method-transfer workflows only after their
   required structured fields have sufficient extraction quality.

Tests must use fixture PaperCards and mocked provider results. They should
assert provenance completeness, stable deterministic signals, partial-provider
failure caveats, Critic downgrade/reject behavior, and prohibited claim wording.
No V2 test should require Discord, real scholarly APIs, a remote LLM, model
weights, a vector database, or a separate worker process.
