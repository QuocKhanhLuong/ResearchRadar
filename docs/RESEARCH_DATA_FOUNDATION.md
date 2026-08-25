# Research Data Foundation

How ResearchRadar stores, caches, indexes and retrieves real research data.

## 1. System of record

| Store | Role | Rebuildable |
| --- | --- | --- |
| `data/research_radar.db` (SQLite) | **Canonical** structured source of truth | No — back this up |
| `data/artifacts/` | PDF bytes, extracted text, section JSON | No — back this up |
| Pinecone | Derived semantic index only | **Yes** — rebuild from SQLite |
| LLM | Reasoning and extraction | N/A — never a source of truth |

Two rules follow from this table and are enforced in code:

1. **No binary content in SQLite.** `document_artifacts` records where an
   artifact lives; the bytes live on disk.
2. **No canonical evidence in Pinecone.** A vector hit is a *candidate*. It is
   resolved against SQLite before use, and an id that no longer resolves is
   discarded rather than surfaced (`research/hybrid.py`).

## 2. Data lifecycle

```
provider search
  -> normalized Paper models          (providers/)
  -> canonical identity grouping      (research/dedup.py, research/canonical.py)
  -> canonical papers + provider rows (storage/repositories.py)
  -> PDF fetch, hash, artifact cache  (reader/cache.py, artifacts/)
  -> bounded section selection        (reader/reader.py)
  -> PaperCard extraction via LLM     (reader/service.py)
  -> derived vectors                  (semantic/)
  -> hybrid candidate retrieval       (research/hybrid.py)
  -> evidence assembly for /ask       (research/ask.py)
```

## 3. Artifact layout

```
data/
  research_radar.db
  artifacts/
    papers/
      <paper_id>/
        <sha256>.pdf
        <sha256>.txt
        <sha256>.sections.json
```

Artifacts are content addressed by the SHA256 of the **source document**, so
every artifact derived from one document version shares a stem. Writes go to a
temporary file in the destination directory, are flushed and `fsync`ed, then
renamed with `os.replace`, so a partially written artifact is never observable.
Writing identical bytes twice returns the same key without rewriting the file.
`paper_id` is validated against a strict allowlist, so no key can escape the
artifact root.

`ARTIFACT_ROOT` selects the root directory.

## 4. Ingestion pipeline

```
ingest_research_topic(query, limit<=50, project_id=None, auto_read=0)
  -> resolve the project reference to a storage id (fails loudly if unknown)
  -> open an IngestionRun (status=running)
  -> fan out to providers concurrently, isolating per-provider failure
  -> record one ProviderRetrieval per provider
  -> canonicalize and deduplicate across providers
  -> persist through upsert_merged_paper
  -> optionally link papers to the project
  -> optionally read up to auto_read papers (default 0)
  -> close the run with discovered_count / canonical_count
```

Bounds are deliberate. Metadata ingestion clamps at
`INGESTION_METADATA_LIMIT` (default 50). `auto_read` defaults to **0**, so
ingestion never implicitly triggers a batch of LLM reads, and `/ingest` never
passes a non-zero value. A failure after the run opens marks it `failed` with a
safe error and re-raises.

Repeating an ingestion creates **no duplicate papers**. It does append a second
run row, because runs are an audit log.

## 5. Provider provenance

One canonical `papers` row carries one `paper_sources` row per contributing
provider. `provider_retrievals` records, per run and per provider: the query,
the identifiers *that provider* returned, a status, a result count and a safe
error summary.

Never persisted: API keys, auth headers, URLs carrying credentials, or raw
provider payloads.

Two identifier rules protect provenance:

- Keys that locate a document rather than identify a publication —
  `pdf_url` — never become `paper_sources` rows.
- Resolver-URL identifiers are reduced to their bare form, so a `pmid` of
  `https://pubmed.ncbi.nlm.nih.gov/17969013` from OpenAlex and a bare
  `17969013` from Semantic Scholar are the same identity.

## 6. Dedup hierarchy

Deterministic, strongest first:

1. Normalized DOI.
2. Trusted cross-provider identifiers (`openalex`, `s2`, `pmid`, `pmcid`, `mag`).
3. Normalized arXiv identifier.
4. Exact normalized-title fallback — applied **only** when the groups being
   bridged carry no conflicting strong identifiers of the same type.

There is no fuzzy or similarity-based title merging. `dedup.group_papers` is the
single implementation; `deduplicate` merges its groups and `canonicalize`
consumes them while retaining provider identity.

Result: one publication found via OpenAlex, Semantic Scholar and arXiv becomes
**one paper with three sources**.

## 7. Reader cache

1. Fetch the PDF through the SSRF-safe bounded fetcher.
2. Hash the bytes.
3. If a sections artifact exists for that hash, **skip parsing**.
4. If a stored PaperCard carries that `document_sha256`, **skip the LLM call**.
5. Otherwise parse, select bounded sections, call the LLM once, persist.

A repeat read of byte-identical content costs **zero LLM requests**. Pass
`force_refresh=True` to re-extract deliberately. Changed bytes produce a new
hash, a new artifact set and a fresh extraction; previous artifacts are kept.

Section selection is unchanged and deliberately bounded — abstract,
introduction and conclusion first, then method, experiments and limitations as
budget allows. A whole PDF is never sent to a model.

## 8. LLM configuration

```
LLM_PROVIDER=mock          # mock | remote
LLM_BASE_URL=
LLM_MODEL=
LLM_API_KEY=
```

`mock` is the default and requires no credentials. `remote` targets any
OpenAI-compatible `/v1/chat/completions` endpoint. One configured model — there
is no model routing and no fallback model.

Usage telemetry (provider, model, operation, token counts, timestamp) is
recorded when the endpoint reports it. Operations are
`paper_card_extraction`, `ask_synthesis`, `critic_review`, `other`.

## 9. GoRouter compatibility

Attempt 1 sends `response_format={"type": "json_object"}`.

If — and only if — the endpoint answers 400/422 with a body naming
`response_format`, the request is retried **exactly once** without that
parameter, with a strict JSON-only instruction appended. The result is
validated through Pydantic either way.

Never retried: authentication failures, rate limits, 5xx responses, timeouts,
and model output that fails validation. **At most one extra HTTP request per
call, ever.** There is no retry loop anywhere in this path.

## 10. Local embeddings

```
EMBEDDING_PROVIDER=disabled     # disabled | local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Install with `pip install -e ".[embeddings]"`. The model loads lazily on first
use, so importing the module downloads nothing and unit tests never touch it —
they use `FakeEmbeddingProvider`.

Vector text is bounded and deterministic:

- **Paper:** title + abstract.
- **PaperCard:** problem, contributions, methods, tasks, modalities, datasets,
  metrics, claims, limitations.

Raw PDFs are never embedded.

## 11. Pinecone semantics

```
SEMANTIC_INDEX=disabled         # disabled | pinecone
PINECONE_API_KEY=
PINECONE_INDEX=
PINECONE_NAMESPACE=research-radar
```

Install with `pip install -e ".[pinecone]"`. Vector ids are deterministic
(`<entity_type>:<paper_id>`), so upserts are idempotent.

Metadata is compact and non-evidential: `entity_id`, `entity_type`, `paper_id`,
`publication_year`, `embedding_schema_version`, `embedding_model`. No titles,
abstracts or claim text.

**Outage behaviour:** every client call is attempted once. A failure logs a
warning, marks the index unavailable, and returns an empty candidate list — the
system continues lexically. Nothing is retried and nothing sleeps.

Incomplete credentials fall back to the disabled index with a warning rather
than failing startup.

## 12. Hybrid retrieval

```
lexical candidates (SQLite)   ─┐
semantic candidates (Pinecone)├─> reciprocal-rank fusion -> bounded top-K ids
project relationship prior    ─┘        -> resolve every id against SQLite
                                        -> discard unresolvable ids
```

Fusion is deterministic: fixed weights, sorted by fused score, tie-broken on
paper id.

**The project prior is capped so it cannot replace retrieval evidence.** With
the shipped weights a single reciprocal-rank contribution spans 0.01639 down to
0.00889, so the cap is `0.008` — strictly below the weakest semantic
contribution. A project relationship therefore cannot compensate for missing an
entire retrieval channel, and a paper neither channel returned is never a
candidate at all. `HybridConfig` refuses any configuration that breaks this.

In `/ask`, semantic candidates **only widen recall**. They rank strictly below
every lexically matched paper, filling unused evidence budget without ever
taking a slot a lexical match would occupy. The lexical gate, rejected-ideas
handling, source-ID validation and the bounded context budget are unchanged.
With `SEMANTIC_INDEX=disabled` the retrieved set is identical to before.

## 13. Reindex workflow

The semantic index is derived, so a model or schema change is a reindex, never
a migration. `embedding_fingerprint` returns
`(schema_version, model_id, dimension)` — changing the model or dimension
changes the fingerprint, which makes the requirement observable rather than
silent.

Current schema versions: `paper-v1`, `papercard-v1`.

Canonical recovery flow:

```
SQLite -> re-embed -> Pinecone upsert
```

## 14. Backup and restore

Back up exactly two things:

```
data/research_radar.db
data/artifacts/
```

Checkpoint or stop the process first so SQLite's WAL is flushed. Restore by
putting both back and restarting.

**Pinecone needs no backup.** It is derived and rebuildable from SQLite, which
is why it is never allowed to hold canonical evidence.

## 15. Operability

```bash
python scripts/inspect_research_store.py
python scripts/inspect_research_store.py --paper-id <id>
python scripts/inspect_research_store.py --project "My Project"
python scripts/inspect_research_store.py --latest-runs 5
```

Prints counts for papers, paper sources, paper cards, document artifacts,
ingestion runs, provider retrievals, projects, project papers, gap candidates
and critic reviews. Offline; no network, no LLM, no dashboard.

## 16. Default startup guarantee

The application starts and the full suite passes with:

```
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=disabled
SEMANTIC_INDEX=disabled
```

Pinecone and the remote LLM are strictly optional, and CI needs no credentials.
