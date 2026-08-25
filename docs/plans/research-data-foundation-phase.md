# Research Data Foundation Phase — Implementation Plan

Status: in progress
Branch: `feat/research-data-foundation`
Starting commit: `3b631d1`

## Goal

Make ResearchRadar usable on real research data, end to end:

```
OpenAlex / Semantic Scholar / arXiv
  -> canonicalize + deduplicate
  -> SQLite canonical research memory
  -> PDF / text artifact cache
  -> PaperCard extraction
  -> local embeddings
  -> Pinecone derived semantic index
  -> hybrid retrieval
  -> Project / Ask / Gap / Critic
  -> Discord
```

## Data ownership (non-negotiable)

| Store | Role |
| --- | --- |
| SQLite (`data/research_radar.db`) | Canonical structured source of truth |
| Local artifact store (`data/artifacts/`) | PDF bytes, extracted text, section JSON |
| Pinecone | Derived semantic retrieval index only, fully rebuildable |
| LLM | Reasoning and extraction, never a persistent source of truth |

Every semantic hit must resolve back into SQLite before it is used as evidence.
A vector ID that no longer resolves is discarded, not surfaced.

## What already exists (reuse, do not re-create)

- `models/paper.py` — `Paper` with `external_ids`, `source`.
- `models/paper_card.py` — `PaperCard`, `EvidenceClaim`, `StructuredEvidence`.
- `models/gap.py`, `models/project.py` — gap and project domain models.
- `storage/tables.py` — `PaperTable`, `PaperSourceTable`, `PaperCardTable`,
  `WatchTopicTable`, `WatchPaperTable`, `DigestRunTable`, `GapCandidateTable`,
  `GapReviewTable`, `ProjectTable`, `ProjectPaperTable`, `ProjectGapTable`.
- `storage/repositories.py` — `ResearchRepository`, including
  `upsert_merged_paper` (persist-time canonical merge), `get_scoped_corpus`,
  `get_papers_for_local_lexical_search`.
- `storage/migrations.py` — additive, idempotent SQLite migration runner.
- `providers/base.py` — `PaperProvider` protocol, `get_with_retry`,
  `provider_timeout`, `safe_provider_error`, `clamp_provider_limit`.
- `providers/normalization.py` — `normalize_doi`, `normalize_arxiv_id`,
  `known_external_ids`.
- `research/dedup.py` — `identity_keys`, `deduplicate`, `merge_papers`.
- `research/scout.py`, `research/service.py`, `research/ask.py`, `research/ranker.py`.
- `reader/fetcher.py` — SSRF-safe bounded `DirectPDFFetcher`.
- `reader/parser.py` — `PDFParser`, `canonical_section_name`.
- `reader/reader.py` — `select_useful_sections`, `validate_card_evidence`.
- `reader/service.py` — `ReaderService` (no cache yet).
- `reader/llm/` — `LLMProvider` protocol, `MockLLMProvider`, `RemoteLLMProvider`.

## Gaps this phase closes

1. No content-addressed artifact store — PDFs are re-downloaded and re-parsed
   on every read, and no extracted text is retained.
2. No relational record of document artifacts, ingestion runs, or per-provider
   retrievals, so ingestion has no provenance and no idempotency evidence.
3. Discovery has no persisting ingestion engine: `ScoutService` fans out to
   providers, but nothing records a bounded run or wires dedup to persistence
   with a project link.
4. `RemoteLLMProvider` always sends `response_format`, has no compatibility
   fallback, and emits no usage telemetry.
5. No embeddings, no semantic index, no hybrid retrieval.
6. No operability entry point for inspecting stored research memory.

## Ownership map

Claude owns the shared integration surface for the whole phase. Workers may not
touch these files:

- `src/research_radar/config.py`
- `src/research_radar/main.py`
- `src/research_radar/storage/tables.py`
- `src/research_radar/storage/repositories.py`
- `src/research_radar/storage/database.py`
- `src/research_radar/research/ask.py`
- `.env.example`, `pyproject.toml`, `.github/workflows/ci.yml`
- every `__init__.py` export list

Claude publishes a contract commit before workers start, containing the new
tables, the new settings, and the empty protocol modules. Workers implement
against those contracts inside files they exclusively own.

## Contract layer (Claude, before Wave A)

New settings on `Settings`:

```
artifact_root: str = "data/artifacts"
embedding_provider: str = "disabled"      # disabled | local
embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
semantic_index: str = "disabled"          # disabled | pinecone
pinecone_api_key: SecretStr | None = None
pinecone_index: str | None = None
pinecone_namespace: str = "research-radar"
```

New tables in `storage/tables.py`:

- `DocumentArtifactTable` — `id`, `paper_id`, `sha256`, `artifact_type`,
  `backend`, `object_key`, `mime_type`, `byte_size`, `source_url`, `created_at`.
  Unique on `(paper_id, sha256, artifact_type)`.
- `IngestionRunTable` — `id`, `query`, `normalized_query`, `status`,
  `started_at`, `completed_at`, `requested_limit`, `discovered_count`,
  `canonical_count`, `providers`, `project_id`, `safe_error`.
- `ProviderRetrievalTable` — `id`, `run_id`, `provider`, `query`,
  `external_ids`, `retrieved_at`, `status`, `result_count`, `safe_error`.

New protocol modules (empty of implementation, owned per worker afterwards):

- `artifacts/base.py` — `ArtifactStore`, `ArtifactRef`, `ArtifactType`.
- `semantic/base.py` — `EmbeddingProvider`, `SemanticIndex`, `SemanticHit`,
  `SemanticRecord`.

There is exactly one artifact abstraction and exactly one semantic-index
abstraction for the whole phase. No `VectorStore`/`SemanticStore`/
`EmbeddingIndex` triplets.

## Execution DAG

```
                        Claude contract commit
                                  |
       +--------+--------+--------+--------+--------+--------+
       |        |        |        |        |        |        |
      W1       W2       W3       W4       W5       W6       W7      (Wave A)
   artifact  schema  openalex  s2+arxiv ingestion reader   llm
    store    repos   provider  providers  engine   cache  hardening
       |        |        |        |        |        |        |
       +--------+--------+--------+--------+--------+--------+
                                  |
                     Claude integration pass A
                                  |
              +--------+----------+----------+--------+
              |        |          |          |        |
             W8       W9         W10        W11      W12          (Wave B)
          embedding pinecone   hybrid    discord+  adversarial
          provider   index    retrieval  inspect      QA
              |        |          |          |        |
              +--------+----------+----------+--------+
                                  |
                     Claude integration pass B
                                  |
                        full test suite green
                                  |
                       Agy independent review
                                  |
                        Claude remediation
                                  |
                          final validation
                                  |
                             push branch
```

Wave A is the data foundation and can run fully in parallel because every
worker codes against the contract commit. Wave B depends on Wave A's persisted
artifacts and ingestion provenance existing.

## Worker assignments

| Worker | Scope | Exclusive files |
| --- | --- | --- |
| W1 | Content-addressed artifact store | `artifacts/local.py`, `tests/test_artifacts.py` |
| W2 | Storage accessors for the three new tables | `storage/ingestion_repository.py`, `tests/test_ingestion_storage.py` |
| W3 | OpenAlex hardening | `providers/openalex.py`, `tests/test_provider_openalex.py` |
| W4 | Semantic Scholar + arXiv | `providers/semantic_scholar.py`, `providers/arxiv.py`, `tests/test_provider_s2_arxiv.py` |
| W5 | Canonicalization + ingestion engine | `research/canonical.py`, `research/ingestion.py`, `tests/test_ingestion_engine.py` |
| W6 | Reader document cache | `reader/cache.py`, `reader/service.py`, `tests/test_reader_cache.py` |
| W7 | Remote LLM hardening + telemetry | `reader/llm/remote.py`, `reader/llm/telemetry.py`, `tests/test_llm_remote.py` |
| W8 | Embedding provider | `semantic/embedding.py`, `tests/test_embedding.py` |
| W9 | Pinecone semantic index | `semantic/index.py`, `tests/test_semantic_index.py` |
| W10 | Hybrid retrieval fusion | `research/hybrid.py`, `tests/test_hybrid_retrieval.py` |
| W11 | `/ingest` command + inspection script | `bot/commands/ingest.py`, `scripts/inspect_research_store.py`, `tests/test_ingest_command.py` |
| W12 | Adversarial cross-component QA | `tests/test_phase_regression.py` |

Each worker runs in its own git worktree on its own branch, all branched from
the contract commit. Claude cherry-picks or merges into the phase branch and
resolves every cross-cutting edit itself.

## Dedup policy

Deterministic identity hierarchy, strongest first:

1. Normalized DOI.
2. Trusted cross-provider identifiers (`openalex`, `s2`/`corpusid`, `pmid`,
   `pmcid`, `mag`).
3. Normalized arXiv identifier.
4. Exact normalized-title fallback, applied only when the groups being bridged
   have no conflicting strong identifiers of the same type.

No fuzzy or similarity-based title merging. One canonical `Paper` row plus one
`PaperSource` row per contributing provider.

## Ingestion flow

```
ingest_research_topic(query, limit<=50, project_id=None, auto_read=0)
  -> open IngestionRun (status=running)
  -> fan out to providers concurrently, isolate per-provider failure
  -> record one ProviderRetrieval per provider (status + safe_error only)
  -> canonicalize + deduplicate across all provider results
  -> persist through ResearchRepository.upsert_merged_paper
  -> optionally link papers to a project
  -> optionally read up to auto_read papers (default 0)
  -> close IngestionRun with discovered_count / canonical_count
```

Repeating the same ingestion produces no duplicate papers. `auto_read` defaults
to zero so ingestion never triggers a batch of LLM calls implicitly.

## Artifact layout

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

Content addressed by SHA256 of the source bytes. Writes go to a temporary file
in the same directory and are atomically renamed into place, so a partially
written artifact is never observable. Writing identical content twice yields
the same key and does not duplicate the file. A changed document produces a new
SHA and therefore a new artifact, and the previous artifact is preserved.
`paper_id` is validated against a strict character allowlist so no key can
escape the artifact root.

## Reader cache behavior

1. Resolve a PDF URL for the paper.
2. Download bytes through the existing SSRF-safe fetcher.
3. Hash to SHA256; if a `.pdf` artifact for that SHA exists, reuse it.
4. If a `.sections.json` artifact exists for that SHA, skip PDF parsing.
5. If a stored `PaperCard` already carries that `document_sha256`, skip the LLM
   call entirely unless `force_refresh=True`.
6. Otherwise parse, select bounded sections, call the LLM once, persist.

Section selection keeps the current two-pass philosophy: abstract, introduction
and conclusion first; method, experiments, ablations and limitations added only
when budget remains. A whole PDF is never sent to the model.

## LLM behavior

`LLM_PROVIDER=remote` targets an OpenAI-compatible `/v1/chat/completions`
endpoint (GoRouter with Claude is the intended deployment).

Attempt 1 sends `response_format={"type": "json_object"}`. Only when the
endpoint explicitly rejects that parameter — an HTTP 400/422 whose body names
`response_format` — is the request retried exactly once without it, with an
appended strict JSON-only instruction. The result is validated through Pydantic
either way. Authentication failures, generic 5xx responses and malformed model
output are never retried. There is at most one extra request per call.

Usage telemetry is recorded when the endpoint reports it: provider, model,
operation (`paper_card_extraction`, `ask_synthesis`, `critic_review`, `other`),
input/output/total tokens, timestamp. No model routing is introduced.

## Embeddings

`EmbeddingProvider` exposes `embed_texts`, `dimension`, `model_id`. The local
implementation lazily loads `sentence-transformers/all-MiniLM-L6-v2` on first
use, batches inputs and bounds input length deterministically. Unit tests use a
`FakeEmbeddingProvider` and never download a model. `sentence-transformers` is
an optional extra, so the default install and CI stay light.

Schema versions `paper-v1` and `papercard-v1` are recorded alongside
`embedding_model` and `embedding_dimension` on every vector, which makes a
model or schema change a visible, deterministic reindex requirement.

Vector text:

- Paper: title + abstract.
- PaperCard: problem, contributions, methods, tasks, modalities, datasets,
  metrics, main claims, limitations.

Raw PDFs are never embedded.

## Pinecone semantics

`SemanticIndex` exposes `upsert`, `search`, `delete` and `status`.
`DisabledSemanticIndex` is the default and is a total no-op that reports
unavailability. `PineconeSemanticIndex` writes compact metadata only:
`entity_id`, `entity_type`, `paper_id`, `publication_year`,
`embedding_schema_version`, `embedding_model`. Vector IDs are deterministic, so
upserts are idempotent.

A Pinecone outage degrades semantic retrieval to unavailable and the system
continues lexically. Failures are caught at the index boundary and never retried
in a storm. Unit tests use a fake index; CI needs no credential.

## Hybrid retrieval

```
lexical candidates (SQLite)  ─┐
semantic candidates (Pinecone)├─> deterministic fusion -> bounded top-K IDs
project relationship prior   ─┤        -> resolve every ID against SQLite
existing relevance signals   ─┘        -> discard unresolvable IDs
                                       -> load Paper / PaperCard / Gap evidence
```

Fusion is reciprocal-rank based with fixed weights and a stable tie-break on
paper ID, so the same inputs always produce the same ordering. Preserved rules:
the lexical gate stays where it is epistemically load-bearing; project rejected
ideas stay excluded; the project prior is capped so it cannot promote an
otherwise irrelevant paper; source-ID validation, the bounded `AskContext` and
the deterministic context budget are unchanged. Semantic similarity never
bypasses evidence provenance. With `SEMANTIC_INDEX=disabled` the current lexical
behavior is exactly preserved.

## Operability

`python scripts/inspect_research_store.py` prints compact counts for papers,
paper sources, paper cards, document artifacts, ingestion runs, projects,
project papers, gap candidates and critic reviews, with optional `--paper-id`,
`--project` and `--latest-runs` detail. No web dashboard.

Discord stays thin. Existing commands keep working. At most one command is
added: `/ingest query:"..." count:20`.

## Backup and recovery

Backup V1: stop or checkpoint the process, then preserve `data/research_radar.db`
and `data/artifacts/`. Pinecone needs no backup because it is derived and
rebuildable. Canonical recovery flow: `SQLite -> reindex -> Pinecone`.

## Default startup guarantee

The application must start and pass its full test suite with:

```
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=disabled
SEMANTIC_INDEX=disabled
```

Pinecone and the remote LLM stay strictly optional.

## Out of scope for this phase

PubMed, Europe PMC, Crossref, LLM rerankers, multi-model routing, automated
model escalation, raw PDF chunk vectorization, GraphRAG, a knowledge-graph
database, a web dashboard, cloud deployment, multi-user support.
