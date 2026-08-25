# Independent Code Audit: Research Data Foundation

**Repository:** `ResearchRadar`  
**Branch Audited:** `feat/research-data-foundation`  
**Base Commit:** `3b631d1`  
**Review Date:** 2026-08-25  
**Reviewer Role:** Independent Reviewer (Pair Programming Assistant)  
**Test Suite Status:** 390 passed, 0 failed, 6 deprecation warnings  
**Linter Status:** `ruff check .` — All checks passed  

---

## Executive Summary & Overall Verdict

An exhaustive, adversarial audit was conducted on the integrated code in `feat/research-data-foundation`. The feature branch introduces:
1. Canonical, provenance-preserving multi-provider discovery (`OpenAlex`, `Semantic Scholar`, `arXiv`) with deterministic graph union-find deduplication.
2. Content-addressed local document artifact storage (`PDF`, extracted text, parsed section JSON).
3. Document caching in `ReaderService` ensuring byte-identical PDF reads cost zero LLM calls.
4. Local embedding generation and an optional derived `Pinecone` index that strictly enforces SQLite as the single source of truth.
5. Deterministic Reciprocal Rank Fusion (RRF) hybrid retrieval with mathematical bounds preventing project priors from overriding retrieval evidence.
6. Safe question-answering (`/ask`) with strict citation validation against `AskContext.allowed_paper_ids` and prompt sanitization against global literature absence claims.
7. Background ingestion orchestration (`/ingest`) with granular, non-sensitive per-provider audit records.
8. Offline SQLite schema migrations and offline CLI inspector `scripts/inspect_research_store.py`.

### Verdict: **APPROVED WITH RECOMMENDED FIXES (NO BLOCKERS FOUND)**
- **BLOCKER:** 0
- **HIGH:** 1 (`AskService.ask` executes blocking SQLite queries and CPU-bound neural embedding / sync HTTP network calls on the asyncio event loop)
- **MEDIUM:** 2 (Blocking synchronous repository calls in Discord slash command handlers; Missing batch paper linking in `IngestionService`)
- **LOW:** 3 (Duplicate identity/normalization helper implementations; `PineconeSemanticIndex._resolve_index` exception handling disparity; Redundant internal query normalization)

All data ownership invariants, secret isolation rules, provider fallback mechanisms, and cost-control guarantees were verified and proven sound.

---

## Strict Data Ownership Invariant Audit

| Invariant Rule | Implementation Status | Verification Evidence |
| :--- | :--- | :--- |
| **SQLite is the ONLY canonical structured truth** | **VERIFIED** | `PaperTable`, `PaperSourceTable`, `PaperCardTable`, `GapCandidateTable`, `GapReviewTable`, `ProjectTable`, `DocumentArtifactTable`, `IngestionRunTable`, `ProviderRetrievalTable`. Every paper/card/gap ID surfaced by search or vector indexing must exist in SQLite. |
| **Local artifact store holds byte payloads; no binary in SQLite** | **VERIFIED** | `LocalArtifactStore` writes bytes to `<artifact_root>/papers/<paper_id>/<sha256>.<suffix>`. SQLite `DocumentArtifactTable` stores only metadata (`object_key`, `sha256`, `byte_size`, `mime_type`, `backend`). |
| **Pinecone is a DERIVED, rebuildable index only** | **VERIFIED** | Vectors contain only `entity_id`, `entity_type`, `paper_id`, `publication_year`, `embedding_schema_version`, `embedding_model`. `HybridRetriever.retrieve` checks `self._repository.get_paper(paper_id)` for every vector match and drops unresolvable IDs. |
| **LLM is for reasoning/extraction, never persistent truth** | **VERIFIED** | LLM extracts `PaperCard` models or synthesizes `/ask` answers. If LLM is offline or unconfigured, system falls back gracefully (mock provider or deterministic template) without crashing. |
| **Default startup works with zero credentials/offline** | **VERIFIED** | Default settings: `LLM_PROVIDER=mock`, `EMBEDDING_PROVIDER=disabled`, `SEMANTIC_INDEX=disabled`. Verified via `test_application_bot_constructs_offline_with_safe_default_settings` which blocks network sockets. |
| **Pinecone & remote LLM remain completely optional** | **VERIFIED** | Disabling Pinecone/LLM runs the complete application and all 390 test suite cases without external API keys or credentials. |

---

## Specific Adversarial Break Checks

### 1. Can any code path make MORE than one extra HTTP request per LLM call?
* **Tracing Target:** [`RemoteLLMProvider.generate_structured`](file:///Users/alvinluong/ResearchRadar/src/research_radar/reader/llm/remote.py#L59-L111)
* **Result:** **NO.**
* **Proof:**
  - `RemoteLLMProvider._send` issues a single `httpx.AsyncClient.post` request.
  - If the endpoint returns HTTP `400` or `422` AND `_rejects_response_format(response)` is `True`, a single fallback request is executed with a system prompt instruction and without `response_format`.
  - The fallback request is never retried further. Any subsequent failure raises `_unavailable_error_for_status(response.status_code)`.
  - There are no loops, recursion, or retry wrappers around `_send`. Maximum HTTP requests per call is strictly capped at `2` (1 initial + 1 fallback).

### 2. Can a project prior in `research/hybrid.py` promote a paper that retrieval never surfaced?
* **Tracing Target:** [`HybridRetriever.retrieve`](file:///Users/alvinluong/ResearchRadar/src/research_radar/research/hybrid.py#L157-L213) and [`assert_prior_cannot_dominate`](file:///Users/alvinluong/ResearchRadar/src/research_radar/research/hybrid.py#L55-L72)
* **Result:** **NO.**
* **Proof & Arithmetic:**
  1. Candidate paper IDs are generated exclusively from `{**lexical_ranks, **semantic_ranks}`:
     ```python
     for paper_id in {**lexical_ranks, **semantic_ranks}:
     ```
     If a paper in the project was not returned by lexical search and was not returned by semantic search, it never enters this iteration and its fused score is never evaluated.
  2. Mathematical domination check:
     - With defaults: `semantic_weight = 0.8`, `semantic_limit = 30`, `rrf_k = 60`.
     - Weakest semantic contribution: 0.8 * (1 / (60 + 30)) = 0.8 / 90 = 0.008888...
     - Maximum project prior: min(0.008, 0.008) = 0.008000.
     - Because 0.008000 < 0.008888, a project prior bonus cannot outrank even the weakest 30th-ranked semantic hit, let alone a lexical hit (where rank 1 is 1.0 / 61 ≈ 0.01639).

### 3. Can a semantic-only candidate displace lexical evidence in `research/ask.py` `build_ask_context`?
* **Tracing Target:** [`AskService.build_ask_context`](file:///Users/alvinluong/ResearchRadar/src/research_radar/research/ask.py#L287-L491)
* **Result:** **NO.**
* **Proof:**
  - Papers with `lexical_score == 0.0` are routed into `semantic_only_papers` and skipped during regular paper scoring.
  - `top_papers` is first populated up to `paper_budget` exclusively from `scored_papers` (where `lexical_score > 0`).
  - Only when `len(top_papers) < paper_budget` does the service take candidates from `semantic_only_papers` to fill the remaining empty slots:
     ```python
     if len(top_papers) < paper_budget:
         semantic_only_papers.sort(key=lambda x: (x[0], x[1].id))
         for _, paper in semantic_only_papers[: paper_budget - len(top_papers)]:
             top_papers.append(paper)
     ```
  - Therefore, a semantic-only candidate cannot displace any lexically matched paper.

### 4. Can a stale or deleted vector ID reach `AskContext.allowed_paper_ids`?
* **Tracing Target:** [`AskContext.allowed_paper_ids`](file:///Users/alvinluong/ResearchRadar/src/research_radar/research/ask.py#L87-L92) and [`HybridRetriever.retrieve`](file:///Users/alvinluong/ResearchRadar/src/research_radar/research/hybrid.py#L180-L183)
* **Result:** **NO.**
* **Proof:**
  - In `HybridRetriever.retrieve`, every vector hit is checked against SQLite:
     ```python
     if self._repository.get_paper(paper_id) is None:
         logger.debug("Discarding a candidate that no longer resolves in storage.")
         continue
     ```
  - In `AskService.build_ask_context`, semantic candidate IDs are resolved once more via `self._repository.get_paper(paper_id)`.
  - `AskContext.allowed_paper_ids` is derived from `retrieved_papers` and `retrieved_cards`, which contain only successfully resolved SQLite entities.

### 5. Can a repeat read of the same PDF trigger a second LLM call?
* **Tracing Target:** [`ReaderService.read_url`](file:///Users/alvinluong/ResearchRadar/src/research_radar/reader/service.py#L62-L144)
* **Result:** **NO.**
* **Proof:**
  - On the first read, the PDF is parsed, stored in `DocumentCache`, and the LLM extracts `PaperCard`, which is saved with `document_sha256 = sha`.
  - On the second read of the same URL and content:
    1. `_load_cached_document` loads the cached parsed document via `DocumentCache.load`.
    2. `self._repository.upsert_merged_paper` resolves the existing `paper_id`.
    3. `self._repository.get_paper_card_record(paper_id)` retrieves the stored card and confirms `stored_record.document_sha256 == sha`.
    4. `read_url` returns `ReadResult(..., from_cache=True)` at line 108, returning before `self._llm.generate_structured(...)` at line 120.

### 6. Can an API key reach a log line, an exception message, a URL, or a DB column?
* **Tracing Target:** [`Settings`](file:///Users/alvinluong/ResearchRadar/src/research_radar/config.py#L25-L53), [`OpenAlexProvider`](file:///Users/alvinluong/ResearchRadar/src/research_radar/providers/openalex.py#L68-L73), [`SemanticScholarProvider`](file:///Users/alvinluong/ResearchRadar/src/research_radar/providers/semantic_scholar.py#L116-L117), [`RemoteLLMProvider`](file:///Users/alvinluong/ResearchRadar/src/research_radar/reader/llm/remote.py#L69-L70), [`PineconeSemanticIndex`](file:///Users/alvinluong/ResearchRadar/src/research_radar/semantic/index.py#L214-L223), [`safe_provider_error`](file:///Users/alvinluong/ResearchRadar/src/research_radar/providers/base.py#L77-L87)
* **Result:** **NO.**
* **Proof:**
  - All secret settings use `pydantic.SecretStr` (preventing string/repr leaks).
  - API keys travel exclusively in HTTP headers (`Authorization: Bearer ...` or `x-api-key: ...`), never in query strings or URLs.
  - Exceptions are sanitized via `safe_provider_error` and `_unavailable_error_for_status`, which format only provider names and status codes without URLs, headers, or keys.
  - Ingestion audit tables (`provider_retrievals`) store `safe_error` bounded to 500 characters of sanitized text; raw request headers and auth payloads are never passed to repository layers.

### 7. Is any blocking SQLAlchemy or filesystem call made on the event loop without `asyncio.to_thread`?
* **Tracing Target:** `AskService.ask`, `ReaderService.read_url`, `IngestionService.ingest_research_topic`, Discord command callbacks
* **Result:** **YES — DETECTED IN `AskService.ask` AND DISCORD COMMAND HANDLERS.** (See Findings below for HIGH and MEDIUM issues).

### 8. Does a Pinecone outage cause retries, sleeps, or an exception escaping to the caller?
* **Tracing Target:** [`PineconeSemanticIndex`](file:///Users/alvinluong/ResearchRadar/src/research_radar/semantic/index.py#L182-L355) and [`HybridRetriever._semantic_ranks`](file:///Users/alvinluong/ResearchRadar/src/research_radar/research/hybrid.py#L220-L250)
* **Result:** **NO.**
* **Proof:**
  - `PineconeSemanticIndex.search`, `upsert`, and `delete` catch `Exception`, call `_note_outage` (logging a single sanitized warning and marking `_available = False`), and immediately return empty results (`[]` or `0`).
  - No sleeps, retry loops, or backoff schedules exist in the Pinecone backend.
  - `HybridRetriever._semantic_ranks` wraps semantic embedding and index calls in a `try...except Exception` block, logging a warning and falling back to lexical search without escaping exceptions.

### 9. Test Falsification Audit (Confirmed Non-Vacuous)
Three critical tests were selected, their underlying production logic was broken, and test failures were confirmed:

1. **Test 1:** [`test_project_prior_cannot_outrank_an_extra_retrieval_channel`](file:///Users/alvinluong/ResearchRadar/tests/test_hybrid_retrieval.py#L187)
   - *Edit:* Modified [`project_prior`](file:///Users/alvinluong/ResearchRadar/src/research_radar/research/hybrid.py#L100) to return `999.0`.
   - *Result:* Test failed with `AssertionError: assert 1 < 0` and `assert 999.0 <= 0.008`.
2. **Test 2:** [`test_second_read_of_same_bytes_skips_parser_and_llm`](file:///Users/alvinluong/ResearchRadar/tests/test_reader_cache.py#L179)
   - *Edit:* In [`ReaderService.read_url`](file:///Users/alvinluong/ResearchRadar/src/research_radar/reader/service.py#L107), bypassed the cache card return by replacing condition with `if False:`.
   - *Result:* Test failed with `AssertionError: assert 2 == 1 (llm.calls)`.
3. **Test 3:** [`test_record_artifact_is_idempotent_on_identity`](file:///Users/alvinluong/ResearchRadar/tests/test_ingestion_storage.py#L74)
   - *Edit:* In [`IngestionRepository.record_artifact`](file:///Users/alvinluong/ResearchRadar/src/research_radar/storage/ingestion_repository.py#L113), forced new row insertion (`if True:`).
   - *Result:* Test failed with `sqlite3.IntegrityError: UNIQUE constraint failed: document_artifacts.paper_id, document_artifacts.sha256, document_artifacts.artifact_type`.

*All falsification edits were fully reverted and confirmed clean with `git diff`.*

### 10. Is there duplicated architecture?
* **Artifact Storage:** Exactly one abstraction: Protocol [`ArtifactStore`](file:///Users/alvinluong/ResearchRadar/src/research_radar/artifacts/base.py#L51) implemented by [`LocalArtifactStore`](file:///Users/alvinluong/ResearchRadar/src/research_radar/artifacts/local.py#L55). [`DocumentCache`](file:///Users/alvinluong/ResearchRadar/src/research_radar/reader/cache.py#L36) sits cleanly on top of `ArtifactStore`.
* **Semantic Vector Store:** Exactly one abstraction: Protocol [`SemanticIndex`](file:///Users/alvinluong/ResearchRadar/src/research_radar/semantic/base.py#L103) implemented by `DisabledSemanticIndex`, `FakeSemanticIndex`, and `PineconeSemanticIndex`.
* **Deduplication:** Core grouping logic is implemented once in [`research_radar.research.dedup.group_papers`](file:///Users/alvinluong/ResearchRadar/src/research_radar/research/dedup.py#L50). [`canonical.py`](file:///Users/alvinluong/ResearchRadar/src/research_radar/research/canonical.py#L23) imports and delegates to `group_papers`.

### 11. Is there dead code or unused configuration?
* **Result:** All configuration properties in [`Settings`](file:///Users/alvinluong/ResearchRadar/src/research_radar/config.py) are actively consumed in application composition (`main.py`) or provider initialization.
* Minor code duplication identified in private storage string normalization helpers (see Finding RR-LOW-01).

---

## Detailed Audit of the Twenty Focus Areas

### 1. Architecture & Boundaries
* **Assessment:** Clean, layered design. Storage, provider, reader, semantic, research, and bot layers maintain well-defined dependency directions. The composition root in `main.py` wires all components without coupling domain logic to Discord.
* **Rating:** Strong.

### 2. SQLite Schema & Migrations
* **Assessment:** Schema definitions in `tables.py` use strict foreign key cascades and composite uniqueness constraints (`uq_paper_sources_provider_id`, `uq_document_artifacts_identity`, `uq_digest_runs_period`). Migrations in `migrations.py` are idempotent and tested against legacy databases.
* **Rating:** Strong.

### 3. Deduplication Correctness
* **Assessment:** Multi-pass union-find in `dedup.py` groups strong keys first (`doi`, `arxiv`, `external:`). Title grouping is guarded by `_groups_have_conflicting_strong_ids` to prevent false merges when strong IDs conflict.
* **Rating:** Strong.

### 4. Provenance
* **Assessment:** Ingestion records bare provider IDs in `PaperSourceTable` and `ProviderRetrievalTable`. URL-shaped resolver IDs (e.g. `https://openalex.org/W...`, `https://pubmed.ncbi.nlm.nih.gov/...`) are stripped to canonical bare identifiers (`openalex`, `pmid`). Locators such as `pdf_url` are explicitly excluded from `PaperSourceTable` via `_NON_IDENTITY_EXTERNAL_IDS`.
* **Rating:** Strong.

### 5. Ingestion Idempotency
* **Assessment:** Running `/ingest` repeatedly creates new `IngestionRunTable` audit entries while deduplicating and updating `papers` and `paper_sources` in place.
* **Rating:** Strong.

### 6. Document Cache Correctness
* **Assessment:** `DocumentCache` indexes by SHA256 of raw PDF bytes. A repeat read of identical bytes skips PDF parsing and skips LLM card generation.
* **Rating:** Strong.

### 7. Artifact Atomicity
* **Assessment:** `LocalArtifactStore._atomic_write` uses `tempfile.NamedTemporaryFile` in the target directory, flushes, syncs to disk (`os.fsync`), and replaces atomically (`os.replace`). Unfinished temp files are cleaned up in a `finally` block.
* **Rating:** Strong.

### 8. Pinecone Derived-Index Semantics
* **Assessment:** Pinecone stores derived vectors with minimal metadata. Every query hit resolves back to SQLite. Unresolvable IDs are dropped.
* **Rating:** Strong.

### 9. Embedding Versioning
* **Assessment:** `embedding_fingerprint` produces `(schema_version, model_id, dimension)`. Text input is NFKC normalized, whitespace-collapsed, and bounded to 2,000 characters.
* **Rating:** Strong.

### 10. Hybrid Retrieval
* **Assessment:** RRF combines lexical and semantic ranks deterministically. The project prior is mathematically constrained so it cannot override retrieval evidence.
* **Rating:** Strong.

### 11. AskContext / Source-ID Safety
* **Assessment:** `AskService.ask` validates LLM response IDs against `AskContext.allowed_paper_ids` and `AskContext.allowed_gap_ids`. Hallucinated or unauthorized IDs are removed. Global literature absence language is sanitized.
* **Rating:** Strong.

### 12. LLM Structured Compatibility
* **Assessment:** `RemoteLLMProvider` validates JSON outputs against Pydantic models with graceful single-retry fallback for providers rejecting `response_format`.
* **Rating:** Strong.

### 13. LLM Cost Amplification / Retry Risk
* **Assessment:** LLM requests are non-recursive and capped at 1 retry. Repeat PDF reads use cache.
* **Rating:** Strong.

### 14. API Secret Leakage
* **Assessment:** Secrets use `SecretStr`. Errors use `safe_provider_error`. Audit tables only record sanitized error strings.
* **Rating:** Strong.

### 15. Provider Outage Behavior
* **Assessment:** Partial outages during `ScoutService.search` return partial results with warnings. Total outages raise `ProviderUnavailableError`.
* **Rating:** Strong.

### 16. Network Retry Behavior
* **Assessment:** `get_with_retry` limits retries to 2 attempts with short delays (`0.2s * attempt`) for transient failures (429, 500, 502, 503, 504).
* **Rating:** Strong.

### 17. Async / Blocking I/O
* **Assessment:** Most background services use `asyncio.to_thread`. However, `AskService.ask`, `bot/commands/project.py`, and `bot/commands/gap.py` invoke synchronous repository/embedding methods directly on the event loop.
* **Rating:** Needs remediation (See Findings RR-HIGH-01 and RR-MED-01).

### 18. Test Quality
* **Assessment:** 390 hermetic tests with mock transports, in-memory databases, and zero socket leaks. Falsification proved test sensitivity.
* **Rating:** Strong.

### 19. Dead Code
* **Assessment:** No orphan modules or unused settings.
* **Rating:** Strong.

### 20. Overengineering
* **Assessment:** The code avoids premature multi-user tables and maintains simple, single-user SQLite abstractions.
* **Rating:** Strong.

---

## Detailed Findings & Remediations

### [HIGH] Finding RR-HIGH-01: Blocking SQLite & Neural Embedding Execution on the Asyncio Event Loop in `AskService.ask`

* **Location:** [`src/research_radar/research/ask.py:509-513`](file:///Users/alvinluong/ResearchRadar/src/research_radar/research/ask.py#L509-L513) in function `AskService.ask`
* **Failure Mode:**
  `AskService.ask` is an `async def` method called directly by the Discord slash command handler. On line 509, it executes:
  ```python
  ctx = self.build_ask_context(
      question,
      project_id_or_name=project_id_or_name,
      max_evidence=max_evidence,
  )
  ```
  `build_ask_context` is a synchronous method that executes 8+ synchronous SQLite transactions (`get_project`, `list_project_papers`, `list_project_gaps`, `search_papers`, `get_paper`, `get_paper_card`, `list_candidates`, `list_critic_reviews`).
  Furthermore, when semantic retrieval is enabled, `_semantic_candidate_ranks` calls `HybridRetriever.retrieve`, which executes CPU-intensive PyTorch model inference (`LocalEmbeddingProvider.embed_texts` via `SentenceTransformer.encode`) or synchronous Pinecone HTTP network queries (`PineconeSemanticIndex.search`) directly on the main event loop.
  This blocks the asyncio event loop thread, preventing Discord gateway heartbeat responses and stalling other concurrent coroutines during question answering.
* **Reproduction / Reasoning Chain:**
  1. Trigger `/ask` with `EMBEDDING_PROVIDER=local` and `SEMANTIC_INDEX=pinecone`.
  2. `AskService.ask` is invoked on the main asyncio thread.
  3. `build_ask_context` runs synchronous PyTorch encoding (`LocalEmbeddingProvider.embed_texts`) and synchronous Pinecone SDK queries on the event loop without `asyncio.to_thread`.
  4. The event loop cannot process incoming Discord heartbeat ACKs or other command interactions until context assembly finishes.
* **Required Remediation:**
  In `AskService.ask`, dispatch `self.build_ask_context` through `asyncio.to_thread`:
  ```python
  ctx = await asyncio.to_thread(
      self.build_ask_context,
      question,
      project_id_or_name=project_id_or_name,
      max_evidence=max_evidence,
  )
  ```

---

### [MEDIUM] Finding RR-MED-01: Direct Synchronous Repository Invocations in Discord Slash Command Handlers

* **Location:** [`src/research_radar/bot/commands/project.py:129, 143, 166, 191, 217`](file:///Users/alvinluong/ResearchRadar/src/research_radar/bot/commands/project.py#L129) and [`src/research_radar/bot/commands/gap.py:195`](file:///Users/alvinluong/ResearchRadar/src/research_radar/bot/commands/gap.py#L195)
* **Failure Mode:**
  The slash command callbacks (`project_create_cmd`, `project_list_cmd`, `project_show_cmd`, `project_add_paper_cmd`, `project_add_gap_cmd`, and `gap_show_cmd`) execute synchronous SQLAlchemy calls (`service.create_project`, `service.list_projects`, `service.get_project`, `service.add_paper_to_project`, `service.add_gap_to_project`, and `gap_service.get_candidate_detail`) directly inside the async command callbacks without `asyncio.to_thread`. While SQLite transactions are fast, executing synchronous database transactions directly on the discord.py event loop violates the asynchronous boundary established elsewhere in the codebase.
* **Required Remediation:**
  Wrap synchronous service/repository calls in `await asyncio.to_thread(...)` within the command callbacks, or provide async adapter methods on the service classes.

---

### [MEDIUM] Finding RR-MED-02: Non-Batched Sequential Transactions During Project Paper Linking in Ingestion

* **Location:** [`src/research_radar/research/ingestion.py:188-197`](file:///Users/alvinluong/ResearchRadar/src/research_radar/research/ingestion.py#L188-L197) in `IngestionService._link_papers_to_project`
* **Failure Mode:**
  When an ingestion run targets a project, `_link_papers_to_project` iterates through `paper_ids` and executes:
  ```python
  for paper_id in paper_ids:
      await asyncio.to_thread(self._repository.add_paper_to_project, project_id, paper_id)
  ```
  For an ingestion limit of 50 papers, this spawns 50 consecutive thread pool context switches and 50 separate database transactions (`session_scope`) in a serial loop.
* **Required Remediation:**
  Add a batch link method on `ResearchRepository` (e.g. `add_papers_to_project(project_id: str, paper_ids: Sequence[str])`) that links all papers within a single SQLite transaction and single `asyncio.to_thread` call.

---

### [LOW] Finding RR-LOW-01: Duplication of Identity Normalization Helpers

* **Location:** [`src/research_radar/storage/repositories.py:1807-1832`](file:///Users/alvinluong/ResearchRadar/src/research_radar/storage/repositories.py#L1807-L1832) vs [`src/research_radar/providers/normalization.py:19-50`](file:///Users/alvinluong/ResearchRadar/src/research_radar/providers/normalization.py#L19-L50)
* **Failure Mode:**
  `storage/repositories.py` defines private functions `_normalize_doi`, `_normalize_arxiv_identity`, and `_normalize_title` which duplicate the logic in `providers/normalization.py` (`normalize_doi`, `normalize_arxiv_id`) and `research/dedup.py` (`normalize_title`). Specifically, `_normalize_arxiv_identity` does not parse legacy archive formats (such as `math.PR/0501234`), whereas `normalize_arxiv_id` does.
* **Required Remediation:**
  Remove private helper duplicates in `repositories.py` and import the canonical helpers from `research_radar.providers.normalization` and `research_radar.research.dedup`.

---

### [LOW] Finding RR-LOW-02: `SemanticIndexError` Escapes Unhandled in `PineconeSemanticIndex._resolve_index`

* **Location:** [`src/research_radar/semantic/index.py:224-234`](file:///Users/alvinluong/ResearchRadar/src/research_radar/semantic/index.py#L224-L234)
* **Failure Mode:**
  `_resolve_index` explicitly re-raises `SemanticIndexError`:
  ```python
  try:
      return self._ensure_index()
  except SemanticIndexError:
      raise
  except Exception as exc:
      self._note_outage("setup", exc)
      return None
  ```
  If `SEMANTIC_INDEX=pinecone` is configured but the `pinecone` library is missing, `_ensure_index` raises `SemanticIndexError`. If a direct caller invokes `index.search(...)`, the error escapes instead of being converted into `_note_outage("setup", exc)` and returning `[]`. (Note: `HybridRetriever` catches `Exception` so it is safely absorbed at the retriever boundary, but direct callers of `PineconeSemanticIndex` experience an escaping exception).
* **Required Remediation:**
  Allow `_resolve_index` to catch `SemanticIndexError` and call `_note_outage("setup", exc)` so all direct index methods consistently degrade to no-ops.

---

### [LOW] Finding RR-LOW-03: Redundant Query Normalization in `HybridRetriever.retrieve`

* **Location:** [`src/research_radar/research/hybrid.py:166`](file:///Users/alvinluong/ResearchRadar/src/research_radar/research/hybrid.py#L166) and [`src/research_radar/storage/repositories.py:1845`](file:///Users/alvinluong/ResearchRadar/src/research_radar/storage/repositories.py#L1845)
* **Failure Mode:**
  `HybridRetriever.retrieve` normalizes the query via `" ".join(query.split())` and passes it to `_lexical_ranks`, which calls `repository.search_papers`, which calls `get_scoped_corpus`, which calls `_lexical_tokens` and `_normalize_search_text`. The multi-layer whitespace stripping is redundant though harmless.
* **Required Remediation:**
  Consolidate query normalization into a single shared utility.

---

## Summary of Audit Verification & Sign-Off

| Area | Status | Notes |
| :--- | :--- | :--- |
| **Data Invariants** | **PASS** | SQLite is sole canonical truth; Pinecone derived only. |
| **Deduplication** | **PASS** | Union-find with conflict prevention verified. |
| **Artifact Caching** | **PASS** | Content-addressed SHA256 storage prevents duplicate LLM calls. |
| **Security & Secrets** | **PASS** | Zero credential leaks across logs, URLs, DB, and errors. |
| **Retrieval Math** | **PASS** | Prior cannot override retrieval channels; proven algebraically. |
| **Test Integrity** | **PASS** | 390 tests; 3-point falsification verified. |
| **Async Architecture** | **ACTION REQUIRED** | Wrap `AskService.build_ask_context` and slash commands in `asyncio.to_thread`. |
