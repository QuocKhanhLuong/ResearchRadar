# Integration, Lifecycle, and Release Audit — Personal Research Chat & Memory

**Phase:** `feat/personal-research-chat-memory`  
**Run / Worker:** Orca Supervised Run `run_c97597f3f039` — Worker A8  
**Phase Base & Tip:** `b41391ac142356a666a6f92ec1eb606ea0299f73` -> branch tip of `feat/personal-research-chat-memory`  
**Last revised:** final review pass (see section 8) — figures below are re-measured, not carried over  
**Scope:** Verification of composition root, lifecycle management, memory command ownership, CI workflow, documentation consistency, and diff validation across all parallel worker deliverables (W1–W12).

---

## 1. Executive Summary & Verification Verdict

The Personal Research Chat & Memory phase delivers an autonomous research assistant with personal context awareness, deterministic-first chat routing, epistemic source-boundary enforcement, and owner-scoped memory observability.

All 12 worker deliverables have been audited and verified for architectural alignment, contract conformance, security guarantees, and hermetic test execution.

- **Automated Test Results:** **807 tests passing**, 0 skipped, 0 failures. The suite is fully live: the optional Graphiti/Kuzu extra is installed in this checkout, so the previously-skipped adapter test now runs.
- **Linter & Formatting:** 100% clean under Ruff (`E,F,I,B,UP`) at 100-character line length.
- **Release Verdict:** **READY FOR MERGE / RELEASE**, after the two remediations in section 8. The earlier verdict was issued before those defects were found.

---

## 2. Composition Root & Lifecycle Audit

### 2.1 Service Construction & Dependency Graph (`src/research_radar/main.py`)

The application composition root (`build_application_bot`) wires all core and optional subsystems offline without requiring active Discord gateway credentials or live backend connections:

1. **Storage & Repositories:** Initializes SQLite database schema, `ResearchRepository`, and `IngestionRepository`.
2. **Artifacts & Cache:** Instantiates `LocalArtifactStore` and `DocumentCache` for content-addressed PDF caching and deduplication.
3. **Scholarly Providers & Scout:** Composes `ArxivProvider`, `OpenAlexProvider`, and `SemanticScholarProvider` into `ScoutService`.
4. **LLM Provider:** Instantiates `RemoteLLMProvider` when remote configuration is valid; otherwise defaults safely to `MockLLMProvider`.
5. **Embedding & Semantic Retrieval:** Configures `LocalEmbeddingProvider` and `PineconeSemanticIndex` when credentials exist; cleanly falls back to `DisabledSemanticIndex` when unconfigured or offline.
6. **User Memory:** Instantiates `GraphitiUserMemoryStore` when `USER_MEMORY_BACKEND=graphiti` or `DisabledUserMemoryStore` by default.
7. **Chat Pipeline:** Composes `ChatService` with `ChatRouter`, `MemoryCapturePolicy`, `ChatBudget`, and `IngestionService`.
8. **Bot & Command Tree:** Builds `ResearchRadarBot` with 16 slash commands and attaches chat-on-mention handler.

### 2.2 Complete Slash Command Registry

All 16 expected slash commands are registered in `bot.tree` (verified by enumerating `bot.tree.get_commands()` against a default-settings build):
- **Core Research:** `/ping`, `/paper`, `/watch`, `/read`, `/digest`, `/gap`, `/gap-show`, `/ask`
- **Project State:** `/project-create`, `/project-list`, `/project-show`, `/project-add-paper`, `/project-add-gap`
- **Ingestion & Discovery:** `/ingest`
- **Memory Observability (Owner-Scoped):** `/memory-status`, `/memory-search`

### 2.3 Lifecycle Hooks & Clean Resource Teardown

- **Startup Hook (`on_startup`):**
  - Binds `DiscordNotificationSink` to the bot client if `discord_channel_id` is configured.
  - Starts `AsyncIOScheduler` for background watch and digest routines.
- **Shutdown Hook (`on_shutdown`):**
  - Halts `AsyncIOScheduler` cleanly (`wait=False`).
  - Calls `await user_memory.close()` to flush and disconnect database drivers.
  - Closes shared HTTP client (`await http_client.aclose()`).
  - Disposes SQLite database engine connection pool (`db.dispose()`).

Verified via unit tests in `tests/test_composition.py`.

---

## 3. Memory Command Ownership & Security Boundary Audit

### 3.1 Owner Verification & Refusal Path (`src/research_radar/bot/commands/memory.py`)

- **Strict Access Check:** `/memory-status` and `/memory-search` evaluate `_is_owner(interaction, settings)`.
- When `discord_owner_user_id` is set, any interaction from a non-owner user is immediately replied to with `OWNER_REFUSAL_MESSAGE` (ephemeral), aborting before any backend call (search, status, or database connection).
- When `discord_owner_user_id` is unset (`None`), single-user/local usage is assumed.

### 3.2 Information Disclosure & Credential Protection

- **Sanitized Status (`render_memory_status_embed`):** Only exposes whitelisted fields: backend name, enabled status, health status, truncated persistence path, and episode count. Never echoes database connection strings, credentials, or internal exception messages.
- **Sanitized Search (`render_memory_search_embed`):** Only displays memory class, secret-redacted fact text, and temporal validity span. Drops internal UUIDs, group IDs, vector scores, and raw graph metadata.
- **Advisory Footers:** Emits `"Advisory personal context — not scientific evidence"` on search embeds to prevent hallucinated authority.

Verified via `tests/bot/test_memory_commands.py` and `tests/test_composition.py`.

---

## 4. CI & Release Readiness Audit

### 4.1 GitHub Actions Workflow (`.github/workflows/ci.yml`)

```yaml
name: CI
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install ruff pytest pytest-asyncio
          pip install -e .
      - name: Run Ruff check
        run: ruff check .
      - name: Run Pytest
        run: pytest
```

### 4.2 Packaging & Dependency Isolation (`pyproject.toml`)

- **Base Dependencies:** Kept minimal (`APScheduler`, `discord.py`, `httpx`, `pydantic`, `PyMuPDF`, `python-dotenv`, `SQLAlchemy`).
- **Optional Extras:**
  - `memory = ["graphiti-core>=0.29,<0.30", "kuzu>=0.11,<0.12"]`
  - `embeddings = ["sentence-transformers>=3.0"]`
  - `pinecone = ["pinecone>=5.0"]`
  - `dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "ruff>=0.5"]`
- **Offline / Hermetic Execution:** All unit, integration, and adversarial tests run completely hermetic without network access.

---

## 5. Documentation & Contract Diff Validation

| Document | Verification Status | Notes |
| --- | --- | --- |
| `docs/PHASE_CONTRACTS.md` | Fully Conforming | All data models, method signatures, and protocols match exact spec. |
| `docs/ENVIRONMENT.md` | Fully Conforming | Covers all new environment variables, defaults, and security notes. |
| `.env.example` | Fully Conforming | Every `Settings` field is present with no extras and no real values (checked by diffing `Settings.model_fields` against the template keys). Clean defaults (`USER_MEMORY_BACKEND=disabled`, `SEMANTIC_INDEX=disabled`, `LLM_PROVIDER=mock`, `CHAT_LIVE_DISCOVERY_LIMIT=10`, `HTTP_TIMEOUT_SECONDS=60`). |
| `README.md` | Fully Conforming | Updated with chat mention usage, personal memory configuration, and architecture summary. |

---

## 6. Test Verification Summary

| Test Suite | Test Count | Status | Description |
| --- | --- | --- | --- |
| `tests/adversarial/` | 23 | Passed | Falsification suite for source boundaries, memory contamination, and rate clamping |
| `tests/bot/` | 61 | Passed | Mention policy, mention handler, interaction helpers, and memory commands |
| `tests/chat/` | 133 | Passed | ChatRouter, EvidencePacket assembly, Prompt builder, ChatService pipeline, and credential redaction |
| `tests/e2e/` | 15 | Passed | End-to-end multi-turn chat scenarios, memory degradation, and mention boundary |
| `tests/memory/` | 146 | Passed | Secret redaction, capture policy, Graphiti store adapter, and fake store |
| `tests/test_config.py` | 23 | Passed | Settings defaults, validators, and list parsing |
| `tests/test_composition.py` | 5 | Passed | Composition root, lifecycle hooks, store builders, and bot tree permissions |
| `tests/test_phase_regression.py` | 23 | Passed | Cross-provider normalization, idempotency, caching, and credential safety |
| Other Core Suites | 378 | Passed | Storage, reader, gap engine, scout, watch, and digest services |
| **Total** | **807** | **All Green** | **0 failures, 0 skipped** |

Ruff (`E,F,I,B,UP`, line length 100) is clean across the tree. The four
operator scripts run green offline: `scripts/seed_demo_research_memory.py`,
`scripts/smoke_test_research_radar.py` (8/8), `scripts/inspect_research_store.py`,
and `scripts/smoke_test_personal_research_chat.py` (15/15).

---

## 7. Risk Analysis & Release Considerations

1. **Graphiti / Kuzu Deprecation Warning:** Upstream Kuzu is marked deprecated in favor of Neo4j/FalkorDB; ResearchRadar suppresses this expected warning at the adapter boundary as documented in `docs/spikes/graphiti_compat.md`.
2. **Live Discovery Turn Bound:** `ChatService` clamps turn-level discovery requests to `min(budget.max_discovery_results, 12)` and divides evenly across active providers, preventing provider over-fetching.
3. **Advisory Authority Rule:** The epistemic prompt builder strictly guarantees that SQLite project memory outranks user memory, and user memory is never cited as scientific evidence.

**Final Recommendation:** Approved for merge into `main` and production release, with
the residual risks in section 8 accepted as documented decisions.

---

## 8. Final Review Pass — Findings and Decisions

This section records a review of the integrated branch itself, rather than of
the individual worker deliverables. Two defects were found and fixed; the
remaining items are explicit decisions, not oversights.

### 8.1 Fixed — HIGH: credential-shaped input escaped through the research path

`MemoryCapturePolicy` blocked secrets from becoming memory, but nothing blocked
them from becoming a *retrieval query*. `ChatRouter.normalize_search_query`
strips conversational framing and keeps the rest of the message, so a
credential pasted into a mention survived into `RouteDecision.search_query` and
from there reached three destinations:

- `ingestion_runs.query` and `provider_retrievals.query` in SQLite (durable
  plaintext provenance),
- the outbound arXiv / OpenAlex / Semantic Scholar requests (third parties),
- the synthesis prompt, including the `Retrieval query used:` line.

Reproduced directly against the branch before the fix.

**Fix:** `ChatService.chat` now applies `redact_secrets` to the normalized turn
text before routing, so every downstream stage — routing, memory retrieval,
stored retrieval, live discovery, prompt assembly — sees only redacted text.
Capture deliberately keeps reading the ORIGINAL text, so a secret-bearing
message is still rejected outright by the policy rather than persisted as a
`[REDACTED]` stub. Covered by three tests in `tests/chat/test_chat_service.py`.

### 8.2 Fixed — HIGH: questions were stored as durable memory

`MemoryCapturePolicy.evaluate_user_message` rejected a question only when
nothing classified it. A question that happened to *contain* a classifier
phrase was therefore accepted and persisted as an assertion:

| Message | Stored as (before) |
| --- | --- |
| `should I drop the GAN baseline?` | `REJECTED_IDEA` |
| `what do I prefer for training frameworks?` | `PREFERENCE` |
| `can you remember what I decided about kuzu?` | `PROJECT_DECISION` |

The first row is the damaging one: it records the opposite of what the user
said, and rejected ideas carry weight in project-authority framing. The module
docstring already claimed bare questions were rejected, so this was an
implementation/contract mismatch rather than a design choice.

**Fix:** interrogative sentences — those that both end in `?` and open with an
interrogative word or a fronted auxiliary — are removed before classification,
and only the surviving statement is classified and persisted. A message that is
nothing but questions is rejected as `question_not_durable`. A tag question
attached to a decision (`We decided to go with SQLite - any objections?`) still
captures correctly, and a statement beside a question stores only the
statement. Covered by `TestInterrogativeSentencesAreNeverStored`.

### 8.3 Fixed — MEDIUM: `ChatBudget` bounds were unenforced at construction

Carried over from the adversarial audit (its finding 2). The `<= 12` discovery
bound existed only as a clamp inside `ChatService`, so any future call path
reaching ingestion another way lost it. `ChatBudget.__post_init__` now validates
`1 <= max_discovery_results <= 12` and rejects negative list bounds. The service
clamp is retained as defence in depth, and the tests that prove it now force the
hostile value onto the frozen instance rather than passing it to a constructor
that would reject it.

### 8.4 Fixed — MEDIUM: `HTTP_TIMEOUT_SECONDS` raised to 60

Changed from `20` to `60` at operator request, in `Settings.http_timeout_seconds`,
`providers.base.DEFAULT_HTTP_TIMEOUT_SECONDS`, `.env.example`, `README.md`, and
`tests/test_config.py`. The `0 < value <= 120` validation is unchanged. This
supersedes the parenthetical in `docs/phase_tasks/W9.md`, which was written when
the default was 20.

### 8.5 Fixed — MEDIUM: dead duplicate rule block in `chat/service.py`

`chat/service.py` carried a private `_SYSTEM_RULES` constant that nothing read;
the live prompt rules live in `chat/prompt.py`. The two had already drifted (7
rules versus 8), which is exactly how a stale copy misleads a future reader.
Removed.

### 8.6 Decided, not changed — `logger.exception` in the memory commands

`/memory-status` and `/memory-search` log tracebacks via `logger.exception`,
which could in principle include backend detail. Kept as-is: it is the
convention at all 25 exception sites in this codebase, logs are local to a
single-user daemon, and `GraphitiUserMemoryStore` degrades internally rather
than raising, so these handlers are effectively unreachable for backend
failures. Changing only these two would buy nothing and cost debuggability.

### 8.7 Decided, not changed — `MemoryStatus.persistence_path` in the status embed

Carried over from the adversarial audit (its finding 3). The absolute database
path appears in the `/memory-status` embed, which is owner-gated and ephemeral.
It is not a credential and it is the field an operator actually needs when
diagnosing a memory backend. Accepted.

### 8.8 Closed — capture of imperative research commands

The adversarial audit's finding 1 asked integration to confirm that
`MemoryCapturePolicy` rejects task-shaped imperatives. Verified against the
shipped policy: `find`, `search for`, `compare`, `summarize`, `show me`,
`list`, and `look up` phrasings all return `not_durable` and store nothing.
Pinned by `test_imperative_research_commands_not_stored`. No change needed.

### 8.9 Residual risks

1. **Authority ordering is prompt-enforced.** "Explicit project state outranks
   inferred user memory" is carried by section ordering and system rule 7, not
   by a mechanism the model cannot bypass. What *is* mechanically enforced is
   the citation namespace: user-memory facts never enter `allowed_paper_ids` or
   `allowed_gap_ids`, so a belief can never be cited as evidence regardless of
   what the model writes.
2. **Secret detection is shape-based.** `research_radar.memory.secrets` catches
   known vendor prefixes, bearer headers, AWS material, Discord tokens, and
   high-entropy opaque runs. A novel credential format could pass. It is a
   backstop, not a guarantee.
3. **`ChatRouter`'s LLM assist sends the turn text to the model** when no
   deterministic rule fires. The text is redacted first (8.1), but it is still
   an outbound call on an otherwise-conversational turn.
4. **Kuzu is deprecated upstream.** Unchanged from section 7; the swap to
   Neo4j/FalkorDB is one constructor call in `graphiti_store._build_default_client`.
5. **No live-credential E2E was run.** No credentials are present in this
   environment (verified by boolean presence checks only, never by printing
   values), so the real-provider path is proven by fakes and by the offline
   smoke scripts, not against live endpoints.
