# ResearchRadar V1: Scope and Acceptance Strategy

## Product outcome

ResearchRadar V1 is a private, single-user Discord research assistant whose
reliable core is independent of Discord.  It must let its owner discover,
normalize, deduplicate, rank, persist, monitor, read, and digest scholarly
papers without pretending that a retrieved corpus is comprehensive.

The release is deliberately one process, one SQLite database, and a small set
of external providers.  Discord is a command and notification surface; it is
not the location of provider, ranking, storage, PDF, or LLM logic.

## Scope decision

### Must-have for the V1 release

1. A Python 3.11+ package that installs and validates locally without secrets,
   together with documented configuration, logging, and repository guardrails.
2. A slash-command-only Discord shell with `/ping`, `/paper`, watch commands,
   `/read`, and `/digest`.  It must not request Message Content Intent.
3. Provider-neutral paper discovery from OpenAlex, arXiv, and Semantic Scholar;
   partial provider failure must still return usable results from the others.
4. Deterministic deduplication and transparent deterministic ranking before any
   LLM call.
5. SQLite-backed paper memory, watch topics, discovery attribution, and
   persistable PaperCards behind repositories rather than raw SQL in domain
   services.
6. In-process scheduled watch scans and daily digests, with bounded Discord
   notifications and no overlapping scan for the same process.
7. Direct-PDF reading: bounded download/input, PyMuPDF extraction, heuristic
   sections, validated structured PaperCards through a provider-neutral LLM
   interface, and honest errors when analysis is unavailable.
8. A truthful README, `.env.example`, `AGENTS.md`, future Gap Engine V2 design
   note, and a small Dockerfile that does not contain secrets or model weights.
9. A deterministic automated suite and Ruff clean run.  No automated test may
   need Discord, a real scholarly API, a real LLM, a GPU, or downloaded model
   weights.

### Explicitly deferrable without blocking V1

* `/ask` is a conditional foundation, not a release blocker.  Ship it only if
  deterministic lexical retrieval, compact cited context, and an explicit
  “retrieved evidence vs. model inference” boundary are complete.  Otherwise
  document it as the immediate next task and do not expose a misleading command.
* DOI, arXiv-identifier, and publisher landing-page resolution for `/read` are
  optional.  V1 must support direct `http`/`https` PDF URLs; all other inputs
  may return a clear unsupported-input response.
* A local MLX/Qwen provider is optional.  The abstraction and remote
  OpenAI-compatible implementation are the V1 contract; no large local-model
  download is permitted during installation, tests, or CI.
* Live deployment to a named host, vector retrieval, embeddings, OCR, figure
  understanding, web UI, multi-user features, queues, workers, and migrations
  beyond lightweight idempotent schema initialization are out of scope.
* GapMiner, Critic, contradiction detection, coverage matrices, and research-gap
  scoring are documentation-only V2 work.  No generic “find gaps” LLM prompt is
  a V1 substitute.

## Release invariants

These rules are acceptance constraints across every phase:

* One user means no accounts, RBAC, tenancy, user tables, SaaS plumbing, Redis,
  Celery, Kubernetes, microservices, or a vector database.
* Domain modules never import Discord, and normalized domain models never expose
  raw provider JSON.
* Every outbound HTTP request has an explicit timeout.  Provider failures have
  meaningful technical logs and concise user-facing errors.
* Secrets are loaded only from the environment, never logged or committed;
  `.env` remains ignored and `.env.example` has placeholders only.
* “No result” means only that the configured providers did not retrieve a result.
  Paper summaries and future gap claims preserve provenance where available and
  never claim exhaustive coverage.
* Modules are added only when exercised by the V1 workflow.  Small repository
  interfaces and an application service composition root are sufficient; a
  dependency-injection framework is not.

## Phase acceptance matrix

The phase labels map to the requested build sequence.  A phase is complete only
when its listed behavior and its corresponding automated checks are complete.

| Phase | Release acceptance criteria |
| --- | --- |
| A — bootstrap | A clean Python 3.11+ virtual environment can run `pip install -e ".[dev]"`, `pytest`, and `ruff check .`.  The package imports without a Discord token.  The repository has src layout, pydantic-settings configuration, standard logging, test/Ruff configuration, ignored `.env`, placeholder-only `.env.example`, `data/.gitkeep`, README setup instructions, and `AGENTS.md` containing the supplied architecture rules. |
| B — Discord shell | Constructing the application initializes configuration and services, registers slash commands, and closes owned resources on shutdown.  `/ping` returns exactly `ResearchRadar is online.`.  A configured `DISCORD_GUILD_ID` uses guild-scoped development sync; absent it leaves global sync available.  No privileged Message Content Intent is requested, and a missing `DISCORD_TOKEN` produces a clear launch-time configuration error rather than an import error. |
| C/D — paper model and OpenAlex | `Paper` is a Pydantic, provider-neutral model with the requested fields and stable validation rules.  A `PaperProvider` protocol exposes async `search(query, limit)`.  OpenAlex uses async `httpx` with an explicit timeout, optionally sends its polite-pool email, reconstructs an inverted-index abstract, tolerates missing metadata, and turns response/API failures into a domain/provider error without exposing response JSON. |
| E — `/paper` | `/paper query:<text>` invokes a research service rather than provider code in the command.  It defaults to five results; if a count option exists it accepts only a documented bounded range (recommended 1–10).  Each rendered result contains title, authors, year, venue, citation count when available, and a DOI or canonical link.  It makes no LLM call and renders no raw provider payload. |
| F — multi-source discovery | arXiv and Semantic Scholar implement the same normalized interface and are optional at startup.  The scout invokes enabled providers concurrently, records non-sensitive warnings, and returns successful provider results even if another provider times out, rate-limits, or rejects an optional key.  Startup never requires a Semantic Scholar key or an OpenAlex email. |
| G — deterministic deduplication | The merge key order is: normalized DOI, arXiv ID, other known external IDs, then conservatively normalized title.  Duplicate merging is deterministic and preserves complementary metadata/provenance rather than arbitrarily discarding it.  It does not use an LLM or fuzzy algorithm that produces run-to-run ambiguity. |
| H — deterministic ranking | Ranking has a documented, unit-testable score composed of lexical query relevance, recency, citation signal, and metadata completeness.  Citation count cannot swamp a clearly more relevant recent paper.  The service returns results in stable order with optional internal score/reason metadata for diagnostics. |
| I — SQLite research memory | Schema initialization is idempotent and repositories, not domain services, own SQLAlchemy queries.  At minimum the database persists papers, provider-source IDs, watch topics, and PaperCards.  A compact discovery-attribution record is permitted because it directly supports per-topic unseen detection and digests.  No user table, tenant column, or migration system is required. |
| J — watchlist commands | A clean slash group or equivalent offers add/list/remove operations for `name`, `query`, `enabled`, and `last_scan_at`.  It validates empty/duplicate names, shows an intelligible empty list, and returns a useful response for an unknown removal target.  Watch topics have no Discord user ID. |
| K — automatic monitoring | APScheduler runs in the bot process at configured `WATCH_SCAN_HOURS`; it is started and stopped with the app.  Each enabled topic runs scout → dedup → rank → unseen comparison → persistence.  An async/process-local guard and scheduler configuration prevent overlapping scans.  A scan failure is logged and does not kill the bot or disable unrelated topics.  Notifications require a configurable destination such as `DISCORD_CHANNEL_ID`; when absent, new papers are still persisted and the skip is logged.  Notifications are capped and limited to sufficiently ranked unseen papers. |
| L — PDF parsing | The reader accepts a local PDF path or bytes through its internal API and produces `PaperDocument(title, sections, full_text, source_url)`.  It bounds bytes/pages/text, extracts text with PyMuPDF, normalizes it, detects common headings heuristically, and raises a clear `PaperParseError` for unreadable/poor extraction.  It neither OCRs scanned PDFs nor silently treats poor extraction as a sound basis for analysis. |
| M — LLM abstraction | `LLMProvider` supports validated structured generation independently of vendor.  `MockLLMProvider` supplies deterministic test fixtures; in normal unconfigured/mock application mode it must surface `LLMUnavailableError` instead of inventing an analysis.  `RemoteLLMProvider` targets a documented OpenAI-compatible endpoint through configured model/base URL/key without hard-coding Qwen, MLX, Hugging Face, or OpenAI. |
| N — structured PaperCards | Pydantic validates PaperCards and EvidenceClaims, including list defaults and nullable unknowns.  The reader selects and caps useful sections rather than sending an unbounded PDF.  An evidence claim’s `source_section` must be a detected section or null, and unknown supporting text is null/empty—not fabricated.  Valid cards persist with their paper and can be read back through repositories. |
| O — `/read` | `/read url:<paper-or-pdf-url>` supports direct HTTP(S) PDF URLs at minimum, stores its normalized paper/card result, and renders problem, contribution, method, datasets, claims, limitations, and future work when present.  URL retrieval rejects unsupported schemes, bounds redirects/size/time, and avoids private/local destinations; local-path support remains an internal API, not a Discord path escape.  Unparseable PDFs, unavailable LLMs, and unsupported URLs yield an actionable Discord response and no fabricated summary. |
| P/Q — digest | The scheduled daily digest uses already persisted discoveries and a timezone-aware `DIGEST_HOUR`; it does not launch an independent provider search.  It identifies papers since the persisted scheduled-digest cursor, ranks them, reports count/top papers/topic activity, and uses PaperCard insight only when it exists.  `/digest` delegates to the same service and does not duplicate formatting or query logic.  Automatic and on-demand paths do not repeatedly advance the same scheduled cursor. |
| R — `/ask` decision gate | Only implement if it can retrieve stored metadata/PaperCard evidence lexically, select a bounded context deterministically, cite the paper titles/IDs used, distinguish evidence from model inference, and say when evidence is insufficient.  It must never imply comprehensive literature coverage.  If any condition is missing, leave it unimplemented and document the gate rather than destabilizing the core release. |
| Documentation and packaging | README accurately distinguishes shipped features from planned work and gives local setup, Discord setup, configuration, persistent-data, and host-process instructions.  `docs/GAP_ENGINE_V2.md` documents CandidateGap provenance, cautious wording, explicit/coverage/contradiction/evaluation/method-transfer gaps, and the Critic re-search loop.  The Dockerfile is Python 3.11-slim, launches the module entry point as a non-root user where practical, accepts persistent mounted data, and contains neither secrets nor model weights. |

## Verification boundary

### Required automated validation

All of the following belong in ordinary local and CI runs, with no network or
credentials:

| Area | Required deterministic coverage |
| --- | --- |
| Configuration and package | Defaults, environment overrides, optional credentials, missing bot-token launch error, package import, `.env` exclusion check, and documented command smoke tests. |
| Providers | Mocked `httpx` transports cover request timeout configuration, query shaping, normal responses, sparse/malformed optional fields, inverted abstract reconstruction, non-2xx responses, invalid JSON, and timeouts for OpenAlex, arXiv, and Semantic Scholar. |
| Discovery core | Concurrent partial success, warnings, deterministic DOI/arXiv/external-ID/title dedup priority and merge behavior, ranking score/order/tie behavior, and bounded result counts. |
| Storage and watchlists | Temporary SQLite database schema initialization, paper/source/card round trips, discovery-attribution/unseen comparison, watch add/list/remove/enable behavior, and no user ownership fields. |
| Scheduler | Direct invocation of the scan/digest services with fakes, interval configuration, lock/no-overlap behavior, per-topic failure isolation, persistence of unseen results, notification caps, and no-channel graceful behavior.  Tests should not sleep for real intervals. |
| Reader and cards | Fixture/minimal generated PDFs for normal text, absent headings, empty/scanned-like extraction, size/page limits, section detection, PaperCard validation, selected-section truncation, and explicit parsing/LLM error paths. |
| LLM and command adapters | Mock LLM success plus unavailable/error behavior, mocked remote responses and validation failures, and fake Discord interaction/rendering tests for thin command delegation and concise errors.  No Discord gateway connection is part of `pytest`. |

The final local release gate is exactly the project’s documented test command,
`pytest`, followed by `ruff check .`, plus an import/module-entrypoint check in
an environment without real credentials.  Report pass/fail counts and commands
actually run; do not describe a mocked test as a live integration test.

### Manual/integration checks, only when configuration is available

These are valuable release checks but must remain separate from the automated
gate:

1. With a private development Discord guild and a real bot token, verify command
   synchronization, `/ping`, a bounded `/paper` response, watch command UX,
   and clean shutdown.  Without a token/guild, report this as unrun rather than
   blocking mock-verified development.
2. With network access, sample one query against each enabled scholarly provider
   to validate the current upstream contract, rate-limit behavior, and rendering.
   This is a smoke test only; it is not proof of exhaustive retrieval.
3. With a configured remote LLM and a permissible direct PDF, verify that `/read`
   returns validated, sourced fields and persists a PaperCard.  Do not test using
   local model downloads or production API keys in CI.
4. With a configured notification channel, run a one-shot scan and a one-shot
   scheduled-digest path; verify an unseen-paper notification is capped and a
   second scan does not resend the same discovery.
5. Build the Docker image and run it with a mounted empty data directory and
   environment file supplied at runtime.  Confirm SQLite survives a restart;
   no live host deployment is implied by this check.

## Release and operational risks

| Risk | V1 mitigation and release decision |
| --- | --- |
| Discord token, invite, gateway, or command-sync issues | Keep construction testable without Discord and run the manual private-guild check when credentials exist.  Missing credentials are a manual-test gap, not a reason to fake live evidence. |
| Provider rate limits, outages, or changing response contracts | Use bounded async timeouts, provider-specific normalization/tests, safe concurrent partial results, and warnings/logs.  Re-check official provider docs at implementation time; one outage must not take down discovery. |
| SQLite data loss on ephemeral hosting or duplicate work from multiple replicas | Document and require a persistent data mount/volume.  V1 is designed for one running process; do not advertise multi-replica safety. |
| Scheduler spam, overlap, or silent non-delivery | Use a local scan guard, `max_instances`-style scheduler protection, ranked caps, persisted unseen markers, configurable channel ID, and logs for skipped/failed delivery. |
| Arbitrary `/read` URLs create SSRF, unexpectedly large downloads, or parser pressure | Accept direct HTTP(S) PDFs only, reject local/private destinations and unsafe redirects, and enforce timeout/redirect/byte/page/text limits.  Treat scanned or poor text extraction as an explicit failure, not a result. |
| Remote LLM outage, cost, or hallucinated evidence | Make LLM optional, never emit mock analysis as real analysis, validate structured output, bound context, and require null/empty evidence locations when absent.  Retrieval provenance remains primary. |
| Scholarly conclusions overstate coverage or novelty | Render source links/IDs, retain provider provenance, and use qualified language: “limited evidence within the retrieved corpus,” never “no one has studied X.” |
| Dependency/API or packaging drift | Pin sensible compatible ranges, retain mock-contract tests, run the documented install/import/test/lint gates from a clean environment, and keep Docker generic rather than vendor-specific. |
| Scope expansion delays a reliable release | Treat `/ask`, local MLX, input resolution beyond direct PDFs, and all Gap Engine functionality as explicit gates/deferrals.  A smaller reliable core is the release priority. |

## Delivery sequencing and evidence

Implement and validate in the dependency order shown in the phase matrix.  Keep
the requested coherent commit boundaries, but create a commit only after the
phase’s tests and Ruff are green; never commit secrets or a knowingly broken
intermediate state.  The final handoff must list actual commits, exact validation
commands/results, manual checks actually performed, required versus optional
environment values, and any deferred gate (especially `/ask`).

The release claim is therefore precise: **ResearchRadar V1 provides a reliable
private research-ingestion and evidence-memory foundation, not an autonomous
research-gap oracle.**
