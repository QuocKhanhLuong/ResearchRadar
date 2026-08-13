# ResearchRadar V1 Architecture

## Decision summary

ResearchRadar V1 should be a single Python 3.11+ process with one Discord bot,
one shared async HTTP client, one SQLite database, and APScheduler running in
that same process. Discord is an adapter, not the application core. The core
accepts normalized provider-neutral papers, does deterministic discovery work,
and exposes small services that Discord commands and scheduled jobs can call.

The implementation should deliberately optimize for reliable ingestion and
evidence-preserving storage, not for agent autonomy, scale, or a broad product
surface. The user is the only operator, but external APIs, PDFs, and remote LLM
responses must still be treated as unreliable inputs.

This document began as the V1 implementation design on 2026-08-13. The
implementation now exists under `src/research_radar`; use the README as the
current feature and operation guide. This document preserves the architectural
rationale, contracts, and deferred-scope decisions rather than acting as a
live implementation inventory.

## Constraints and non-goals

The following are hard constraints for every implementation decision:

- One private user and one private Discord server.
- Python application code is under src/research_radar using a src layout.
- SQLite through SQLAlchemy is the default durable store.
- External scholarly discovery is asynchronous and provider-neutral.
- Discord command handlers remain thin and domain modules never import Discord.
- Every outgoing HTTP call has an explicit timeout and a bounded failure mode.
- Tests use fake or mocked external systems; no credentials, model weights,
  network, GPU, or Discord connection are needed for the test suite.
- The default LLM provider must not fabricate a paper analysis when it is
  unconfigured.
- Discovery absence is never evidence of literature absence. Provenance must be
  retained wherever the current V1 data model can do so.

AGENTS.md should copy the 30 non-negotiable rules from the product brief,
including the single-user, no-secrets, provider abstraction, deterministic-first,
and post-phase test/lint requirements. It is an implementation guardrail, not
an optional contributor note.

## Approaches considered

### Approach A: synchronous command-oriented monolith

Each slash-command callback would call HTTP providers, parse PDFs, and use
SQLAlchemy directly.

Benefits:

- Fewest conceptual components and dependencies.
- Very quick to bootstrap.

Costs:

- Slow provider calls and PyMuPDF extraction occupy Discord command execution.
- It is easy for command modules to absorb research and persistence logic.
- Scheduled scans and interactions compete without a clear lifecycle boundary.

This is small enough for a prototype, but it weakens the most important
architecture rule: Discord must only be an interface.

### Approach B: one-process async application with narrow synchronous adapters

Discord, HTTP providers, the scheduler, and service orchestration are async.
SQLite remains regular synchronous SQLAlchemy because transactions are short and
the application has one user. Blocking database batches and PDF extraction are
run through asyncio.to_thread at their service boundary. A single Runtime
dataclass owns resources explicitly.

Benefits:

- Natural concurrency for the three scholarly providers and remote LLM calls.
- Long reads and scheduled scans do not need to block unrelated Discord events.
- Retains the simplicity of one process and normal SQLAlchemy/SQLite.
- Makes resource ownership and shutdown testable without introducing a DI
  framework.

Costs:

- Repository calls must not leak arbitrary sessions across threads.
- The application needs disciplined initialization and shutdown ordering.

This is the recommended approach.

### Approach C: async database, worker queue, and service split

Use an async SQLite/Postgres layer, task queue, separate scheduler process, and
possibly provider workers.

Benefits:

- Higher throughput and independent failure domains.

Costs:

- Adds a message broker, worker lifecycle, deployment complexity, and duplicate
  state coordination with no V1 benefit.
- Violates the stated one-process/no-Celery/no-Redis design philosophy.

Reject for V1. It can only become relevant after observed load or reliability
requirements justify it.

## Recommended component layout

Only create a module once the phase that uses it is implemented. The following
is the complete V1 landing layout; it is not a request to pre-create empty
packages.

~~~
src/research_radar/
  __init__.py
  main.py                 # runtime construction, process entrypoint, shutdown
  config.py               # Settings and validation
  logging.py              # standard logging setup and secret-safe formatting
  errors.py               # small shared domain error hierarchy

  bot/
    __init__.py
    client.py             # discord.Client/Bot lifecycle and command sync
    embeds.py             # pure presentation helpers for Discord embeds
    commands/
      ping.py
      paper.py
      read.py
      watch.py
      digest.py
      ask.py              # create only if Phase R is accepted

  models/
    paper.py              # Paper and RankedPaper
    document.py           # PaperDocument and extraction metadata
    paper_card.py         # PaperCard and EvidenceClaim

  providers/
    base.py               # PaperProvider protocol and provider errors
    openalex.py
    arxiv.py
    semantic_scholar.py

  research/
    dedup.py              # deterministic identity keys and metadata merge
    ranker.py              # transparent lexical ranking
    scout.py               # concurrent provider fan-out and partial failures
    service.py             # query validation and public discovery facade

  storage/
    database.py           # engine/session factory/schema initialization
    tables.py              # SQLAlchemy declarative table definitions
    repositories.py       # all SQLAlchemy query and transaction boundaries

  watch/
    service.py             # scan-once workflow and notification policy
    scheduler.py           # APScheduler job registration only

  reader/
    fetcher.py            # bounded HTTP PDF retrieval and supported URL forms
    parser.py             # PyMuPDF text extraction and section heuristics
    service.py            # fetch -> parse -> select evidence -> LLM -> persistence flow
    llm/
      base.py
      mock.py
      remote.py

  digest/
    service.py

  ask/
    service.py            # create only with the bounded Phase R implementation
~~~

main.py may contain a small Runtime dataclass rather than a container package.
It should hold Settings, the shared httpx.AsyncClient, SQLAlchemy session
factory/repository, composed services, scheduler, and Discord bot. This is
explicit dependency wiring, not a speculative dependency-injection framework.

## Domain contracts

### Paper normalization boundary

Providers return only the Pydantic Paper model. They do not return response
objects, raw JSON, XML elements, or provider-specific nested structures to
research code.

~~~
Paper
  id: str                         # provider record identity at provider boundary
  title: str
  abstract: str | None
  authors: list[str]
  publication_year: int | None
  venue: str | None
  doi: str | None
  url: str | None
  citation_count: int | None
  source: str                     # "openalex", "arxiv", or "semantic_scholar"
  external_ids: dict[str, str]    # known normalized identifiers only
~~~

The provider-local id should be namespaced, for example openalex:W123 or
arxiv:2401.01234, so that it cannot collide accidentally. It is not the
database primary key. The storage layer assigns a stable database id after
deduplication and returns a storage record for relationships such as PaperCard.

An adapter should normalize DOI values by removing common resolver prefixes,
trimming whitespace, and case-folding. It should normalize arXiv identifiers by
removing the version only for identity comparisons while retaining the original
identifier in source provenance. Empty strings become None; missing API fields
must never cause a provider-wide parsing failure.

OpenAlex inverted-index abstract reconstruction belongs entirely in
providers/openalex.py. arXiv Atom/XML parsing belongs in providers/arxiv.py.
Semantic Scholar field mapping belongs in providers/semantic_scholar.py. Before
writing each adapter, implementation must verify the current official API
documentation and record no guessed response fields in the domain layer.

### Provider and search result contracts

~~~
PaperProvider.search(query: str, limit: int = 10) -> list[Paper]

ProviderSearchResult
  papers: list[Paper]
  warnings: list[ProviderWarning]       # provider name and safe summary
  provider_counts: dict[str, int]

ResearchService.search(query: str, limit: int = 5) -> ProviderSearchResult
~~~

PaperProvider is a Python Protocol. Providers receive an injected shared
httpx.AsyncClient with an explicit httpx.Timeout; they do not own or close a
client themselves. Settings choose enabled providers. OpenAlex and arXiv can
work without credentials. Semantic Scholar is attempted only when enabled and
is allowed to run anonymously when the current API permits it; a missing key is
not a startup failure.

ScoutService runs enabled providers with asyncio.gather and
return_exceptions=True. It turns each expected provider failure into a warning,
logs technical detail, and continues with successful sources. If all enabled
providers fail, it raises one safe ProviderUnavailableError. A partial result
still returns ranking and may show a concise Discord note such as
"Some sources were unavailable; results are partial."

Search limits should be validated centrally: nonempty trimmed query, default
five display results, and a documented maximum such as 20. Scout may retrieve a
small bounded overfetch per provider, for example max(10, requested * 2) capped
at 25, then deduplicate and return the requested count. This preserves result
quality without uncontrolled API traffic.

### Deterministic deduplication

Deduplication uses no LLM or fuzzy semantic model. It builds stable identity
keys in this priority order:

1. normalized DOI;
2. versionless arXiv id;
3. any other recognized external provider id;
4. exact normalized title.

Title normalization uses Unicode NFKC normalization, case-folding, conservative
punctuation stripping, and whitespace collapse. It must not use approximate
matching. Empty or implausibly short titles are not title-match keys. A
union-find or equivalent grouping algorithm should handle a legitimate bridge
where records share different identifiers across providers.

Within a group, merge deterministically:

- retain all recognized external ids and provider source mappings;
- choose the richest nonempty abstract and most complete author list;
- retain a normalized DOI and canonical landing URL when available;
- use the maximum citation count rather than summing incomparable provider
  counts;
- use stable provider priority only to break otherwise equal conflicts;
- never silently concatenate contradictory scalar metadata.

The merged Paper remains a clean domain object. Its storage upsert independently
checks known source identifiers and normalized title so a later scan can merge a
paper first found through a different provider.

### Transparent ranking

Ranker returns RankedPaper containing a Paper, numeric score, and internal
reason components. It must be deterministic, documented, and stable under ties.
The initial score is deliberately lexical:

~~~
score =
  0.50 * query/title token-set overlap
  0.25 * query/abstract token-set overlap
  0.15 * recency_score
  0.07 * log-normalized citation_score
  0.03 * metadata_completeness_score
~~~

All components are clamped to [0, 1]. Recency can decay linearly over ten years
from the current calendar year; an unknown year contributes zero. Citation
normalization uses log1p and a fixed cap, so an old highly cited paper cannot
overwhelm title and abstract relevance. Ties sort by score, then newer year,
then citation count, then normalized title. The exact weighting should live in
one documented function and be covered by fixed expected-order tests. It is a
replaceable V1 ranker, not an embedding interface yet.

### Paper documents, cards, and evidence

~~~
PaperDocument
  title: str | None
  sections: dict[str, str]
  full_text: str
  source_url: str
  page_count: int
  warnings: list[str]

EvidenceClaim
  claim: str
  source_section: str | None
  supporting_text: str | None

PaperCard
  paper_id: str                 # storage paper UUID once persisted
  problem: str | None
  motivation: str | None
  contributions: list[str]
  methods: list[str]
  datasets: list[str]
  metrics: list[str]
  main_claims: list[EvidenceClaim]
  limitations: list[str]
  future_work: list[str]
  failure_cases: list[str]
~~~

PyMuPDF parsing is blocking and belongs behind asyncio.to_thread in the reader
service. Reader parsing accepts direct PDF bytes or a path through a defined
API; URL downloading is separate so tests can exercise parsing without network.
Bounded download and extraction defaults should include a PDF byte limit, page
limit, extracted text limit, and LLM input character limit. The fetcher accepts
only HTTP(S), follows only a small bounded number of redirects, streams with a
size cap, and checks PDF magic bytes rather than trusting Content-Type.

V1 supports only direct public HTTP(S) PDF URLs. An arXiv URL works only when
it is already a direct PDF URL; bare arXiv identifiers and abstract pages are
not resolved. It returns a clear unsupported-source error for arbitrary
publisher landing pages rather than scrape HTML. DOI resolution is an optional
additive resolver only after the direct PDF path is reliable.

Section detection is a line-based heuristic with normalized aliases for
Abstract, Introduction, Related Work, Method/Methodology, Experiments, Results,
Discussion, Limitations, Conclusion, and References. The parser must surface
poor extraction, especially likely scanned PDFs, rather than invoke an LLM on
near-empty text. OCR is out of scope.

Reader selects a bounded ordered set of useful sections: Abstract,
Introduction, Method, Experiments/Results, Discussion/Limitations, and
Conclusion. If a section is absent it uses safe, labeled excerpts; it never
sends an uncontrolled full PDF to a model.

LLMProvider exposes at least generate_structured(messages, response_model).
RemoteLLMProvider uses a documented OpenAI-compatible messages endpoint through
httpx, parses JSON content, and validates it with the supplied Pydantic model.
It makes no assumption about a named vendor or local model runner. MockLLMProvider
is a test double that returns an injected fixture; without an injected fixture
it raises LLMUnavailableError. Therefore LLM_PROVIDER=mock is safe as the
default and cannot fabricate a user-visible reading result.

The card prompt must require null evidence fields when unknown. After model
validation, reader code verifies every non-null source_section is among the
actual selected sections and every non-null supporting_text occurs in that
section after normalized comparison. Invalid claimed locations are cleared
along with their supporting text and logged; they are never presented as
evidence. A valid card is then persisted. No raw PDF, raw provider response,
token, or LLM authorization data is stored.

## Persistence design

Use SQLAlchemy declarative tables in storage/tables.py and one short-lived
Session per repository method/transaction. database.py initializes the
file-backed SQLite directory, engine, pragmas appropriate for a single process
(foreign keys and a bounded busy timeout; WAL where supported), and
Base.metadata.create_all(). Lightweight schema initialization is sufficient for
the first unreleased V1 schema. Introduce migrations only once schema upgrades
must preserve deployed data.

Repository methods are the only place with SQLAlchemy queries. Async services
call batch or potentially slow repository work through asyncio.to_thread; a
Session is created and closed inside that thread, never passed between tasks.
Small transactions and one-process scheduling keep SQLite lock contention
bounded.

| Table | Required fields and constraints | Purpose |
| --- | --- | --- |
| papers | id UUID/text PK; canonical_key UNIQUE; title; normalized_title indexed; abstract; authors JSON; publication_year; venue; doi indexed; url; citation_count; primary_source; first_discovered_at; created_at; updated_at | Canonical normalized metadata, never raw provider payloads. |
| paper_sources | id; paper_id FK; provider; external_id; source_url; retrieved_at; UNIQUE(provider, external_id) | Provider-specific identity and provenance after a merge. |
| watch_topics | id UUID/text PK; name; normalized_name UNIQUE; query; enabled; created_at; last_scan_at; last_error | Single-user saved query. There is no user_id. |
| watch_papers | watch_topic_id FK; paper_id FK; first_seen_at; last_seen_at; rank_score; notified_at; composite PK(topic, paper) | Per-topic unseen detection, activity, and notification idempotency. |
| paper_cards | paper_id PK/FK; problem; motivation; JSON arrays for contributions/methods/datasets/metrics/limitations/future_work/failure_cases; claims JSON; source_url; document_sha256; selected_sections JSON; llm_provider; llm_model; created_at; updated_at | One validated V1 analysis per paper plus analysis provenance. |
| digest_runs | id; period_start; period_end; status; paper_count; created_at; sent_at; safe_error | Prevents duplicate scheduled digests and records their covered interval. |

The repository needs these concrete operations:

- initialize_schema()
- upsert_merged_paper(paper) -> stored paper id
- get_papers_for_local_lexical_search(...)
- upsert_paper_card(card)
- add_watch_topic(name, query), list_watch_topics(), remove_watch_topic(id or
  unambiguous name)
- list_enabled_watch_topics()
- record_watch_discovery(topic_id, paper_id, rank_score) -> is_new
- list_pending_notifications(topic_id, cap), mark_notified(...)
- mark_watch_scan_success/failure(...)
- list_digest_candidates(period_start, period_end)
- get_last_successful_digest_end(), record_digest_run(...)

upsert_merged_paper must be transactional. It searches source ids, DOI, and
normalized title before inserting, updates useful missing metadata, and creates
source rows. PaperCard persistence is only attempted after a successful,
validated reader result. A failed reader does not leave a fake card.

## Runtime and resource lifecycle

The lifecycle should be explicit and ordered:

~~~
python -m research_radar.main
  -> load Settings and configure logging
  -> validate runtime-only requirements for launching Discord
  -> create SQLite engine/session factory and initialize schema
  -> create one AsyncClient with explicit timeout
  -> construct providers, repositories, research/reader/watch/digest services
  -> construct Discord bot with the service references
  -> register slash commands
  -> synchronize commands (guild scoped when DISCORD_GUILD_ID is set)
  -> start APScheduler jobs
  -> bot.start(DISCORD_TOKEN)

shutdown or startup failure
  -> stop scheduler without waiting for a new job
  -> close Discord bot
  -> close shared AsyncClient
  -> dispose SQLAlchemy engine
  -> log only safe lifecycle details
~~~

Missing DISCORD_TOKEN does not make Settings globally invalid: tests, package
imports, and non-bot utilities remain usable. The main bot entrypoint calls a
specific require_discord_token method and fails with a clear ConfigurationError.
Optional provider credentials never block startup. Secrets use Pydantic SecretStr
or equivalent and are not interpolated into logs.

Discord command synchronization uses a configured guild for development; when
DISCORD_GUILD_ID is unset it performs normal global sync. The bot requests only
the intents necessary for slash commands and explicitly does not enable Message
Content Intent.

APScheduler uses AsyncIOScheduler in the main event loop. Register two jobs:

- interval watch scan with WATCH_SCAN_HOURS, coalesce=True, max_instances=1;
- daily digest with a CronTrigger using DIGEST_HOUR and TIMEZONE, also
  coalesced and single-instance.

WatchService has its own asyncio.Lock in addition to max_instances=1, so direct
future triggers cannot overlap an active scan. A job exception is caught,
logged, and converted to a failed topic/run state; it must never terminate the
Discord process. If all provider searches fail for a topic, retain its previous
last_scan_at and save a safe last_error so the next interval can retry.

## Command-to-service flows

All commands defer an interaction when network, parsing, or LLM work can exceed
Discord's initial response deadline. Renderer functions receive already
normalized domain output and create embeds or plain text only.

### /ping

~~~
Discord command -> Ping handler -> immediate "ResearchRadar is online."
~~~

No provider, database, or scheduler call is needed.

### /paper query count

~~~
Paper command
  -> ResearchService.search(query, count)
  -> ScoutService concurrent enabled PaperProvider.search calls
  -> deterministic deduplication
  -> deterministic ranking and cap
  -> renderer
  -> Discord embeds: title, authors, year, venue, citations, DOI/canonical URL
~~~

The command does not persist ordinary ad-hoc searches. If some providers fail,
the response remains useful and labels results as partial rather than treating
absence as exhaustive discovery.

### /watch add, /watch list, /watch remove

~~~
Watch command -> WatchTopic repository operation -> renderer -> Discord
~~~

Use a Discord application command group named watch. add validates nonempty name
and query; list has no external calls; remove accepts a stable topic id or an
unambiguous normalized name. Topics have no Discord user id and no tenant
metadata.

### Scheduled watch scan

~~~
APScheduler
  -> WatchService.scan_all()
  -> for each enabled topic, ResearchService.search()
  -> dedup/rank (per query)
  -> repository upsert papers + record topic discoveries
  -> query pending, sufficiently ranked, unnotified discoveries
  -> NotificationSink (Discord implementation) if configured
  -> mark only successfully delivered notifications as notified
~~~

Scan topics sequentially to make rate behavior predictable; providers within one
topic remain concurrent. Persisted discoveries are not rolled back because a
Discord notification fails. Pending unnotified discoveries can be retried on a
later successful scan. Enforce a small per-topic and overall notification cap,
for example three per topic and ten per scan. A single optional
`DISCORD_CHANNEL_ID` identifies the private notification destination. When it is
missing, scanning and persistence still happen and the lack of delivery is
logged once per scan rather than treated as a fatal configuration error.

NotificationSink is a neutral protocol owned outside bot imports. The Discord
adapter implements it; watch/domain code never imports discord.py.

### /read url

~~~
Read command (defer)
  -> ReaderService.read_url(url)
  -> bounded PDF resolver/fetcher
  -> PyMuPDF parser in a worker thread
  -> useful-section selector with input cap
  -> LLMProvider.generate_structured(PaperCard)
  -> evidence-location verification
  -> repository upsert Paper and PaperCard
  -> renderer -> Discord structured summary
~~~

The Discord summary presents Problem, Main contribution, Method, Datasets, Main
claims, Limitations, and Future work. An unsupported URL, invalid PDF, poor
text extraction, unavailable LLM, malformed LLM output, or storage failure maps
to a concise actionable message and a technical log entry. The command never
inventories a result from the mock provider.

### /digest and scheduled digest

~~~
Discord /digest or APScheduler
  -> DigestService.build(period)
  -> repository-only recent discovery query
  -> deterministic sort using stored discovery rank and metadata
  -> group activity by watch topic
  -> shared digest renderer
  -> Discord response or configured notification channel
~~~

DigestService never triggers a new provider search. A scheduled digest starts
from the end of the last successful digest; the first successful run covers a
bounded recent window such as 24 hours. /digest is on-demand and uses the same
builder for the recent window without incorrectly marking a scheduled digest as
sent. If cards exist, a short stored contribution/problem can be included;
otherwise the digest only uses metadata.

### Conditional Phase R: /ask

Implement /ask only after the core flow is stable. The bounded version is:

~~~
Ask command (defer)
  -> local repository lexical retrieval over paper title, abstract, and stored
     PaperCard text
  -> select small labeled evidence snippets
  -> LLMProvider.generate_structured(EvidenceBackedAnswer)
  -> verify every returned citation belongs to supplied evidence
  -> renderer labels retrieved evidence and model synthesis separately
~~~

If no relevant stored evidence exists, say exactly that. It must say "within the
stored/retrieved corpus" and never imply comprehensive literature coverage. If
LLM is unavailable, return selected evidence or a clear unavailable message,
not a fabricated answer. No vector database or embedding dependency is justified
for this phase.

## Error and observability boundaries

Keep exceptions small and action-oriented:

- ConfigurationError for invalid required-at-runtime settings.
- ProviderUnavailableError for timeout, rate-limit, transport, or unavailable
  provider failures; ProviderResponseError for malformed external payloads.
- PaperNotFoundError or InvalidQueryError for safe domain input failures.
- PaperDownloadError and PaperParseError for reader boundaries.
- LLMUnavailableError for absent credentials/configuration; LLMResponseError
  for an invalid remote response.
- StorageError only when repository code cannot complete its transaction.

The Discord adapter translates known errors to concise user language and logs
the exception with provider, operation, and nonsecret identifiers. Unexpected
exceptions get a generic temporary-failure response plus traceback logging.
No handler exposes raw HTTP response bodies, authorization headers, token
values, stack traces, or provider-specific JSON.

Useful structured log events include startup/shutdown, command sync, provider
duration/failure/count, dedup input/output counts, rank results count, topic
scan start/end, persisted/new/notified counts, scheduler job failure, reader
byte/page/quality outcome, card validation outcome, and database transaction
failure. Log titles and URLs only if normal operational logging policy permits;
never log secrets or entire paper text.

## Configuration and deployment decisions

Settings use pydantic-settings with .env loading and environment-variable
override. .env.example contains placeholders only:

~~~
DISCORD_TOKEN=
DISCORD_GUILD_ID=
DISCORD_CHANNEL_ID=
DATABASE_URL=sqlite:///data/research_radar.db
OPENALEX_EMAIL=
SEMANTIC_SCHOLAR_API_KEY=
WATCH_SCAN_HOURS=6
DIGEST_HOUR=8
TIMEZONE=Asia/Bangkok
LLM_PROVIDER=mock
LLM_MODEL=
LLM_BASE_URL=
LLM_API_KEY=
~~~

Optional sensible settings include provider enable flags, HTTP/LLM timeout
seconds, and reader byte/page/text limits. They should have safe defaults rather
than making .env.example noisy. Settings validate positive scan intervals,
digest hour range, parseable timezone, URL scheme for configured LLM endpoint,
and SQLite/file database location. The implementation must create only the
configured database parent directory; it must not silently choose a second
database path.

.gitignore must ignore .env, virtual environments, Python/build/test caches,
SQLite database files plus WAL/SHM sidecars under data, and never ignore
.env.example. Before each commit, inspect tracked files for token-like values
and preserve untracked generated AgenTeam files unless their inclusion is
explicitly intended.

After the application is stable, Dockerfile is intentionally small:

- python:3.11-slim base;
- non-root application user where practical;
- install the package, not model weights;
- expose a volume-compatible data directory;
- run python -m research_radar.main;
- accept all secrets and database URL at runtime only.

Deployment documentation must state that Discord supplies bot identity only; a
cheap persistent CPU host runs this single process and owns durable SQLite
storage. The remote model endpoint remains external to the container.

## Execution contract and verification plan

Implementation proceeds in coherent commits. Do not commit generated
intermediate artifacts or unrelated pre-existing files. At every checkpoint run
pytest and ruff check ., fix failures before the next commit, and report exact
results rather than inferred status.

| Checkpoint | Delivered behavior | Required focused tests |
| --- | --- | --- |
| Bootstrap | pyproject, src import, Settings, logging, .gitignore, .env.example, README, AGENTS.md, test/Ruff setup | Settings defaults/invalid input, package import, pytest collection, ruff. Commit: chore: bootstrap ResearchRadar. |
| Discord shell | Runtime lifecycle, slash-only /ping, development/global sync strategy, clean closure | Command registration and fake lifecycle tests; no token needed. Commit: feat: add Discord bot shell. |
| OpenAlex discovery | Paper Pydantic model, provider protocol, OpenAlex normalization, ResearchService, /paper rendering | MockTransport success, inverted abstract, missing fields, timeout/error, service orchestration. Commit: feat: add OpenAlex paper discovery. |
| Multi-source discovery | arXiv and Semantic Scholar adapters; partial-failure fan-out | Atom/XML fixture parsing, Semantic Scholar fixture parsing, one-provider failure retains other results. Commit: feat: add multi-source paper discovery. |
| Dedup and rank | Identity keys, metadata merge, transparent score and stable order | DOI/arXiv/title positive and negative cases, bridge grouping, merge choices, newer-vs-citation ordering. Commit: feat: add paper deduplication and ranking. |
| SQLite memory | Schema initialization and repositories | Temp SQLite CRUD, upsert/source merge, transactional rollback, card storage. Commit: feat: add SQLite research memory. |
| Watchlists and monitoring | watch command group, scan_once service, idempotent discoveries, scheduler registration, notification cap | Watch CRUD, no user id, fake scout/notifier, lock/non-overlap, partial provider errors, notification retry. Commits: feat: add research watchlists; feat: add automatic paper monitoring. |
| Reader foundation | Bounded PDF fetch, PyMuPDF parsing, quality error, section extraction | Generated/minimal fixture PDF, heading aliases, empty/scanned-like PDF, byte/page/text limits. Commit: feat: add PDF paper parsing. |
| LLM and PaperCard | Provider protocol, mock/unavailable behavior, remote response validation, evidence checks | Canned card, unconfigured mock failure, malformed remote JSON, invalid citations/sections cleared. Commits: feat: add LLM provider abstraction; feat: add structured paper analysis. |
| /read | End-to-end fake URL/PDF -> parsed card -> persistence -> rendering | Fake HTTP transport, unsupported URL, parser/LLM/storage error mapping, successful persisted card. Commit: feat: add paper reading workflow. |
| Digest | Shared digest builder, on-demand command, scheduled state | No external provider call, period boundaries, activity grouping, duplicate scheduled-run protection. Commit: feat: add daily research digest. |
| Finish | README accuracy, Dockerfile, GAP_ENGINE_V2.md, optional stable /ask | Docker build if Docker is available; docs path checks; /ask only if core acceptance remains green. Commit: docs: document future gap engine, plus a separate Docker commit if needed. |

Provider tests should inject httpx.MockTransport rather than use the internet.
Discord tests should inject service fakes rather than require a bot token.
Reader tests may generate a tiny PDF with PyMuPDF in the test fixture to avoid
an opaque binary fixture. Scheduler tests call scan_once or invoke the job
function directly; they never sleep for six hours. Remote LLM tests use a mock
transport and JSON fixtures. Database tests use a temporary file database so
SQLite relationship and transaction behavior is real.

Final validation is:

~~~
pip install -e ".[dev]"
pytest
ruff check .
python -m compileall -q src
python -c "import research_radar; print(research_radar.__file__)"
git check-ignore .env
git status --short
~~~

Manual verification remains separate and must be labeled as such:

- /ping and command synchronization require a real Discord token and invited
  private bot.
- live OpenAlex/arXiv/Semantic Scholar queries require network access and may
  be rate-limited.
- remote reading requires a configured LLM endpoint.
- a production scheduler/notification check requires a configured Discord
  notification channel and a persistent host.

## Explicit YAGNI exclusions

Do not add any of the following in V1:

- user accounts, RBAC, tenants, billing, or SaaS infrastructure;
- Redis, Celery, message brokers, distributed workers, or Kubernetes;
- microservices, a service mesh, complex dependency injection, or a plugin
  system;
- Postgres, Neo4j, vector databases, FAISS, pgvector, or embedding pipelines;
- a web dashboard, FastAPI, REST API, email, Google Drive, or GitHub
  integrations;
- mandatory local MLX/Qwen downloads, GPU support, or model-serving process;
- OCR, figure extraction, multimodal PDF understanding, publisher web scraping,
  or automatic experiment generation;
- naive LLM gap detection, contradiction claims, coverage matrices, or
  autonomous coding agents;
- migrations, caching, retry queues, telemetry vendors, or elaborate retention
  policy before actual V1 data and operational needs exist.

## V2 Gap Engine preparation, not implementation

docs/GAP_ENGINE_V2.md should document a future normal-Python-services design
that consumes persisted PaperCards and fresh Scout queries. It must define a
CandidateGap with title, description, gap_type, research_question,
supporting_papers, conflicting_papers, evidence_count, novelty_score,
evidence_score, importance_score, feasibility_score, confidence, search_scope,
and caveats.

The future engine may evaluate explicit limitations/future work, coverage
matrices, contradiction candidates, evaluation gaps, and method-transfer gaps.
Every candidate must retain queries, sources searched, retrieval time, retrieved
papers, supporting evidence, and conflicting evidence. Its language must be
"limited evidence within the retrieved corpus", never "no one has studied X".

GapMiner output must pass a Critic service that creates alternative literature
queries, scouts again, looks for overlap, and downgrades, rejects, or preserves
the candidate. This is a bounded verification loop in normal Python services,
not a generic multi-agent framework. V1 stores enough normalized identifiers,
watch provenance, card claims, and source evidence to make this future work
possible without pretending it exists today.

## Risks to watch during implementation

| Risk | Mitigation |
| --- | --- |
| API response drift or rate limits | Verify official docs at adapter implementation time; normalize in one module; use timeout, safe errors, partial results, and MockTransport fixtures. |
| Duplicate or noisy notifications | Persist topic-paper discovery state, only mark notification after success, cap notifications, and guard scans with both lock and scheduler max_instances. |
| SQLite contention | One process, short repository transactions, per-call sessions, WAL/busy timeout, and no database work held across HTTP/LLM awaits. |
| Discord response expiry | Defer /paper, /read, /digest, and /ask before external work; keep /ping and local watch mutations immediate. |
| Large, scanned, or malformed PDFs | Stream/size/page/text caps; run parser off-loop; surface extraction quality failure; do not add OCR. |
| LLM hallucinated evidence | Default mock is unavailable, bounded evidence input, Pydantic validation, post-validation section/text checks, and clear distinction between evidence and synthesis. |
| Scope creep | Implement only the checkpoint feature; leave V2 directories absent until needed; retain one Runtime and direct constructor injection. |
