# ResearchRadar

ResearchRadar is a private, single-user research assistant. Discord is the
interface; discovery, normalization, ranking, paper reading, research memory,
and scheduled monitoring remain ordinary Python services.

V1 intentionally runs as one Python process with SQLite. It is not a SaaS,
multi-user system, agent framework, or web dashboard.

## Current V1 features

- Slash-command Discord bot with `/ping`, `/paper`, `/watch`, `/read`, and
  `/digest`.
- Concurrent discovery from OpenAlex, arXiv, and Semantic Scholar through
  provider-neutral models. A failed provider produces a warning; it is never
  treated as evidence that no relevant literature exists.
- Deterministic duplicate merging (DOI, arXiv ID, other known IDs, then a
  conservative title match) and transparent lexical/recency/citation ranking.
- SQLite-backed papers, provider-source provenance, watch topics, discovery
  state, PaperCards, and digest state.
- In-process APScheduler monitoring and daily digests over already persisted
  discoveries, with bounded notifications and overlap protection.
- Bounded direct-PDF download and PyMuPDF extraction, heuristic section
  detection, validated structured PaperCards, and an optional remote,
  OpenAI-compatible LLM endpoint.

`/ask` is deliberately **not implemented** in V1. Gap mining, contradiction
detection, coverage matrices, embeddings, vector databases, OCR, and
multi-agent workflows are also out of scope. See
[the future Gap Engine design](docs/GAP_ENGINE_V2.md) for the evidence-first
requirements before those features are considered.

## Architecture

```text
Discord slash commands and notifications
                 |
                 v
          Discord adapter layer
                 |
                 v
     Research services (no Discord imports)
       |             |                |
       v             v                v
providers      reader pipeline    watch/digest
OpenAlex       fetch -> parse     APScheduler
arXiv          -> bounded LLM        |
Semantic          |                  |
Scholar           +---------+--------+
                          |
                          v
              SQLAlchemy + SQLite research memory
```

The domain boundary uses normalized `Paper`, `PaperDocument`, and `PaperCard`
models. Provider response JSON and LLM-vendor details do not escape their
adapters.

## Local development

Python 3.11 or later is required.

```bash
git clone git@github.com:QuocKhanhLuong/ResearchRadar.git
cd ResearchRadar

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"

cp .env.example .env
pytest
ruff check .

python -m research_radar.main
```

Installing and running the automated suite do not require a Discord token,
scholarly API key, remote LLM, GPU, or model-weight download. Launching the
Discord bot does require `DISCORD_TOKEN`.

## Configuration

Copy `.env.example` to `.env`; `.env` is ignored by Git. Do not commit a token
or API key.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `DISCORD_TOKEN` | To launch the bot | — | Bot token from the Discord Developer Portal. |
| `DISCORD_GUILD_ID` | No | — | Development guild for faster guild-scoped command synchronization. Omit for global synchronization. |
| `DISCORD_CHANNEL_ID` | No | — | Private channel for scheduled watch and digest notifications. Scans still persist findings when it is unset. |
| `DATABASE_URL` | No | `sqlite:///data/research_radar.db` | SQLAlchemy database URL. Use a persistent path in deployment. |
| `OPENALEX_EMAIL` | No | — | Optional contact/polite-pool email for OpenAlex requests. |
| `OPENALEX_API_KEY` | No | — | Optional OpenAlex API key. |
| `SEMANTIC_SCHOLAR_API_KEY` | No | — | Optional Semantic Scholar API key. |
| `WATCH_SCAN_HOURS` | No | `6` | Watch scan interval; valid range is 1–168 hours. |
| `DIGEST_HOUR` | No | `8` | Local hour for the daily digest; valid range is 0–23. |
| `TIMEZONE` | No | `Asia/Bangkok` | IANA timezone used for scheduled digest timing. |
| `LLM_PROVIDER` | No | `mock` | `mock` is safe/default; use the configured remote mode for live structured analysis. |
| `LLM_MODEL` | With a remote LLM | — | Model name accepted by the OpenAI-compatible endpoint. |
| `LLM_BASE_URL` | With a remote LLM | — | Base URL for an OpenAI-compatible chat-completions endpoint. |
| `LLM_API_KEY` | Endpoint-dependent | — | Optional bearer credential for the remote endpoint. |
| `HTTP_TIMEOUT_SECONDS` | No | `20` | Bounded timeout for external HTTP requests. |

Scholarly credentials are optional where providers permit anonymous access.
They can improve rate limits or availability but are not needed to install or
test the application.

## Discord setup

1. Open the [Discord Developer Portal](https://discord.com/developers/applications)
   and create an application.
2. In **Bot**, create the bot and copy its token into `DISCORD_TOKEN` in your
   local `.env` file. Treat the token like a password.
3. In **OAuth2 -> URL Generator**, select `bot` and `applications.commands`,
   then use the generated URL to invite the bot to the private server.
4. Enable Discord Developer Mode if you want a development-only command sync,
   then copy the server ID into `DISCORD_GUILD_ID`. Optionally copy a private
   destination channel ID into `DISCORD_CHANNEL_ID` for scheduled delivery.
5. Start the application with `python -m research_radar.main`.

ResearchRadar uses application commands only and does not need the privileged
Message Content Intent.

## Commands

| Command | What it does |
| --- | --- |
| `/ping` | Returns `ResearchRadar is online.` |
| `/paper query:<text> [count]` | Searches enabled scholarly providers, deduplicates and ranks results, and renders up to 10 results (5 by default). |
| `/watch add name:<text> query:<text>` | Adds a single-user monitoring topic. |
| `/watch list` | Lists saved monitoring topics and their latest scan state. |
| `/watch remove topic:<name-or-id>` | Removes a monitoring topic. |
| `/read url:<direct-pdf-url>` | Downloads, extracts, and—when a remote LLM is configured—analyzes a direct public PDF. |
| `/digest` | Renders a recent digest from stored discoveries only; it does not run a new provider search. |

### `/read` limitation and LLM behavior

V1 accepts only a **direct, public HTTP(S) PDF URL** for `/read`. It does not
resolve a DOI, bare arXiv identifier, abstract page, or arbitrary publisher
landing page. Local paths are an internal parser capability and are not accepted
through Discord. Downloads are bounded and reject local/private destinations,
unsafe redirects, non-PDF payloads, scanned/poorly extractable PDFs, and overly
large inputs; V1 does not perform OCR.

With the default `LLM_PROVIDER=mock`, ResearchRadar deliberately does **not**
fabricate a PaperCard or paper summary. `/read` can extract the PDF but reports
that structured analysis is unavailable until a compatible remote model is
configured. The remote adapter validates structured output and preserves only
evidence locations that correspond to extracted sections.

## Persistence and scheduled work

The default database is `data/research_radar.db`. It is intentionally local and
single-process, so the `data/` directory must live on persistent storage in a
hosted deployment. Do not run multiple replicas against the same SQLite file.

Watch scans run at `WATCH_SCAN_HOURS`; each enabled topic performs
multi-provider discovery, deterministic deduplication/ranking, unseen-paper
persistence, and optionally a capped Discord notification. Failures are logged
and do not take down the bot. Daily digests use previously persisted discoveries
rather than performing a separate literature search.

## Deployment

GitHub stores the source code. Discord supplies the bot identity and token. A
separate, continuously running Python process must run on a persistent host
(for example, a small VPS, Railway, Render, or another worker-capable host):

```text
Discord
  |
  v
ResearchRadar process
  |- APScheduler
  |- SQLite on persistent storage
  `- optional remote LLM endpoint
```

The repository includes a small Docker image. It has no baked-in `.env`, API
keys, or model weights, runs as a non-root user, and exposes `/app/data` as its
persistent-data mount.

```bash
docker build -t research-radar .
docker run --rm \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  research-radar
```

The default `DATABASE_URL` resolves to the mounted `/app/data` directory. If
you override it, point it at persistent storage as well. The container does not
host an LLM or download model weights; configure a remote endpoint if paper
analysis is needed.

## Testing and validation boundaries

Run the deterministic local checks:

```bash
pytest
ruff check .
python -c "import research_radar; print(research_radar.__file__)"
git check-ignore .env
```

The test suite mocks Discord, scholarly APIs, and remote LLM calls. It covers
settings, provider parsing/failures, discovery orchestration, deduplication,
ranking, SQLite repositories, watch scans, scheduler behavior, PDF parsing,
LLM validation, PaperCards, digest behavior, and thin command adapters.

Automated checks are not live integration evidence. This README does not claim
that a live Discord gateway, upstream provider, remote LLM, notification
channel, or deployed Docker host has been manually validated. When credentials
are available, manually verify command synchronization and `/ping`, a bounded
`/paper` query, watch notification delivery, a direct-PDF `/read` with a remote
LLM, a scheduled digest, and SQLite persistence across a restart.

## Further design notes

- [V1 architecture rationale](docs/designs/researchradar-v1-architecture.md)
- [Provider and runtime contracts](docs/research/provider-and-runtime-contracts-v1.md)
- [V1 acceptance strategy](docs/strategies/researchradar-v1-acceptance-strategy.md)
- [Future Gap Engine V2](docs/GAP_ENGINE_V2.md)
