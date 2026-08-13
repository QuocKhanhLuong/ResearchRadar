# ResearchRadar V1: provider and runtime contracts

Verified 2026-08-13 against the linked first-party documentation. This is an implementation guide, not a reason to expose provider response objects outside `providers/`.

## Decisions for V1

- Make all three discovery providers optional at composition time and give each one a single provider-neutral `async search(query, limit) -> list[Paper]` boundary. `ScoutService` should run enabled providers concurrently, collect successful lists, and retain provider warnings; one failure must not turn an incomplete retrieval into “no papers exist.”
- Own one injected `httpx.AsyncClient` for the application lifetime, with an explicit timeout such as `httpx.Timeout(20.0, connect=5.0)`, and close it during bot shutdown. Do not create a new client per request.
- Treat external response text as untrusted display content. Never log credentials, authorization headers, or complete raw payloads.
- Use a namespaced, provider-stable `Paper.id` (for example, `openalex:W2741809807`, `arxiv:2301.01234`, `semantic_scholar:<paperId>`). Keep cross-provider identifiers in normalized `external_ids`; deduplication can then prefer DOI, arXiv ID, other IDs, and finally title.

## OpenAlex Works search

**Request.** `GET https://api.openalex.org/works`, with `search=<query>` and `per_page=<1..100>`. The list envelope is `{"meta": ..., "results": [...]}`. The `search` parameter searches work title, abstract, and full text. Use a narrow `select=` list for the fields below. [API overview](https://help.openalex.org/api/), [search](https://help.openalex.org/api/searching/).

**Credential change to account for.** Current official documentation says basic anonymous API use remains available, but a free API key raises the daily budget tenfold. It may be sent as `api_key` or a bearer token. Add optional `OPENALEX_API_KEY`; keep `OPENALEX_EMAIL` only if desired as a non-authenticating contact/User-Agent setting, not as a dependency or rate-limit mechanism. [Authentication and rate limits](https://help.openalex.org/api/authentication/).

| Normalized `Paper` field | OpenAlex field(s) that are safe to use | V1 mapping rule |
| --- | --- | --- |
| `id` | `id` | Take the final `W...` segment and namespace it as `openalex:`. Skip a malformed record with no ID or title. |
| `title` | `title` | Use `title`; it is documented as identical to `display_name`. |
| `abstract` | `abstract_inverted_index` | Reconstruct by allocating positions `0..max(position)`, placing each token at its listed positions, then joining. Missing/empty/malformed index becomes `None`, never a provider-wide failure. |
| `authors` | `authorships[].author.display_name` | Preserve order and omit missing names. The list is capped at 100 authors. |
| `publication_year` | `publication_year` | Pass through nullable integer. |
| `venue` | `primary_location.source.display_name` | Nullable. Do **not** use removed `host_venue`; OpenAlex now documents `primary_location`/`locations` instead. |
| `doi` | `doi` (fallback `ids.doi`) | Normalize away DOI-resolver prefixes and case for identity; preserve a canonical DOI value in the domain model. |
| `url` | `primary_location.landing_page_url`, `open_access.oa_url`, `doi`, `id` | Use this fallback order. `primary_location` is the closest copy to the version of record. |
| `citation_count` | `cited_by_count` | Nullable integer. |
| `external_ids` | `ids` | Normalize only known scalar IDs (at least `openalex`, `doi`, `pmid`, `pmcid`, `mag`) to strings; do not pass through arbitrary provider JSON. |

The work-object definitions, including inverted abstracts, title, IDs, locations, authorships, and citations, are documented in [Work attributes](https://help.openalex.org/data/works/attributes/).

**Failure/rate guidance.** `429` means either the daily budget or more than 100 requests/second; inspect `X-RateLimit-Remaining` and `X-RateLimit-Reset`, use bounded exponential backoff with jitter for `429`, timeouts, and `5xx`, and do not retry ordinary `400` validation errors. A `301` means a merged entity (relevant to future direct lookups). The documented error body contains `error` and `message`. [Error handling](https://help.openalex.org/api/errors/). A five-result V1 search needs only one request, so never page broadly by default.

## arXiv API search

**Request.** The current manual documents `GET http://export.arxiv.org/api/query` with `search_query`, `start` (zero-based), `max_results` (default 10), `sortBy`, and `sortOrder`. Build the query with `httpx` parameters rather than string concatenation; keep the base URL configurable because the official manual currently documents HTTP. V1 should request only its small result limit. [arXiv API User's Manual](https://info.arxiv.org/help/api/user-manual.html).

The response is Atom 1.0 XML, not JSON. Parse it with `xml.etree.ElementTree` using these namespaces:

```text
atom=http://www.w3.org/2005/Atom
arxiv=http://arxiv.org/schemas/atom
opensearch=http://a9.com/-/spec/opensearch/1.1/
```

| Normalized `Paper` field | Atom field(s) | V1 mapping rule |
| --- | --- | --- |
| `id` / `external_ids["arxiv"]` | `entry/id` | It is an abstract-page URL. Remove the URL prefix and a trailing `vN` only for the stable arXiv identifier; namespace the `Paper.id` as `arxiv:`. |
| `title` | `entry/title` | Collapse XML whitespace. |
| `abstract` | `entry/summary` | Collapse XML whitespace; nullable if absent. |
| `authors` | `entry/author/name` | Preserve Atom order. |
| `publication_year` | `entry/published` | Parse the ISO timestamp and take its year. `published` is first submission; `updated` is the retrieved version. |
| `venue` | `arxiv:journal_ref` | This is optional journal-reference text, not guaranteed structured venue metadata. Otherwise `None`. |
| `doi` | `arxiv:doi` | Prefer this exact DOI text over a resolved DOI link. |
| `url` | `entry/id` or `link[rel="alternate"]` | The documented `id` resolves to the abstract page. A `link` titled `pdf` is the PDF, not the canonical metadata URL. |
| `citation_count` | — | Set `None`; this API does not provide citation counts. |

The manual specifies that every response body, including errors, is an Atom feed. A semantic API error is a single `<entry>` whose title is `Error` and whose summary explains the problem, so do not mistake it for a paper or an empty search. It also documents `opensearch:totalResults`, `startIndex`, and `itemsPerPage`; V1 need not rely on them for its first-page discovery. [Response and error details](https://info.arxiv.org/help/api/user-manual.html#_3-2-the-api-response).

**Rate guidance.** arXiv explicitly asks callers making multiple requests to wait **three seconds** between calls. Enforce this in the provider with one lock and a monotonic last-request timestamp; do not let concurrent watch topics defeat the limit. Its documented ceiling is 30,000 total results and no more than 2,000 in a slice, far above V1's small search limit. [Paging and rate guidance](https://info.arxiv.org/help/api/user-manual.html#_3-1-1-2-start-and-max_results-paging).

## Semantic Scholar Academic Graph paper search

**Request.** `GET https://api.semanticscholar.org/graph/v1/paper/search` with required plain-text `query`, `limit`, and explicit comma-separated `fields`. The API has no special query syntax; its documentation specifically says hyphenated query terms yield no matches, so preserve the original query for provenance but send a hyphen-to-space normalized copy to this provider. `limit` defaults to 100 and must be at most 1,000; V1 should cap much lower. Search is relevance-ranked, limited to 1,000 results, and has a 10 MB response cap. [Official API reference](https://api.semanticscholar.org/api-docs/), [live Graph OpenAPI specification](https://api.semanticscholar.org/graph/v1/swagger.json).

Request exactly:

```text
fields=paperId,externalIds,title,abstract,authors,year,venue,url,citationCount
```

`paperId` is returned even when fields are omitted, but an explicit field list prevents accidental schema coupling. If configured, send the key only as the documented `x-api-key` request header; never put it in a URL or logs. [API access and rate policy](https://www.semanticscholar.org/product/api).

| Normalized `Paper` field | Semantic Scholar field(s) | V1 mapping rule |
| --- | --- | --- |
| `id` | `paperId` | Namespace as `semantic_scholar:`. |
| `title` | `title` | Required for a usable normalized result. |
| `abstract` | `abstract` | Nullable; the official schema says it can be missing for legal reasons even if the website displays one. |
| `authors` | `authors[].name` | Preserve order and omit missing names. |
| `publication_year` | `year` | Nullable integer. |
| `venue` | `venue` | Nullable string. |
| `doi` | `externalIds.DOI` | Normalize the DOI. |
| `url` | `url` | Semantic Scholar's canonical paper page. |
| `citation_count` | `citationCount` | Nullable integer. |
| `external_ids` | `paperId`, `externalIds` | Store the provider ID plus known scalar identifiers (notably `DOI`, `ArXiv`, `PubMed`, `PubMedCentral`, `DBLP`, `ACL`, `MAG`) after string conversion. |

The top-level batch is `total`, `offset`, optional `next`, and `data`. `total` is an approximate search result count and should not drive persistence or completeness claims. [Graph schema](https://api.semanticscholar.org/graph/v1/swagger.json).

**Rate guidance.** Semantic Scholar says most endpoints are public but share an unauthenticated limit and may be additionally throttled; API keys get higher limits, with the documented introductory key limit of 1 RPS. Treat neither number as a per-client throughput guarantee: serialize V1 calls per provider, cap retries, and turn `429`, timeouts, and `5xx` into a retryable provider failure rather than an empty result. A read-only, one-result keyless probe from this environment returned the documented JSON-like `{"message": ..., "code": "429"}` on 2026-08-13, so graceful degradation without a key is important. Do not retry `400` field/query errors; the OpenAPI schema documents these as `error` messages.

## Discord.py slash-command lifecycle

- Construct `commands.Bot(..., intents=discord.Intents.default())`. Slash-command-only V1 does not need Message Content, Members, or Presence intents. Discord classifies `MESSAGE_CONTENT` as privileged, and it is only needed when reading ordinary message content. [Discord Gateway intents](https://docs.discord.com/developers/events/gateway), [discord.py intent guide](https://discordpy.readthedocs.io/en/stable/intents.html).
- Register application commands on the bot's built-in `CommandTree`. Registration is local until `await tree.sync(...)` is called. [CommandTree API](https://discordpy.readthedocs.io/en/stable/interactions/api.html#discord.app_commands.CommandTree.sync).
- Put asynchronous startup in `Bot.setup_hook()`: discord.py calls it once after login and before the WebSocket connection, explicitly making it safer than `on_ready()` for service construction, command sync, and scheduler start. Do not await `wait_until_ready()` there because it deadlocks. [Bot lifecycle](https://discordpy.readthedocs.io/en/stable/ext/commands/api.html#discord.ext.commands.Bot.setup_hook).
- If `DISCORD_GUILD_ID` exists, use a development guild object, `tree.copy_global_to(guild=...)`, then `await tree.sync(guild=...)`. This copying is expressly a development aid and overwrites conflicting guild commands. Without it, call `await tree.sync()` for global commands. [Guild command FAQ](https://discordpy.readthedocs.io/en/stable/faq.html#how-do-i-restrict-a-command-to-a-specific-guild).
- Use `async with bot: await bot.start(token)` so shutdown closes the bot. The bot invite needs both `bot` and `applications.commands` scopes. [Bot context manager](https://discordpy.readthedocs.io/en/stable/ext/commands/api.html), [invite requirement](https://discordpy.readthedocs.io/en/stable/faq.html#my-bot-s-commands-are-not-showing-up).
- `/ping` can reply once with `interaction.response.send_message("ResearchRadar is online.")`. `/paper`, `/read`, `/digest`, and watch commands that may touch I/O must first `await interaction.response.defer(thinking=True)`, then edit the original response or send a follow-up. Discord requires the initial response within three seconds; an interaction can receive only one initial response. [Discord interaction deadline](https://docs.discord.com/developers/interactions/receiving-and-responding), [discord.py defer](https://discordpy.readthedocs.io/en/stable/interactions/api.html#discord.InteractionResponse.defer).
- Keep command modules as adapters only: validate Discord parameters, defer/respond/render embeds, and call an injected application service. Map domain/provider errors to concise responses while retaining technical context in logs.

## APScheduler in the single Discord process

- Pin `APScheduler>=3.11,<4` (current stable is 3.11.3). `AsyncIOScheduler` is the 3.x API and runs native coroutine jobs; v4 replaces it with the incompatible AnyIO-based `AsyncScheduler`. [3.x asyncio scheduler](https://apscheduler.readthedocs.io/en/3.x/modules/schedulers/asyncio.html), [current PyPI release](https://pypi.org/project/APScheduler/), [v4 migration guide](https://apscheduler.readthedocs.io/en/master/migration.html).
- Use the default in-memory job store. Recreate the two schedules at process start; persist only research state and watch topics in ResearchRadar's SQLite tables. APScheduler's own persistent SQLAlchemy job store is unnecessary for this one-process V1. [Scheduler/job-store guidance](https://apscheduler.readthedocs.io/en/3.x/userguide.html).
- Add an `async def` watch-scan job with stable ID `watch-scan`, `replace_existing=True`, `max_instances=1`, and `coalesce=True`; schedule it with an interval of `WATCH_SCAN_HOURS`. Add an `asyncio.Lock` inside the scan service as defense against a manual trigger and scheduler trigger racing. `max_instances` limits concurrent executions and `coalesce` collapses missed runs. [3.x job API](https://apscheduler.readthedocs.io/en/3.x/modules/schedulers/base.html).
- Add daily digest with a cron trigger using `ZoneInfo(settings.timezone)` and `DIGEST_HOUR`. Register a listener for `EVENT_JOB_ERROR`, `EVENT_JOB_MISSED`, and `EVENT_JOB_MAX_INSTANCES`, and also catch/log exceptions inside each job. Scheduler failures must be observable but must not terminate Discord.
- Start the scheduler once from `setup_hook()` after composing services. On bot close, call `scheduler.shutdown(...)` only if it is running, then close HTTP resources. APScheduler documents that shutdown does not interrupt jobs already running; bounded provider timeouts and job-level exception handling are therefore essential. [AsyncIOScheduler shutdown](https://apscheduler.readthedocs.io/en/3.x/modules/schedulers/asyncio.html).

## Tests implied by these contracts

- Use `httpx.MockTransport` with fixture JSON/XML: OpenAlex inverted abstracts and nullable nested metadata; arXiv normal/zero-result/error Atom feeds; Semantic Scholar explicit-field batch and `429`/malformed response cases.
- Test `ScoutService` with one provider failing and at least one succeeding; assert the result carries a warning and never claims the failure means absence.
- Unit-test command rendering and lifecycle choices without a Discord token. Test the scheduler job function and scan lock directly; do not wait six hours in a test.
