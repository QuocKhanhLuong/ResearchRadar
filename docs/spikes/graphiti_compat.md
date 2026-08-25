# Spike: Graphiti compatibility for user memory (W1)

Investigation only — no production code was changed. All signatures below were
resolved with `inspect` out of the installed package in the shared venv
(`/Users/alvinluong/ResearchRadar/.venv`, Python 3.12), not recalled from
memory. Reproduce with:

```bash
/Users/alvinluong/ResearchRadar/.venv/bin/python scripts/spike_graphiti.py
```

## 1. Versions and dependency compatibility

Installed in the shared venv at spike time:

| Package | Version | Notes |
| --- | --- | --- |
| graphiti-core | 0.29.3 | latest; matches coordinator's claim |
| kuzu | 0.11.3 | latest; matches coordinator's claim |
| pydantic | 2.13.4 | repo needs >=2.7 |
| pydantic_core | 2.46.4 | |
| pydantic-settings | 2.15.0 | repo needs >=2.3 |
| openai | 3.3.1 | graphiti-core requires `openai>=1.91.0`; 3.x satisfies it |
| tenacity | 9.1.4 | graphiti-core requires `>=9.0.0` |
| numpy | 2.5.2 | graphiti-core requires `>=1.0.0` |
| neo4j | 6.2.0 | pulled by graphiti-core (`neo4j>=5.26.0`); unused by us |
| posthog | 7.44.0 | telemetry dep of graphiti-core |

graphiti-core 0.29.3 core runtime requirements (from its dist metadata):
`neo4j>=5.26.0`, `numpy>=1.0.0`, `openai>=1.91.0`, `posthog>=3.0.0`,
`pydantic>=2.11.5`, `python-dotenv>=1.0.1`, `tenacity>=9.0.0`. Kuzu itself is
an optional extra of graphiti-core (`kuzu>=0.11.3; extra == 'kuzu'`), so our
proposed extra must name both packages explicitly.

**Pydantic verdict (highest-risk item): NO CONFLICT.** graphiti-core demands
`pydantic>=2.11.5`; this repo declares `pydantic>=2.7`. The shared venv has
2.13.4 which satisfies both. The one consequence: a fresh environment that runs
`pip install -e '.[memory]'` will have its effective pydantic floor raised to
**2.11.5** by the resolver (pip upgrades as needed; nothing pins below that).
That is compatible with every repo API in use (pydantic v2 throughout), but it
should be stated in review: installing the `memory` extra implies
pydantic >= 2.11.5.

No other conflicts found: httpx 0.28.1 (repo `>=0.27`), SQLAlchemy 2.0.52,
discord.py 2.7.1, APScheduler 3.11.3 all coexist; graphiti-core does not depend
on any of them.

## 2. Exact import paths and signatures

All copied verbatim from the installed package via `inspect.signature`.

```python
from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.nodes import EpisodeType
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.graphiti import AddEpisodeResults
```

```text
Graphiti.__init__(self, uri: str | None = None, user: str | None = None,
    password: str | None = None,
    llm_client: graphiti_core.llm_client.client.LLMClient | None = None,
    embedder: graphiti_core.embedder.client.EmbedderClient | None = None,
    cross_encoder: graphiti_core.cross_encoder.client.CrossEncoderClient | None = None,
    store_raw_episode_content: bool = True,
    graph_driver: graphiti_core.driver.driver.GraphDriver | None = None,
    max_coroutines: int | None = None, tracer: ... = None,
    trace_span_prefix: str = 'graphiti')

KuzuDriver.__init__(self, db: str = ':memory:', max_concurrent_queries: int = 1)
# async close(): no-op ("rely on GC"); Graphiti.close() -> await driver.close()

async Graphiti.build_indices_and_constraints(self, delete_existing: bool = False)

async Graphiti.add_episode(
    self, name: str, episode_body: str, source_description: str,
    reference_time: datetime.datetime,
    source: EpisodeType = EpisodeType.message,
    group_id: str | None = None,
    uuid: str | None = None,
    update_communities: bool = False,
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    previous_episode_uuids: list[str] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
    edge_type_map: dict[tuple[str, str], list[str]] | None = None,
    custom_extraction_instructions: str | None = None,
    saga: str | SagaNode | None = None,
    saga_previous_episode_uuid: str | None = None,
) -> AddEpisodeResults   # fields: episode, episodic_edges, nodes, edges,
                         #        communities, community_edges

async Graphiti.search(
    self, query: str, center_node_uuid: str | None = None,
    group_ids: list[str] | None = None,
    num_results=10,                      # DEFAULT_SEARCH_LIMIT
    search_filter: SearchFilters | None = None,
    driver: GraphDriver | None = None,
) -> list[EntityEdge]

async Graphiti.close(self)               # -> await self.driver.close()
```

EpisodeType members: `message`, `json`, `text`, `fact_triple`
(`graphiti_core.nodes.EpisodeType`).

**Gotchas verified empirically:**

- **A cross_encoder must be supplied explicitly.** If omitted,
  `Graphiti.__init__` eagerly constructs `OpenAIRerankerClient()`, which raises
  `openai.OpenAIError("Missing credentials...")` when no `OPENAI_API_KEY` is
  set — even though basic `search()` never uses it (RRF hybrid search). W3's
  adapter should pass a tiny local `CrossEncoderClient` stub whose
  `async rank(query: str, passages: list[str]) -> list[tuple[str, float]]`
  returns passages unchanged.
- **Always pass `group_id`.** When `None`, add_episode/search fall back to
  `get_default_group_id(provider)` which is `""` for Kuzu
  (`graphiti_core.helpers.get_default_group_id`). Episodes written without a
  group are invisible to `search(group_ids=['primary-user'])`.
- `reference_time` must be a timezone-aware `datetime` (use `.astimezone()`).
- `add_episode` performs LLM entity extraction + embedding; it cannot run
  offline. `build_indices_and_constraints` and raw `driver.execute_query` do.
- `execute_query(cypher, **params)` binds `$name` placeholders from kwargs and
  returns `(list[dict], None, None)`.

## 3. OpenAI-compatible custom base_url LLM client + embedder

This is exactly what this repo needs to talk to GoRouter through its existing
OpenAI-compatible config (see `src/research_radar/reader/llm/remote.py`).

```python
from graphiti_core.llm_client.config import LLMConfig          # plain class, not pydantic
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

llm_config = LLMConfig(
    api_key=..., model=..., base_url=...,      # base_url supported natively
    temperature=1.0, max_tokens=16384, small_model=None,
)
llm_client = OpenAIGenericClient(config=llm_config, structured_output_mode="json_object")
embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
    api_key=..., base_url=..., embedding_model=..., embedding_dim=1024))
```

Exact constructor signatures:

```text
OpenAIGenericClient.__init__(self, config: LLMConfig | None = None,
    cache: bool = False, client: Any = None, max_tokens: int = 16384,
    structured_output_mode: Literal['json_schema', 'json_object'] = 'json_schema')
LLMConfig(api_key=None, model=None, base_url=None, temperature=1,
    max_tokens=16384, small_model=None)
OpenAIEmbedder.__init__(self, config: OpenAIEmbedderConfig | None = None,
    client: AsyncOpenAI | AsyncAzureOpenAI | None = None)
OpenAIEmbedderConfig fields: embedding_dim, embedding_model, api_key, base_url
```

Notes for W3:

- With no explicit `client=`, `OpenAIGenericClient` builds
  `AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)` internally —
  so a custom GoRouter base URL needs no wrapper class at all.
- Prefer `structured_output_mode="json_object"` for GoRouter: `"json_schema"`
  mode sends native constrained-decoding response_format, which many
  OpenAI-compatible gateways reject (the repo's own remote provider already
  carries a `response_format` fallback for exactly this reason).
- `LLMConfig.small_model` exists because some clients pick a smaller model for
  cheap prompts; set it to the same model when only one is configured.
- Construction is offline-safe; first actual call hits the network. `search()`
  embeds the query even on an empty database, so search always requires a live
  embedder endpoint.

## 4. Custom entity/edge types

Supported in 0.29.x on **add_episode** only:

- `entity_types: dict[str, type[BaseModel]]` — name -> pydantic model.
- `edge_types: dict[str, type[BaseModel]]`,
  `edge_type_map: dict[tuple[src, dst], list[str]]`.
- `excluded_entity_types: list[str]` and
  `custom_extraction_instructions: str`.

The parameter names above are exact (copied from the signature).

**`Graphiti.search()` has NO entity_types parameter.** Type-based narrowing at
query time goes through `SearchFilters`:
`node_labels: list[str]`, `edge_types: list[str]` (label strings),
plus date filters (`valid_at`/`invalid_at`/`created_at`/`expired_at` as
`list[list[DateFilter]]`), `edge_uuids`, and `property_filters`. Simplest
pattern for W3: extract with default types, filter returned `EntityEdge`s in
Python, or pass `SearchFilters(node_labels=[...])`.

## 5. Temporal fields on returned edges

`graphiti_core.edges.EntityEdge` (pydantic model) fields, verbatim:
`uuid, group_id, source_node_uuid, target_node_uuid, created_at, name, fact,
fact_embedding, episodes, expired_at, valid_at, invalid_at, reference_time,
attributes`.

Read them as plain attributes after search:

```python
edges = await graphiti.search(query, group_ids=["primary-user"])
for e in edges:
    e.fact            # str — extracted fact sentence
    e.created_at      # datetime — ingestion time (non-null)
    e.valid_at        # datetime | None — when the fact became true
    e.invalid_at      # datetime | None — when it stopped being true
```

These map directly onto W2's `MemoryFact(valid_at, invalid_at, fact)`.
`expired_at` is a separate soft-delete marker; treat `invalid_at or expired_at`
as "no longer current" if temporal filtering is wanted later.

## 6. Kuzu persistence across restarts

**Verified experimentally by scripts/spike_graphiti.py**: a `KuzuDriver(db=<path>)`
writes a **single file** at `<path>` (~1.1 MB right after schema creation; it is
a file, not a directory tree). The script creates the DB + indices, inserts an
Episodic row via raw Cypher, then re-opens the same path in a **child process**
and reads the row back: persistence SURVIVES the restart (10 tables, 1 row).
No checkpoint API call was needed; dropping references + GC sufficed, and an
explicit `CHECKPOINT;` attempt is wrapped in try/except in case future versions
change that.

Schema tables created by `build_indices_and_constraints()` / `setup_schema()`:
`Community, Entity, Episodic, HAS_EPISODE, HAS_MEMBER, MENTIONS, NEXT_EPISODE,
RELATES_TO, RelatesToNode_, Saga`.

Caveats: `KuzuDriver(db=':memory:')` obviously does not persist; concurrent
processes opening the same file is untested and should be avoided (single
daemon process owns the file); `max_concurrent_queries=1` (default) serialises
queries through one `kuzu.AsyncConnection` (which internally dispatches to a
thread-pool executor, so it does not block the event loop).

## 7. Dependency proposal for pyproject.toml

```toml
memory = [
    "graphiti-core>=0.29",
    "kuzu>=0.11",
]
```

- Matches PHASE_CONTRACTS.md §13 exactly.
- No changes to core `dependencies`; `pip install -e .` keeps working without
  the extra (verified conceptually: graphiti/kuzu are not imported anywhere in
  `src/`; W3 must keep imports lazy inside the adapter).
- Conflict analysis: none blocking. Side effects of installing the extra:
  effective pydantic floor becomes 2.11.5 (see §1), plus new transitive deps
  `openai>=1.91.0`, `tenacity>=9.0.0`, `numpy>=1.0.0`, `neo4j>=5.26.0`
  (imported lazily by graphiti-core, unused with Kuzu), `posthog>=3.0.0`.
  posthog may attempt outbound telemetry; if that matters, set env
  `POSTHOG_DISABLED=1` at the adapter boundary (untested here — flagging only).
- Suggested pin style stays `>=` per contract; if the team wants reproducibility
  later, add upper bounds in a lock step, not in this extra.

## 8. Deviation record

- **Kuzu is deprecated upstream.** `KuzuDriver.__init__` emits
  `DeprecationWarning: The Kuzu backend is deprecated and will be removed in a
  future release — the upstream Kuzu project is no longer maintained. Migrate
  to Neo4j or FalkorDB.` (emitted by graphiti-core at
  `graphiti_core/driver/kuzu_driver.py:148`; plain `kuzu.Database()` does not
  warn). It remains present and functional in graphiti-core 0.29.3.
- **Why we keep it:** ResearchRadar is a single-user daemon (AGENTS.md rule 10);
  Neo4j/FalkorDB require operating a server process, while Kuzu is the only
  embedded/zero-server backend and persists to one file. Per phase contracts,
  W3 suppresses exactly this warning category at the adapter boundary.
- **Migration escape hatch:** the adapter talks to `Graphiti` +
  `GraphDriver` abstractions, so swapping `KuzuDriver` for
  `Neo4jDriver(uri, user, password)` or FalkorDB later is a construction-site
  change (settings-driven), not a call-site change. User data migration would
  need re-ingestion or an export/import over Cypher; acceptable for advisory,
  rebuildable personal context. Revisit if Kuzu removal actually lands in a
  graphiti-core release we upgrade to.
