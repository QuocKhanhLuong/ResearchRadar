# Phase Contracts — Personal Research Chat & Memory

This document is the **binding interface contract** for the
`feat/personal-research-chat-memory` phase. Every parallel worker implements
against the names, paths, and signatures written here. Do not rename a module,
class, field, or setting defined below; if a contract is wrong, escalate to the
coordinator instead of unilaterally changing it.

## 1. Architectural ownership

| Store | Owns |
| --- | --- |
| SQLite | Canonical scientific/research state: papers, PaperCards, gaps, projects, provenance |
| Pinecone | Derived semantic retrieval index; rebuildable, never canonical |
| Graphiti (Kuzu) | Personal/user temporal context memory; advisory only |
| LLM | Synthesis, extraction, reasoning |
| Discord | Thin interaction boundary; no research orchestration logic |

**Authority rule.** Explicit SQLite project state (`Project.constraints`,
`Project.hypotheses`, `Project.rejected_ideas`) outranks inferred Graphiti user
memory whenever they conflict. A user belief or preference recovered from
Graphiti must never be rendered as scientific evidence or as a published
finding.

## 2. Package layout

```
src/research_radar/memory/
    __init__.py          # re-exports the public surface
    models.py            # W2 — memory value types
    base.py              # W2 — UserMemoryStore protocol
    disabled.py          # W2 — DisabledUserMemoryStore
    fakes.py             # W2 — FakeUserMemoryStore (test double, shipped in src)
    capture.py           # W4 — MemoryCapturePolicy
    secrets.py           # W4 — secret detection/redaction
    graphiti_store.py    # W3 — GraphitiUserMemoryStore
src/research_radar/chat/
    __init__.py          # re-exports the public surface
    models.py            # W5 — ChatRequest / ChatResponse / ChatMode
    router.py            # W5 — ChatRouter
    evidence.py          # W6 — evidence packet assembly + ID validation
    prompt.py            # W8 — bounded prompt construction
    service.py           # W6 — ChatService
src/research_radar/bot/
    mention.py           # W7 — mention parsing + admission filter
    commands/memory.py   # W10 — /memory-status, /memory-search
```

Test layout mirrors it: `tests/memory/`, `tests/chat/`, `tests/bot/`,
`tests/e2e/`, `tests/adversarial/`.

## 3. `research_radar.memory.models` (W2)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

class MemoryClass(StrEnum):
    PREFERENCE = "preference"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    RESEARCH_INTEREST = "research_interest"
    PROJECT_DECISION = "project_decision"
    REJECTED_IDEA = "rejected_idea"
    TOOL_PREFERENCE = "tool_preference"
    WORKFLOW_PREFERENCE = "workflow_preference"
    RESEARCH_DIRECTION = "research_direction"
    TEMPORAL_PLAN = "temporal_plan"

@dataclass(frozen=True, slots=True)
class MemoryFact:
    """One advisory user-context fact recovered from the memory backend."""
    fact: str
    memory_class: MemoryClass | None = None
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    source: str = "graphiti"
    score: float | None = None

@dataclass(frozen=True, slots=True)
class UserMemoryContext:
    """Bounded, advisory user context handed to prompt assembly."""
    facts: tuple[MemoryFact, ...] = ()
    backend: str = "disabled"
    degraded: bool = False

    @property
    def available(self) -> bool:
        return bool(self.facts)

@dataclass(frozen=True, slots=True)
class MemoryStatus:
    """Safe-to-display backend health. Must never contain credentials."""
    backend: str
    enabled: bool
    healthy: bool
    detail: str = ""
    persistence_path: str | None = None
    episode_count: int | None = None
```

## 4. `research_radar.memory.base` (W2)

```python
class UserMemoryStore(Protocol):
    """Async personal-memory boundary. Never a source of scientific evidence."""

    @property
    def backend_name(self) -> str: ...

    @property
    def enabled(self) -> bool: ...

    async def add_episode(
        self,
        content: str,
        *,
        source_description: str = "discord-chat",
        reference_time: datetime | None = None,
        memory_class: MemoryClass | None = None,
    ) -> bool:
        """Persist one durable user episode. Returns False when not stored."""

    async def search(self, query: str, *, limit: int = 8) -> list[MemoryFact]: ...

    async def get_context(self, query: str, *, limit: int = 8) -> UserMemoryContext: ...

    async def status(self) -> MemoryStatus: ...

    async def close(self) -> None: ...
```

Every implementation must satisfy these invariants:

- No method raises to callers. Backend failure yields `degraded=True`,
  an empty `UserMemoryContext`, `add_episode() -> False`, or an unhealthy
  `MemoryStatus`. Log a sanitized warning at most once per failure class; no
  retry storms.
- No exception message, log line, or `MemoryStatus.detail` may echo the
  episode content or any credential.

`DisabledUserMemoryStore` returns `enabled=False`, empty results, and
`add_episode() -> False`. `FakeUserMemoryStore` is a deterministic in-memory
implementation used by tests and offline smoke scripts; it supports seeding
facts and asserting captured episodes, and it can be constructed in a
`fail=True` mode that simulates a total backend outage.

## 5. `research_radar.memory.capture` and `.secrets` (W4)

```python
@dataclass(frozen=True, slots=True)
class CaptureDecision:
    should_store: bool
    memory_class: MemoryClass | None
    reason: str          # short, safe-to-log; never echoes rejected content
    redacted_text: str   # the text safe to persist ("" when should_store is False)

class MemoryCapturePolicy:
    def __init__(self, *, enabled: bool = True) -> None: ...
    def evaluate_user_message(self, text: str) -> CaptureDecision: ...
    def evaluate_assistant_message(self, text: str) -> CaptureDecision:
        """Always returns should_store=False; assistant output is never memory."""
```

`secrets.py` exposes:

```python
def contains_secret(text: str) -> bool: ...
def redact_secrets(text: str) -> str: ...
```

Detection must cover at minimum: Discord bot tokens, `sk-`/`sk-ant-`/`pcsk_`
style API keys, generic `Bearer <token>` headers, `Authorization:` headers,
`AWS_SECRET_ACCESS_KEY`-shaped assignments, `password=`/`passwd=`/`token=`/
`api_key=` assignments, and long high-entropy opaque strings. When a secret is
detected, the whole message is rejected for storage — never log or re-emit the
matched substring.

Reject also: stack tracebacks, raw PDF text dumps, shell output blocks, and
messages that carry no durable signal (greetings, thanks, one-word replies).

## 6. `research_radar.chat.models` (W5)

```python
class ChatMode(StrEnum):
    CONVERSATIONAL = "conversational"
    PERSONAL_MEMORY = "personal_memory"
    RESEARCH_STORED = "research_stored"
    RESEARCH_LIVE = "research_live"
    PROJECT_RESEARCH = "project_research"

class EvidenceScope(StrEnum):
    NONE = "none"
    USER_MEMORY = "user_memory"
    STORED_CARDS = "stored_cards"
    STORED_METADATA = "stored_metadata"
    DISCOVERY_METADATA = "discovery_metadata"
    MIXED = "mixed"

@dataclass(frozen=True, slots=True)
class ChatRequest:
    text: str
    discord_user_id: str | None = None
    channel_id: str | None = None
    message_id: str | None = None
    project_hint: str | None = None

@dataclass(frozen=True, slots=True)
class ChatResponse:
    text: str
    mode: ChatMode
    paper_ids: tuple[str, ...] = ()
    gap_ids: tuple[str, ...] = ()
    used_user_memory: bool = False
    live_discovery_used: bool = False
    evidence_scope: EvidenceScope = EvidenceScope.NONE
    degraded: bool = False
```

## 7. `research_radar.chat.router` (W5)

```python
@dataclass(frozen=True, slots=True)
class RouteDecision:
    mode: ChatMode
    needs_user_memory: bool
    needs_stored_research: bool
    allows_live_discovery: bool
    search_query: str          # normalized retrieval query ("" when not research)
    project_hint: str | None = None

class ChatRouter:
    def __init__(self, *, llm_provider: LLMProvider | None = None) -> None: ...
    async def route(self, request: ChatRequest) -> RouteDecision: ...
```

Deterministic rules run first and are authoritative for the clear cases:

- Greeting / small talk / meta question about the bot -> `CONVERSATIONAL`,
  `needs_stored_research=False`, `allows_live_discovery=False`.
- First-person memory question ("what do I", "my interests", "remember",
  "my preferences", "what am I working on") -> `PERSONAL_MEMORY`,
  `allows_live_discovery=False`.
- First-person durable statement ("I prefer", "I don't want to pursue",
  "my goal is") -> `CONVERSATIONAL` with `needs_user_memory=True`; capture is
  decided later by `MemoryCapturePolicy`.
- Research intent ("find work on", "recent papers", "compare X and Y",
  "state of the art", "survey", "literature") -> `RESEARCH_STORED` with
  `allows_live_discovery=True`.
- Explicit project reference (`project_hint` set, or "my project X")
  -> `PROJECT_RESEARCH`.

The LLM assist is optional refinement only. If the LLM is absent or fails, the
deterministic decision stands; the router never raises.

`RESEARCH_LIVE` is set by `ChatService`, not by the router: the router grants
permission (`allows_live_discovery`) and the service decides from stored
evidence sufficiency.

## 8. `research_radar.chat.evidence` (W6)

```python
@dataclass(frozen=True, slots=True)
class StoredEvidenceItem:
    paper_id: str            # canonical SQLite id, always resolved
    title: str
    year: int | None
    venue: str | None
    abstract: str | None
    has_paper_card: bool
    card_summary: str | None = None

@dataclass(frozen=True, slots=True)
class DiscoveryEvidenceItem:
    paper_id: str            # canonical SQLite id after ingestion; never a raw provider id
    title: str
    year: int | None
    venue: str | None
    abstract: str | None

@dataclass(frozen=True, slots=True)
class ProjectMemory:
    project_id: str
    name: str
    constraints: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    rejected_ideas: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class EvidencePacket:
    stored: tuple[StoredEvidenceItem, ...] = ()
    discovery: tuple[DiscoveryEvidenceItem, ...] = ()
    gap_ids: tuple[str, ...] = ()
    project: ProjectMemory | None = None
    user_memory: UserMemoryContext = UserMemoryContext()
    live_discovery_used: bool = False

    @property
    def allowed_paper_ids(self) -> set[str]: ...
    @property
    def allowed_gap_ids(self) -> set[str]: ...
    @property
    def evidence_scope(self) -> EvidenceScope: ...
```

**Hard rule:** every id in `stored` and `discovery` must have been read back
from SQLite. A semantic/vector hit is a candidate id only; it becomes evidence
only after SQLite resolution. Citation validation drops any id from the LLM
answer that is not in `allowed_paper_ids` / `allowed_gap_ids`.

## 9. `research_radar.chat.prompt` (W8)

```python
def build_chat_prompt(
    request: ChatRequest,
    decision: RouteDecision,
    packet: EvidencePacket,
) -> list[LLMMessage]: ...
```

Sections appear in this exact order with these exact headers:

```
SYSTEM RULES
USER MEMORY (ADVISORY — NOT SCIENTIFIC EVIDENCE)
EXPLICIT PROJECT MEMORY (CANONICAL USER/PROJECT STATE)
STORED SCIENTIFIC EVIDENCE (CANONICAL)
LIVE DISCOVERY EVIDENCE (METADATA/ABSTRACT-LEVEL ONLY)
QUESTION
```

Empty sections are omitted entirely, except `SYSTEM RULES` and `QUESTION`.
The system rules must state, verbatim in substance:

- Never treat USER MEMORY as scientific support or as a published result.
- Never treat a user hypothesis or belief as an established finding.
- Never claim the literature contains no work on a topic; the corpus is partial.
- Cite only the paper ids and gap ids listed in the evidence sections.
- Distinguish stored PaperCard evidence from abstract-only evidence, and say
  explicitly when an answer rests only on abstract/discovery-level metadata.
- Say when the available evidence is insufficient rather than inventing an
  answer.
- When EXPLICIT PROJECT MEMORY and USER MEMORY conflict, EXPLICIT PROJECT
  MEMORY wins.

## 10. `research_radar.chat.service` (W6)

```python
@dataclass(frozen=True, slots=True)
class ChatBudget:
    max_stored_evidence: int = 8
    max_discovery_results: int = 10   # hard-clamped to 12
    max_user_memory_facts: int = 8
    stored_sufficiency_threshold: int = 3
    auto_read_pdfs: int = 0           # must remain 0 in this phase

class ChatService:
    def __init__(
        self,
        *,
        repository: ResearchRepository,
        router: ChatRouter,
        user_memory: UserMemoryStore,
        capture_policy: MemoryCapturePolicy,
        llm_provider: LLMProvider | None = None,
        ingestion_service: IngestionService | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        semantic_index: SemanticIndex | None = None,
        budget: ChatBudget | None = None,
    ) -> None: ...

    async def chat(self, request: ChatRequest) -> ChatResponse: ...
```

Pipeline, in order:

1. Normalize the query; empty text returns the usage hint without any backend call.
2. `router.route(request)`.
3. If `needs_user_memory`: `user_memory.get_context(...)` (bounded, never raises).
4. If `PROJECT_RESEARCH` or `project_hint`: load `ProjectMemory` from SQLite.
5. If `needs_stored_research`: hybrid stored retrieval (lexical + optional
   semantic; a Pinecone outage degrades to lexical, never fails).
6. If `allows_live_discovery` and stored evidence count is below
   `stored_sufficiency_threshold`: one bounded
   `ingestion_service.ingest_research_topic(query, limit=min(budget, 12), auto_read=0)`
   call, then re-resolve the returned canonical ids from SQLite. Mode becomes
   `RESEARCH_LIVE`. Never more than one discovery call per chat turn.
7. `build_chat_prompt(...)` and one `llm_provider.generate_structured(...)` call.
8. Validate cited ids against the packet; drop unknown ids.
9. Build `ChatResponse`.
10. Capture: `capture_policy.evaluate_user_message(request.text)`; if
    `should_store`, `await user_memory.add_episode(decision.redacted_text, ...)`.

Failure behaviour:

- LLM missing or failing: return a concise safe error (no invented answer),
  `degraded=True`; if live discovery actually persisted papers, say so and
  include those `paper_ids`. **No memory write happens on an LLM failure path.**
- User-memory backend down: chat still succeeds with `used_user_memory=False`.
- Pinecone down: lexical retrieval and live discovery continue.
- Ingestion failure: log sanitized, continue with stored evidence only.

Async safety: every SQLite, filesystem, embedding, and Pinecone call reachable
from `chat()` must be wrapped with `asyncio.to_thread` (or already be async).
No blocking work may run directly on the Discord event loop.

## 11. `research_radar.bot.mention` (W7)

```python
@dataclass(frozen=True, slots=True)
class MentionAdmission:
    accepted: bool
    reason: str
    text: str = ""
    is_dm: bool = False

class MentionPolicy:
    def __init__(self, settings: Settings) -> None: ...
    def admit(self, message: discord.Message, *, bot_user_id: int) -> MentionAdmission: ...
```

Rejection order (first match wins, all return `accepted=False`):
`self_message`, `bot_author`, `dm_disabled`, `mention_disabled`,
`no_mention`, `channel_not_allowed`, `owner_only`.

Accepted text has every form of the bot mention (`<@id>` and `<@!id>`) removed
and whitespace collapsed. An accepted-but-empty text yields
`accepted=True, text=""`; the caller replies with a short usage hint and makes
no backend call.

Intents (`research_radar.bot.client._application_intents`) become exactly:

```python
intents = discord.Intents.none()
intents.guilds = True
intents.guild_messages = True
intents.dm_messages = True   # only when settings.discord_dm_chat
```

The privileged **Message Content** intent is deliberately NOT requested.
Discord delivers content for messages that mention the app and for DMs to the
app, which is exactly the surface this feature uses.

The `on_message` handler contains no research logic: admit, hand text to
`ChatService`, chunk the reply to <= 1900 characters, send. Exceptions are
logged sanitized and answered with a short safe message.

## 12. Settings additions (W9 owns `config.py`)

```python
discord_owner_user_id: int | None = None
discord_allowed_channel_ids: tuple[int, ...] = ()   # parsed from a CSV env value
discord_chat_on_mention: bool = True
discord_dm_chat: bool = True

user_memory_backend: str = "disabled"      # {"disabled", "graphiti"}
user_memory_db_path: str = "data/user_memory"
user_memory_group_id: str = "primary-user"
user_memory_max_results: int = Field(default=8, ge=1, le=50)
user_memory_capture: bool = True

chat_live_discovery_limit: int = Field(default=10, ge=1, le=12)
```

`user_memory_backend` gets a validator rejecting anything outside
`{"disabled", "graphiti"}`. Empty optional env values must not crash startup
(`env_ignore_empty=True` is already set). `.env.example` and
`docs/ENVIRONMENT.md` are W9's alone.

## 13. Graphiti backend (W3)

Findings that bind the implementation:

- Package: `graphiti-core` (latest 0.29.3). Embedded local backend:
  `graphiti_core.driver.kuzu_driver.KuzuDriver(db=<path>)`.
- **Kuzu is deprecated upstream** and emits a `DeprecationWarning` directing
  users to Neo4j or FalkorDB. It remains the only embedded/zero-server option,
  so this phase keeps Kuzu, suppresses that specific warning at the adapter
  boundary, and documents the deviation. Neo4j/FalkorDB are explicitly out of
  scope: they require an operational server for a private single-user daemon.
- `Graphiti(graph_driver=driver, llm_client=..., embedder=...)`;
  `await graphiti.build_indices_and_constraints()` once at startup;
  `await graphiti.add_episode(...)`; `await graphiti.search(...)`.
- Reuse the existing LLM/embedding configuration; do not add duplicate
  provider keys unless the Graphiti API technically requires them.
- Dependency lives in a new optional extra `memory = ["graphiti-core>=0.29", "kuzu>=0.11"]`.
  `pip install -e .` must keep working without it. All imports of
  `graphiti_core` are lazy and inside the adapter; an `ImportError` degrades to
  a disabled store with a sanitized log line.
- CI must never require network. Tests use fakes for the Graphiti client.

## 14. Working agreement for every worker

- Base branch: `feat/personal-research-chat-memory` at `b41391a`.
- Stay strictly inside your owned files. If you need a change in a file another
  worker owns, escalate with `orca orchestration ask`; do not edit it.
- Run tests with the shared interpreter, no per-worktree venv:

  ```bash
  cd <your worktree>
  PYTHONPATH="$PWD/src" /Users/alvinluong/ResearchRadar/.venv/bin/python -m pytest -q
  /Users/alvinluong/ResearchRadar/.venv/bin/ruff check .
  ```

- Line length 100, ruff rules `E,F,I,B,UP`, `from __future__ import annotations`
  at the top of every module, docstrings on public classes and methods, and
  `asyncio_mode = "auto"` for pytest (no `@pytest.mark.asyncio` needed).
- Commit on your own branch with a coherent message. Never push to `main`,
  never merge, never open a PR.
- Do not add dependencies outside the `memory` extra without escalating.
- Never write a real credential into any file, test, log, or commit.
