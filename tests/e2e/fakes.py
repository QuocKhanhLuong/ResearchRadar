"""Reusable deterministic fakes and prompt-inspection helpers for the chat e2e harness.

Everything in this module runs fully offline: no network, no credentials, and no
external services. The user-memory test double is NOT defined here; the shipped
``research_radar.memory.fakes.FakeUserMemoryStore`` is reused through the small
adapters at the bottom of this file so the suite never maintains a second fake.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Annotated, Any, Literal, get_args, get_origin

import discord
from pydantic import BaseModel

from research_radar.models.paper import Paper
from research_radar.reader.llm.base import LLMMessage
from research_radar.semantic.base import (
    EntityType,
    SemanticHit,
    SemanticIndexStatus,
    SemanticRecord,
)

FAKE_LLM_ANSWER = "[fake-llm] Synthesis grounded strictly in the supplied evidence."

BOT_USER_ID = 111222333444555666
HUMAN_USER_ID = 777888999000111222
OTHER_BOT_USER_ID = 333444555666777888
GUILD_ID = 555000111
CHANNEL_ID = 987654321

DEFAULT_CHAT_TOPIC = "quantum error correction"

# Deliberately fake secret material for capture-policy scenarios. It matches the
# documented detection shapes (api_key= assignment plus an sk- prefixed token)
# while being unmistakably not a real credential.
SECRET_PAYLOAD = "api_key=sk-fake000000000000000000000000deadbeef"

PROMPT_SYSTEM_HEADER = "SYSTEM RULES"
PROMPT_USER_MEMORY_HEADER = "USER MEMORY (ADVISORY — NOT SCIENTIFIC EVIDENCE)"
PROMPT_PROJECT_HEADER = "EXPLICIT PROJECT MEMORY (CANONICAL USER/PROJECT STATE)"
PROMPT_STORED_EVIDENCE_HEADER = "STORED SCIENTIFIC EVIDENCE (CANONICAL)"
PROMPT_DISCOVERY_HEADER = "LIVE DISCOVERY EVIDENCE (METADATA/ABSTRACT-LEVEL ONLY)"
PROMPT_QUESTION_HEADER = "QUESTION"
PROMPT_SECTION_HEADERS = (
    PROMPT_SYSTEM_HEADER,
    PROMPT_USER_MEMORY_HEADER,
    PROMPT_PROJECT_HEADER,
    PROMPT_STORED_EVIDENCE_HEADER,
    PROMPT_DISCOVERY_HEADER,
    PROMPT_QUESTION_HEADER,
)
SCIENTIFIC_EVIDENCE_HEADERS = (PROMPT_STORED_EVIDENCE_HEADER, PROMPT_DISCOVERY_HEADER)


class FakeScoutProvider:
    """Deterministic stand-in for one scholarly provider with a call counter."""

    def __init__(self, name: str, papers: Sequence[Paper]) -> None:
        self.name = name
        self.papers = list(papers)
        self.calls = 0
        self.queries: list[tuple[str, int]] = []

    async def search(self, query: str, limit: int = 10) -> list[Paper]:
        """Return the canned records and record that this provider was called."""

        self.calls += 1
        self.queries.append((query, limit))
        return list(self.papers)


def _slugify(topic: str) -> str:
    """Return a stable lowercase identifier fragment for a topic phrase."""

    slug = re.sub(r"[^a-z0-9]+", "-", topic.strip().casefold()).strip("-")
    return slug or "topic"


def _stable_identity_numbers(topic: str, work_number: int) -> tuple[int, int]:
    """Return two deterministic integers that are unique per (topic, work).

    Provider identities must never collide across different synthetic topics,
    otherwise canonical ingestion would merge unrelated works exactly because
    they share an OpenAlex or arXiv identifier.
    """

    digest = hashlib.sha256(
        f"{_slugify(topic)}|{work_number}".encode()
    ).digest()
    value = int.from_bytes(digest[:8], "big")
    return value % 1_000_000_000, (value // 1_000_000_000) % 100_000_000


def synthetic_work_papers(topic: str, work_number: int) -> dict[str, Paper]:
    """Build one scholarly work in OpenAlex, Semantic Scholar, and arXiv shapes.

    All three records describe the same work through a shared DOI so canonical
    deduplication must collapse them into a single SQLite row. Identifiers are
    derived deterministically from the topic so different topics never share a
    provider identity.
    """

    stem = topic.strip().title()
    title = f"{stem} Research Direction {work_number}"
    doi = f"10.7777/{_slugify(topic)}.{work_number}"
    year = 2024 + work_number % 2
    venues = ("Journal of Offline Studies", "Transactions on Synthetic Science", "arXiv")
    venue = venues[work_number % len(venues)]
    shared = {
        "title": title,
        "publication_year": year,
        "venue": venue,
        "doi": doi,
        "citation_count": 3 * work_number,
    }
    abstract = (
        f"Deterministic synthetic abstract about {topic} for offline harness testing "
        f"(study {work_number})."
    )
    openalex_number, arxiv_number = _stable_identity_numbers(topic, work_number)
    return {
        "openalex": Paper(
            id=f"openalex:W{openalex_number}",
            abstract=abstract,
            url=f"https://example.test/openalex/W{openalex_number}",
            external_ids={"openalex": f"W{openalex_number}", "doi": doi},
            source="openalex",
            **shared,
        ),
        "semantic_scholar": Paper(
            id=f"semantic_scholar:s2-{_slugify(topic)}-{work_number}",
            external_ids={
                "semantic_scholar": f"s2-{_slugify(topic)}-{work_number}",
                "doi": doi,
            },
            source="semantic_scholar",
            **shared,
        ),
        "arxiv": Paper(
            id=f"arxiv:2603.{arxiv_number:08d}",
            external_ids={"arxiv": f"2603.{arxiv_number:08d}", "doi": doi},
            source="arxiv",
            **shared,
        ),
    }


def provider_trio_for_topic(
    topic: str = DEFAULT_CHAT_TOPIC, works: int = 3
) -> list[FakeScoutProvider]:
    """Create an OpenAlex/Semantic Scholar/arXiv trio serving one topic."""

    per_provider: dict[str, list[Paper]] = {"openalex": [], "semantic_scholar": [], "arxiv": []}
    for number in range(1, works + 1):
        for provider_name, paper in synthetic_work_papers(topic, number).items():
            per_provider[provider_name].append(paper)
    return [FakeScoutProvider(name, papers) for name, papers in per_provider.items()]


def total_scout_calls(providers: Sequence[FakeScoutProvider]) -> int:
    """Sum every recorded provider search across the trio."""

    return sum(provider.calls for provider in providers)


@dataclass(frozen=True, slots=True)
class RecordedLLMCall:
    """One captured structured-generation call."""

    messages: tuple[LLMMessage, ...]
    response_model: str


def joined_prompt_text(messages: Sequence[LLMMessage]) -> str:
    """Join every message content so section headers can be located robustly."""

    return "\n".join(message.content for message in messages)


def prompt_section_span(prompt_text: str, header: str) -> tuple[int, int]:
    """Return the (start, end) character span of one prompt section's body.

    The end is the start of the next present contractual header (or the end of
    the text). Raises AssertionError when the requested header is absent.
    """

    match = re.search(rf"(?m)^{re.escape(header)}\s*$", prompt_text)
    assert match is not None, f"Prompt is missing the required section header: {header!r}"
    start = match.end()
    following: list[int] = []
    for other in PROMPT_SECTION_HEADERS:
        if other == header:
            continue
        for found in re.finditer(rf"(?m)^{re.escape(other)}\s*$", prompt_text):
            if found.start() > match.start():
                following.append(found.start())
    end = min(following) if following else len(prompt_text)
    return start, end


def prompt_section(prompt_text: str, header: str) -> str:
    """Return the full body text of one contractual prompt section."""

    start, end = prompt_section_span(prompt_text, header)
    return prompt_text[start:end]


@dataclass(slots=True)
class RecordingLLMProvider:
    """Recording fake LLM returning a valid structured response every call.

    Required fields of whatever ``response_model`` the caller passes are filled
    deterministically: string fields receive the configured answer text, other
    scalars receive neutral values, and nested models are synthesized
    recursively. The answer never invents citation identifiers.
    """

    answer: str = FAKE_LLM_ANSWER
    calls: list[RecordedLLMCall] = field(default_factory=list)

    async def generate_structured(self, messages: list[LLMMessage], response_model: type[Any]):
        """Record the call and return a deterministic structured instance."""

        self.calls.append(
            RecordedLLMCall(
                messages=tuple(message.model_copy(deep=True) for message in messages),
                response_model=getattr(response_model, "__name__", str(response_model)),
            )
        )
        return _synthesized_response(response_model, self.answer)

    @property
    def call_count(self) -> int:
        """Return how many generation calls were made."""

        return len(self.calls)

    @property
    def last_messages(self) -> tuple[LLMMessage, ...]:
        """Return the messages of the most recent call."""

        assert self.calls, "No LLM call was recorded."
        return self.calls[-1].messages

    @property
    def last_prompt_text(self) -> str:
        """Return the joined prompt text of the most recent call."""

        return joined_prompt_text(self.last_messages)


class FailingLLMProvider:
    """Fake LLM that simulates a total outage by raising on every call."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or RuntimeError("Simulated total LLM outage.")
        self.attempts = 0

    async def generate_structured(self, messages: list[LLMMessage], response_model: type[Any]):
        """Count the attempt and raise the simulated failure."""

        self.attempts += 1
        raise self.error


def _unwrap_annotation(annotation: Any) -> Any:
    """Strip Annotated wrappers and Optional unions down to a concrete type."""

    current = annotation
    for _ in range(8):
        origin = get_origin(current)
        if origin is None:
            return current
        if origin is Literal:
            return current
        if origin is Annotated or getattr(origin, "__metadata__", None) is not None:
            current = get_args(current)[0]
            continue
        args = [arg for arg in get_args(current) if arg is not type(None)]
        if len(args) == 1:
            current = args[0]
            continue
        return current
    return current


def _value_for_field(name: str, annotation: Any, answer: str) -> Any:
    """Deterministically fill one required field based on its declared type."""

    resolved = _unwrap_annotation(annotation)
    if get_origin(resolved) is Literal:
        return get_args(resolved)[0]
    if resolved is str:
        return answer
    if resolved is bool:
        return False
    if resolved is int:
        return 0
    if resolved is float:
        return 0.0
    origin = get_origin(resolved)
    if origin in (list, set):
        return []
    if origin in (tuple, frozenset):
        return ()
    if origin is dict:
        return {}
    if isinstance(resolved, type) and issubclass(resolved, BaseModel):
        return _synthesized_response(resolved, answer)
    if isinstance(resolved, type) and hasattr(resolved, "__members__"):
        members = resolved.__members__
        return next(iter(members.values()))
    raise TypeError(
        f"Fake LLM cannot synthesize a value for required field {name!r} ({annotation!r})."
    )


def _synthesized_response(response_model: type[Any], answer: str) -> Any:
    """Construct an instance of ``response_model`` with every required field set."""

    fields = getattr(response_model, "model_fields", None)
    if fields is None:
        raise TypeError(f"Fake LLM requires a pydantic model, got {response_model!r}.")
    kwargs: dict[str, Any] = {}
    for field_name, field_info in fields.items():
        if field_info.is_required():
            kwargs[field_name] = _value_for_field(field_name, field_info.annotation, answer)
    return response_model(**kwargs)


class FakeSemanticIndex:
    """Switchable in-memory stand-in for the derived Pinecone index."""

    backend = "fake-pinecone"

    def __init__(self) -> None:
        self._available = True
        self.upserted_records: dict[str, SemanticRecord] = {}
        self.upsert_calls = 0
        self.search_queries = 0

    def set_available(self, available: bool) -> None:
        """Flip the simulated outage state of the index."""

        self._available = available

    @property
    def available(self) -> bool:
        """Return whether semantic retrieval can currently be attempted."""

        return self._available

    def upsert(self, records: Sequence[SemanticRecord]) -> int:
        """Idempotently store records by entity id and report acceptance."""

        self.upsert_calls += 1
        for record in records:
            self.upserted_records[record.entity_id] = record
        return len(records)

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 10,
        entity_type: EntityType | None = None,
    ) -> list[SemanticHit]:
        """Return deterministic candidates, or nothing while unavailable."""

        self.search_queries += 1
        if not self._available:
            return []
        hits: list[SemanticHit] = []
        for position, record in enumerate(self.upserted_records.values()):
            if entity_type is not None and record.entity_type != entity_type:
                continue
            hits.append(
                SemanticHit(
                    entity_id=record.entity_id,
                    entity_type=record.entity_type,
                    paper_id=record.paper_id,
                    score=1.0 - position * 0.01,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def delete(self, entity_ids: Sequence[str]) -> int:
        """Remove stored entities and report how many were deleted."""

        removed = 0
        for entity_id in entity_ids:
            if self.upserted_records.pop(entity_id, None) is not None:
                removed += 1
        return removed

    def status(self) -> SemanticIndexStatus:
        """Return a compact availability summary."""

        return SemanticIndexStatus(backend=self.backend, available=self._available)


class FakeEmbeddingProvider:
    """Deterministic hash-based embedding provider for offline runs."""

    model_id = "fake-embedder-v1"

    @property
    def dimension(self) -> int:
        """Return the fixed vector dimension produced by this provider."""

        return 4

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed each text into a stable pseudo-random unit-ish vector."""

        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([byte / 255.0 for byte in digest[: self.dimension]])
        return vectors


def make_discord_message(
    content: str,
    *,
    author_id: int = HUMAN_USER_ID,
    author_is_bot: bool = False,
    bot_user_id: int = BOT_USER_ID,
    dm: bool = False,
) -> discord.Message:
    """Build a real ``discord.Message`` offline from gateway-shaped payload data."""

    author: dict[str, Any] = {
        "id": str(author_id),
        "username": f"user{author_id}",
        "discriminator": "0",
        "avatar": None,
    }
    if author_is_bot or author_id == bot_user_id:
        author["bot"] = True
    mentions: list[dict[str, Any]] = []
    if f"<@{bot_user_id}" in content or f"<@!{bot_user_id}" in content:
        bot_data = {
            "id": str(bot_user_id),
            "username": "research-radar",
            "discriminator": "0",
            "avatar": None,
            "bot": True,
        }
        mentions.append(bot_data)
        if author_id == bot_user_id:
            author.update(bot_data)
    payload: dict[str, Any] = {
        "id": "123456789012345678",
        "author": author,
        "content": content,
        "timestamp": "2026-08-26T00:00:00+00:00",
        "edited_timestamp": None,
        "tts": False,
        "mention_everyone": False,
        "mentions": mentions,
        "mention_roles": [],
        "attachments": [],
        "embeds": [],
        "pinned": False,
        "type": 0,
    }
    state = _FakeGatewayState()
    channel: Any
    if dm:
        me = discord.ClientUser(
            state=state,
            data={
                "id": str(bot_user_id),
                "username": "research-radar",
                "discriminator": "0",
                "avatar": None,
            },
        )
        channel = discord.DMChannel(
            state=_FakeGatewayState(),
            me=me,
            data={
                "id": str(CHANNEL_ID),
                "recipients": [
                    {
                        "id": str(author_id),
                        "username": f"user{author_id}",
                        "discriminator": "0",
                        "avatar": None,
                    }
                ],
            },
        )
        state = channel._state
    else:
        guild = SimpleNamespaceWithTypes(id=GUILD_ID, name="offline-guild")
        channel = SimpleNamespaceWithTypes(
            id=CHANNEL_ID, guild=guild, type=discord.ChannelType.text
        )
    return discord.Message(state=state, channel=channel, data=payload)


class SimpleNamespaceWithTypes(SimpleNamespace):
    """Namespace stand-in for guild/channel objects used only for attributes."""


class _FakeGatewayState:
    """Minimal connection-state surface required to construct messages."""

    def __init__(self) -> None:
        self.self_id = BOT_USER_ID
        self.user = None

    def store_user(self, data: dict[str, Any], cache: bool = True) -> discord.user.BaseUser:
        """Build a real user object exactly as the gateway state would."""

        return discord.user.BaseUser(state=self, data=data)

    def get_user(self, _user_id: int) -> None:
        """No cached users exist in the offline fake state."""

        return None


# ---------------------------------------------------------------------------
# Adapters around the SHIPPED research_radar.memory.fakes.FakeUserMemoryStore.
# W2 ships that fake concurrently with this harness; the adapters below pin the
# whole integration surface (seeding + episode assertions) into tiny functions
# so any naming drift is fixed in exactly one place.


def build_fake_user_memory(*facts: Any, fail: bool = False) -> Any:
    """Create the shipped FakeUserMemoryStore, seeding facts when possible."""

    from research_radar.memory.fakes import FakeUserMemoryStore

    fact_objects = tuple(_as_memory_fact(fact) for fact in facts)
    for kwargs in ({"facts": fact_objects}, {"seed_facts": fact_objects}):
        try:
            return FakeUserMemoryStore(**kwargs, fail=fail)
        except TypeError:
            continue
    try:
        return FakeUserMemoryStore(fact_objects, fail=fail)
    except TypeError:
        pass
    store = FakeUserMemoryStore(fail=fail)
    if fact_objects:
        _seed_after_construction(store, fact_objects)
    return store


def seed_user_memory(store: Any, *facts: Any) -> None:
    """Seed additional facts onto an already-constructed fake memory store."""

    fact_objects = tuple(_as_memory_fact(fact) for fact in facts)
    if not fact_objects:
        return
    if _try_seed_method(store, fact_objects):
        return
    if _try_seed_attribute(store, fact_objects):
        return
    raise RuntimeError(
        "Could not seed FakeUserMemoryStore: no known seeding API matched. "
        "Update tests/e2e/fakes.py adapters."
    )


def captured_episodes(store: Any) -> list[str]:
    """Return every episode captured by the fake store as plain text."""

    for attribute in ("episodes", "captured_episodes", "stored_episodes", "recorded_episodes"):
        if hasattr(store, attribute):
            entries = getattr(store, attribute)
            if not callable(entries):
                return [_episode_text(entry) for entry in entries]
    for method_name in ("get_episodes", "list_episodes", "captured_episode_texts"):
        method = getattr(store, method_name, None)
        if callable(method):
            return [_episode_text(entry) for entry in method()]
    raise RuntimeError(
        "Could not read episodes from FakeUserMemoryStore: no known episodes API "
        "matched. Update tests/e2e/fakes.py adapters."
    )


def _as_memory_fact(fact: Any) -> Any:
    """Normalize strings into MemoryFact instances, passing objects through."""

    if isinstance(fact, str):
        from research_radar.memory.models import MemoryClass, MemoryFact

        return MemoryFact(fact=fact, memory_class=MemoryClass.PREFERENCE)
    return fact


def _seed_after_construction(store: Any, fact_objects: tuple[Any, ...]) -> None:
    """Best-effort seeding on stores without seeding constructor arguments."""

    if _try_seed_method(store, fact_objects) or _try_seed_attribute(store, fact_objects):
        return
    raise RuntimeError(
        "Could not seed FakeUserMemoryStore after construction: update adapters."
    )


def _try_seed_method(store: Any, fact_objects: tuple[Any, ...]) -> bool:
    """Seed through whichever seeding method name the shipped fake exposes."""

    for method_name in ("seed", "add_fact", "add_facts", "remember", "set_facts", "remember_fact"):
        method = getattr(store, method_name, None)
        if not callable(method):
            continue
        try:
            method(*fact_objects)
            return True
        except TypeError:
            for fact_object in fact_objects:
                method(fact_object)
            return True
    return False


def _try_seed_attribute(store: Any, fact_objects: tuple[Any, ...]) -> bool:
    """Seed by extending whichever facts list attribute the fake exposes."""

    for attribute in ("facts", "_facts", "seeded_facts", "_seeded_facts", "known_facts"):
        existing = getattr(store, attribute, None)
        if isinstance(existing, list):
            existing.extend(fact_objects)
            return True
    return False


def _episode_text(entry: Any) -> str:
    """Render one stored episode entry as scannable text."""

    if isinstance(entry, str):
        return entry
    content = getattr(entry, "content", None)
    if isinstance(content, str):
        return content
    return repr(entry)
