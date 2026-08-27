"""Graphiti/Kuzu-backed ``UserMemoryStore`` (phase W3).

All ``graphiti_core`` and ``kuzu`` imports are lazy and confined to this
module, so importing ``research_radar.memory`` never requires the optional
``memory`` extra. Every method degrades instead of raising: an ImportError, a
driver-construction failure, or any backend exception marks the store degraded
until process restart, logs one sanitized warning per failure class, and makes
subsequent calls behave like a total outage.

Recorded deviations and binding spike findings (docs/spikes/graphiti_compat.md):

- **Kuzu deprecation.** graphiti-core >=0.29 emits a ``DeprecationWarning``
  from ``KuzuDriver.__init__`` because the upstream Kuzu project is no longer
  maintained. Kuzu stays for now: it is the only embedded/zero-server backend,
  which is what this single-user daemon needs. The warning is suppressed with a
  narrowly scoped filter around driver construction only. Escape hatch: swap
  ``KuzuDriver`` for ``Neo4jDriver(uri, user, password)`` or a FalkorDB driver
  at that one construction site; no call sites change because everything else
  talks to the ``Graphiti``/``GraphDriver`` abstractions.
- **Custom entity types.** Spike section 4 confirms graphiti-core 0.29.x
  supports ``entity_types`` on ``add_episode``, so each ``MemoryClass`` value
  is carried as a custom entity type keyed ``memory_<value>``.
  ``Graphiti.search()`` has no equivalent query-time parameter, so the class is
  not recoverable at retrieval time and ``search()`` reports
  ``MemoryFact.memory_class=None``; temporal evolution stays recoverable via
  ``valid_at``/``invalid_at``/``expired_at`` instead of type collapse.
- **Embeddings.** The embedder reuses the repo's existing OpenAI-compatible
  endpoint configuration (``llm_base_url`` + ``llm_api_key``); no duplicate
  provider settings exist. The embedding model name and dimension have no repo
  setting, so they default to graphiti-core's own defaults and can be
  overridden per-construction for gateways whose embedding endpoint differs.
- **Environment hygiene.** graphiti_core calls ``load_dotenv()`` at import
  time, which mutates ``os.environ`` from whatever ``.env`` file is discovered
  on disk. The adapter snapshots the environment around those lazy imports and
  rolls every change back so application Settings stay authoritative.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, create_model

from research_radar.config import Settings
from research_radar.memory.models import (
    MemoryClass,
    MemoryFact,
    MemoryStatus,
    UserMemoryContext,
)

_LOGGER = logging.getLogger(__name__)

__all__ = ["GraphitiUserMemoryStore"]

_BACKEND_NAME = "graphiti"
_UNAVAILABLE_DETAIL = "user memory backend unavailable"

#: Factory returning a Graphiti-like client exposing the async surface used
#: here: ``build_indices_and_constraints()``, ``add_episode(...)``,
#: ``search(...)``, and ``close()``. Injectable so tests can run offline.
ClientFactory = Callable[[], Any]


def _as_utc(moment: datetime | None) -> datetime | None:
    """Normalize a datetime to aware UTC; naive values are assumed UTC."""

    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _reference_moment(reference_time: datetime | None) -> datetime:
    """Return a timezone-aware reference time, defaulting to now in UTC."""

    if reference_time is None:
        return datetime.now(tz=UTC)
    if reference_time.tzinfo is None:
        return reference_time.replace(tzinfo=UTC)
    return reference_time.astimezone(UTC)


def _superseded_at(edge: Any) -> datetime | None:
    """Return when an edge stopped holding, via invalid_at or expired_at."""

    invalid_at = _as_utc(getattr(edge, "invalid_at", None))
    expired_at = _as_utc(getattr(edge, "expired_at", None))
    if invalid_at is not None and expired_at is not None:
        return min(invalid_at, expired_at)
    if invalid_at is not None:
        return invalid_at
    return expired_at


def _edge_to_fact(edge: Any) -> MemoryFact:
    """Map one returned EntityEdge onto a MemoryFact, carrying what exists."""

    score = getattr(edge, "score", None)
    raw_fact = getattr(edge, "fact", None)
    parsed_score: float | None = None
    if score is not None:
        try:
            parsed_score = float(score)
        except (ValueError, TypeError):
            parsed_score = None
    return MemoryFact(
        fact="" if raw_fact is None else str(raw_fact),
        memory_class=None,
        valid_at=_as_utc(getattr(edge, "valid_at", None)),
        invalid_at=_as_utc(getattr(edge, "invalid_at", None)),
        score=parsed_score,
    )


def _restore_environ(snapshot: dict[str, str]) -> None:
    """Undo any os.environ changes made while the snapshot was taken.

    graphiti_core calls ``load_dotenv()`` at import time, which writes values
    from whatever ``.env`` file it finds on disk into the process environment.
    The host application's Settings must stay the single source of truth, so
    every added, changed, or removed key is rolled back after the imports.
    """

    for key in [key for key in os.environ if key not in snapshot]:
        del os.environ[key]
    for key, value in snapshot.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


@lru_cache(maxsize=1)
def _memory_entity_type_table() -> dict[str, type[BaseModel]]:
    """Build (once) custom entity types keyed to every MemoryClass value."""

    table: dict[str, type[BaseModel]] = {}
    for cls in MemoryClass:
        camel = "".join(part.title() for part in cls.value.split("_"))
        table[f"memory_{cls.value}"] = create_model(
            f"Memory{camel}",
            __doc__=f"User-memory entity carrying the '{cls.value}' memory class.",
        )
    return table


class GraphitiUserMemoryStore:
    """Advisory user-memory store over an embedded Graphiti/Kuzu temporal graph.

    Initialization (driver construction plus index build) happens lazily on
    first use, exactly once, guarded by an :class:`asyncio.Lock` so concurrent
    chat turns cannot double-initialize. Blocking work (filesystem prep, client
    construction) runs in a worker thread via :func:`asyncio.to_thread`; the
    async Graphiti API is awaited directly. The store never raises to callers
    and its warnings and statuses never echo episode content or credentials.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        client_factory: ClientFactory | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        """Configure the store from settings without touching any backend.

        ``client_factory`` overrides how the Graphiti client is built (used by
        tests to inject a fake); it must be a zero-argument synchronous callable
        because it runs inside :func:`asyncio.to_thread`.
        """

        self._settings = settings
        self._db_path = settings.user_memory_db_path_resolved()
        self._group_id = settings.user_memory_group_id
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        self._client_factory: ClientFactory = client_factory or self._build_default_client
        self._client: Any | None = None
        self._init_lock = asyncio.Lock()
        self._degraded = False
        self._closed = False
        self._warned_classes: set[str] = set()

    @property
    def backend_name(self) -> str:
        """Report the stable graphiti backend identifier."""

        return _BACKEND_NAME

    @property
    def enabled(self) -> bool:
        """Report the store as enabled even while degraded or closed."""

        return True

    async def add_episode(
        self,
        content: str,
        *,
        source_description: str = "discord-chat",
        reference_time: datetime | None = None,
        memory_class: MemoryClass | None = None,
    ) -> bool:
        """Persist one durable user episode; report whether it was stored."""

        if not content.strip():
            return False
        if not await self._ensure_initialized():
            return False
        client = self._client
        if client is None:
            return False
        try:
            await client.add_episode(
                name=f"user-episode-{uuid.uuid4().hex[:12]}",
                episode_body=content,
                source_description=source_description,
                reference_time=_reference_moment(reference_time),
                group_id=self._group_id,
                entity_types=(_memory_entity_type_table() if memory_class is not None else None),
            )
        except Exception:
            self._mark_degraded(
                "episode-write",
                "graphiti user memory: episode write failed; episode not stored",
            )
            return False
        return True

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        include_historical: bool = False,
    ) -> list[MemoryFact]:
        """Return bounded advisory facts relevant to the query.

        Facts invalidated or expired before now are excluded by default so
        answers reflect current context; pass ``include_historical=True`` to
        reach superseded facts so temporal evolution stays observable. Fewer
        than ``limit`` facts can come back when trailing edges were filtered.
        """

        if limit <= 0 or not query.strip():
            return []
        if not await self._ensure_initialized():
            return []
        client = self._client
        if client is None:
            return []
        try:
            edges = await client.search(query, group_ids=[self._group_id], num_results=limit)
        except Exception:
            self._mark_degraded(
                "search",
                "graphiti user memory: search failed; returning no facts",
            )
            return []
        now = datetime.now(tz=UTC)
        facts: list[MemoryFact] = []
        for edge in edges:
            ended = _superseded_at(edge)
            if not include_historical and ended is not None and _as_utc(ended) <= now:
                continue
            facts.append(_edge_to_fact(edge))
        return facts[:limit]

    async def get_context(self, query: str, *, limit: int = 8) -> UserMemoryContext:
        """Return the bounded advisory context handed to prompt assembly.

        The context is flagged degraded whenever the backend has failed in this
        process, even if some facts were recovered afterwards.
        """

        facts = tuple(await self.search(query, limit=limit))
        return UserMemoryContext(
            backend=_BACKEND_NAME,
            degraded=self._degraded,
            facts=facts,
        )

    async def status(self) -> MemoryStatus:
        """Report sanitized health with fixed detail strings only."""

        await self._ensure_initialized()
        healthy = self._client is not None and not self._degraded
        return MemoryStatus(
            backend=_BACKEND_NAME,
            enabled=True,
            healthy=healthy,
            detail="" if healthy else _UNAVAILABLE_DETAIL,
            persistence_path=str(self._db_path),
            episode_count=None,
        )

    async def close(self) -> None:
        """Release backend resources; idempotent and safe before any use."""

        client = self._client
        self._closed = True
        self._client = None
        if client is None:
            return
        try:
            await client.close()
        except Exception:
            if "close" not in self._warned_classes:
                self._warned_classes.add("close")
                _LOGGER.warning("graphiti user memory: backend raised while closing")

    async def _ensure_initialized(self) -> bool:
        """Initialize lazily exactly once; return True when usable."""

        if self._closed or self._degraded:
            return False
        if self._client is not None:
            return True
        async with self._init_lock:
            if self._closed or self._degraded:
                return False
            if self._client is not None:
                return True
            if self._missing_llm_configuration():
                # Graphiti needs an LLM for entity extraction and an embedder
                # for search, so it cannot run against LLM_PROVIDER=mock. Say
                # so explicitly: the generic "initialization failed" message
                # sends operators looking at Kuzu instead of at their config.
                self._mark_degraded(
                    "configuration",
                    "USER_MEMORY_BACKEND=graphiti also requires LLM_BASE_URL, "
                    "LLM_MODEL and LLM_API_KEY; personal memory stays disabled.",
                )
                return False
            try:
                await asyncio.to_thread(self._ensure_storage_directory)
                client = await asyncio.to_thread(self._client_factory)
                try:
                    await client.build_indices_and_constraints()
                except Exception:
                    try:
                        await client.close()
                    except Exception:
                        pass
                    raise
            except ImportError:
                self._mark_degraded(
                    "dependency",
                    "graphiti user memory unavailable: install the optional 'memory' extra",
                )
            except Exception:
                self._mark_degraded(
                    "initialization",
                    "graphiti user memory initialization failed; degraded until restart",
                )
            else:
                if self._closed:
                    try:
                        await client.close()
                    except Exception:
                        pass
                    return False
                self._client = client
                return True
            return False

    def _missing_llm_configuration(self) -> bool:
        """Report whether the LLM settings Graphiti needs are absent.

        Checked before any filesystem work so a misconfigured install does not
        leave an empty Kuzu database behind. Never reads the secret value.
        """

        settings = self._settings
        api_key = (
            settings.llm_api_key.get_secret_value() if settings.llm_api_key is not None else ""
        )
        base_url = settings.llm_base_url or ""
        model = settings.llm_model or ""
        return not api_key.strip() or not base_url.strip() or not model.strip()

    def _ensure_storage_directory(self) -> None:
        """Create the database parent directory on first use, never at import."""

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _mark_degraded(self, failure_class: str, message: str) -> None:
        """Flag the store degraded and warn once per failure class."""

        self._degraded = True
        if failure_class not in self._warned_classes:
            self._warned_classes.add(failure_class)
            _LOGGER.warning("%s", message)

    def _build_default_client(self) -> Any:
        """Construct the real Graphiti client; all lazy imports live here."""

        environ_snapshot = dict(os.environ)
        try:
            from graphiti_core import Graphiti
            from graphiti_core.cross_encoder.client import CrossEncoderClient
            from graphiti_core.driver.kuzu_driver import KuzuDriver
            from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
            from graphiti_core.llm_client.config import LLMConfig
            from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        finally:
            _restore_environ(environ_snapshot)

        api_key = (
            self._settings.llm_api_key.get_secret_value()
            if self._settings.llm_api_key is not None
            else None
        )
        base_url = self._settings.llm_base_url
        model = self._settings.llm_model
        llm_config = LLMConfig(api_key=api_key, model=model, base_url=base_url, small_model=model)

        class PassthroughCrossEncoder(CrossEncoderClient):
            """Identity ranker; basic hybrid search never invokes the reranker."""

            async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
                """Return passages unchanged so no reranker endpoint is needed."""

                return [(passage, 0.0) for passage in passages]

        embedder_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
        if self._embedding_model is not None:
            embedder_kwargs["embedding_model"] = self._embedding_model
        if self._embedding_dim is not None:
            embedder_kwargs["embedding_dim"] = self._embedding_dim

        with warnings.catch_warnings():
            # graphiti-core >=0.29 raises DeprecationWarning from
            # KuzuDriver.__init__ (upstream Kuzu unmaintained). We keep Kuzu as
            # the only embedded zero-server option for this single-user daemon;
            # migrating later means swapping this one constructor call for
            # Neo4jDriver/FalkorDB (see module docstring and spike section 8).
            warnings.simplefilter("ignore", DeprecationWarning)
            driver = KuzuDriver(db=str(self._db_path))

        return Graphiti(
            graph_driver=driver,
            llm_client=OpenAIGenericClient(
                config=llm_config,
                structured_output_mode="json_object",
            ),
            embedder=OpenAIEmbedder(config=OpenAIEmbedderConfig(**embedder_kwargs)),
            cross_encoder=PassthroughCrossEncoder(),
        )
