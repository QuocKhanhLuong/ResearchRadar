"""Offline inspector for the personal user-memory backend (read-only).

Prints the sanitized MemoryStatus and, with ``--query``, matching facts as
whitelisted lines (fact text, memory class, temporal validity only). It never
prints credentials, raw driver objects, or graph internals. With
USER_MEMORY_BACKEND=disabled it reports that user memory is disabled and
exits 0.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import cast

from research_radar.bot.commands.memory import (
    format_fact_line,
    format_status_lines,
)
from research_radar.config import get_settings
from research_radar.memory import (
    DisabledUserMemoryStore,
    MemoryFact,
    MemoryStatus,
    UserMemoryStore,
)

_MIN_LIMIT = 1
_MAX_LIMIT = 50

_DISABLED_NOTICE = "user memory is disabled"
_UNAVAILABLE_NOTICE = "personal memory is unavailable"


class UnavailableUserMemoryStore:
    """Stand-in reporting a configured graphiti backend that cannot load."""

    def __init__(self) -> None:
        """Create an enabled-but-unhealthy store placeholder."""

    @property
    def backend_name(self) -> str:
        """Report the configured backend name."""

        return "graphiti"

    @property
    def enabled(self) -> bool:
        """Report the store as configured (enabled) even while unavailable."""

        return True

    async def search(self, query: str, *, limit: int = 8) -> list[MemoryFact]:
        """Return no facts because the backend is unavailable."""

        return []

    async def status(self) -> MemoryStatus:
        """Report the backend as unhealthy without any exception detail."""

        return MemoryStatus(backend=self.backend_name, enabled=True, healthy=False)

    async def close(self) -> None:
        """Hold no resources; closing is a no-op."""


def build_store_from_settings(settings: object) -> UserMemoryStore:
    """Build the configured store, degrading to unavailable when it cannot load."""

    backend = str(getattr(settings, "user_memory_backend", "disabled"))
    if backend != "graphiti":
        return DisabledUserMemoryStore()
    try:
        from research_radar.memory.graphiti_store import GraphitiUserMemoryStore
    except ImportError:
        # The optional `memory` extra is not installed in this environment.
        return UnavailableUserMemoryStore()
    try:
        return cast(UserMemoryStore, GraphitiUserMemoryStore(settings))
    except Exception:
        # Constructor mismatch or missing configuration must never crash the CLI.
        return UnavailableUserMemoryStore()


async def inspect_memory(store: UserMemoryStore, *, query: str | None, limit: int) -> int:
    """Print sanitized status plus optional matching facts; returns exit code 0."""

    status = await store.status()
    if not status.enabled:
        print(_DISABLED_NOTICE)
        await store.close()
        return 0
    print("\n".join(format_status_lines(status)))
    if not status.healthy:
        print(_UNAVAILABLE_NOTICE)
        await store.close()
        return 0
    if query is not None:
        facts = await store.search(query, limit=limit)
        print()
        if facts:
            print(f"facts matching {query!r}:")
            for fact in facts:
                print(format_fact_line(fact))
        else:
            print("no matching facts")
    await store.close()
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the personal-memory inspector's command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Inspect the ResearchRadar personal user-memory backend."
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Print stored facts matching this query",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"Max facts to print ({_MIN_LIMIT}..{_MAX_LIMIT}; default from settings)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point that maps failures to safe exit codes without tracebacks."""

    args = _parse_args(argv)
    settings = get_settings()
    default_limit = int(getattr(settings, "user_memory_max_results", 8))
    if args.limit is None:
        limit = max(_MIN_LIMIT, min(default_limit, _MAX_LIMIT))
    else:
        limit = max(_MIN_LIMIT, min(args.limit, _MAX_LIMIT))
    store = build_store_from_settings(settings)
    exit_code = asyncio.run(inspect_memory(store, query=args.query, limit=limit))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
