#!/usr/bin/env python
"""Graphiti/Kuzu compatibility spike (W1).

Standalone investigation script. NOT imported by the package and NOT production
code. It prints the installed versions and resolved signatures discovered in the
shared venv, then proves (offline) that a file-backed KuzuDriver persists data
across a process restart by re-opening the database in a child process.

No network access and no API key are required for the default introspection +
persistence path. Anything that would call an LLM/embedding endpoint is gated
behind ``--live`` and skipped by default.

Usage:
    python scripts/spike_graphiti.py [--live] [--keep]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import gc
import importlib.metadata as md
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

REOPEN_MARKER = "--spike-reopen-child"


def _version(dist_name: str) -> str:
    """Return the installed version of a distribution or '<not installed>'."""
    try:
        return md.version(dist_name)
    except md.PackageNotFoundError:
        return "<not installed>"


def _sig(obj: object) -> str:
    """Return a best-effort signature string for any callable."""
    try:
        return f"{inspect.signature(obj)}"
    except (TypeError, ValueError):
        return "<signature unavailable>"


def _print_versions() -> None:
    """Print installed versions of graphiti-core and its relevant dependencies."""
    print("== VERSIONS ==")
    for dist in (
        "graphiti-core",
        "kuzu",
        "pydantic",
        "pydantic_core",
        "pydantic-settings",
        "openai",
        "tenacity",
        "posthog",
        "numpy",
        "neo4j",
    ):
        print(f"  {dist}=={_version(dist)}")


def _print_signatures() -> None:
    """Print exact signatures resolved out of the installed graphiti-core."""
    import graphiti_core
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.edges import EntityEdge
    from graphiti_core.graphiti import AddEpisodeResults
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.search.search_filters import SearchFilters

    print("\n== SIGNATURES (resolved from installed package) ==")
    print(f"  graphiti_core.__file__ = {graphiti_core.__file__}")
    print(f"  Graphiti.__init__{_sig(graphiti_core.Graphiti.__init__)}")
    print(f"  KuzuDriver.__init__{_sig(KuzuDriver.__init__)}")
    print(f"  KuzuDriver.close{inspect.signature(KuzuDriver.close)} "
          f"(async={inspect.iscoroutinefunction(KuzuDriver.close)})")
    build = graphiti_core.Graphiti.build_indices_and_constraints
    print(f"  Graphiti.build_indices_and_constraints{_sig(build)} "
          f"(async={inspect.iscoroutinefunction(build)})")
    add = graphiti_core.Graphiti.add_episode
    print(f"  Graphiti.add_episode{_sig(add)}")
    search = graphiti_core.Graphiti.search
    print(f"  Graphiti.search{_sig(search)}")
    print(f"  Graphiti.close{inspect.signature(graphiti_core.Graphiti.close)}")
    print(f"  OpenAIGenericClient.__init__{_sig(OpenAIGenericClient.__init__)}")
    print(f"  LLMConfig.__init__{_sig(LLMConfig.__init__)}")
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

    print(f"  OpenAIEmbedder.__init__{_sig(OpenAIEmbedder.__init__)}")
    print(f"  OpenAIEmbedderConfig fields: "
          f"{list(OpenAIEmbedderConfig.model_fields)}")
    print(f"  EpisodeType members: {[e.value for e in EpisodeType]}")
    print(f"  EntityEdge fields: {list(EntityEdge.model_fields)}")
    print(f"  SearchFilters fields: {list(SearchFilters.model_fields)}")
    print(f"  AddEpisodeResults fields: {list(AddEpisodeResults.model_fields)}")


def _child_read_back(db_path: str) -> int:
    """Re-open an existing Kuzu database and return the episode count found."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from graphiti_core.driver.kuzu_driver import KuzuDriver

    async def count() -> tuple[int, int]:
        driver = KuzuDriver(db=db_path)
        try:
            rows, _, _ = await driver.execute_query(
                "MATCH (e:Episodic) RETURN count(e) AS n;"
            )
            episodes = int(rows[0]["n"]) if rows else -1
            schema_rows, _, _ = await driver.execute_query(
                "CALL show_tables() RETURN *;"
            )
            return episodes, len(schema_rows)
        finally:
            del driver
            gc.collect()

    episodes, tables = asyncio.run(count())
    print(f"[child pid={os.getpid()}] reopened db at {db_path}: "
          f"tables={tables}, episodic_nodes={episodes}")
    return episodes


def _prove_persistence() -> None:
    """Create a file-backed KuzuDriver, write a row, and read it back in a new process."""
    from graphiti_core.driver.kuzu_driver import KuzuDriver

    tmp = Path(tempfile.mkdtemp(prefix="spike_graphiti_"))
    db_path = str(tmp / "user_memory")
    print(f"\n== PERSISTENCE PROBE ==\n  db path: {db_path}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        driver = KuzuDriver(db=db_path)
    for w in caught:
        print(f"  [warn] {w.category.__name__}: {w.message}")

    async def write(active_driver: KuzuDriver) -> None:
        await active_driver.build_indices_and_constraints()
        await active_driver.execute_query(
            "CREATE (:Episodic {uuid: $uuid, name: 'spike', group_id: 'primary-user', "
            "source: 'message', content: 'probe row', created_at: $created_at});",
            uuid="spike-uuid-0001",
            created_at=datetime.datetime(2026, 8, 26, 12, 0, 0),
        )
        # Force a WAL checkpoint so the child process observes durable state.
        try:
            await active_driver.execute_query("CHECKPOINT;")
        except Exception as exc:  # noqa: BLE001 - spike: report, do not fail
            print(f"  [note] explicit CHECKPOINT unsupported ({exc!r}); relying on GC close")

    asyncio.run(write(driver))
    del driver
    gc.collect()

    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__), REOPEN_MARKER, db_path],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        print(f"  CHILD FAILED rc={result.returncode}\n{result.stderr}")
        persisted = -1
    else:
        persisted = int(result.stdout.strip().rsplit("episodic_nodes=", 1)[-1])

    layout = sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*") if p.is_file())[:20]
    total_bytes = sum(p.stat().st_size for p in tmp.rglob("*") if p.is_file())
    kind = "single file" if (tmp / "user_memory").is_file() else "directory tree"
    print(f"  on-disk layout: {kind} ({total_bytes} bytes total)")
    for entry in layout:
        print(f"    {entry}")
    verdict = "SURVIVES restart" if persisted >= 1 else "DID NOT persist"
    print(f"  => persistence across process restart: {verdict}")
    if "--keep" not in sys.argv:
        shutil.rmtree(tmp, ignore_errors=True)


async def _live_probe() -> None:
    """Exercise add_episode/search against a real LLM endpoint (requires --live)."""
    from graphiti_core import Graphiti
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    base_url = os.environ.get("SPIKE_LLM_BASE_URL", "")
    api_key = os.environ.get("SPIKE_LLM_API_KEY", "")
    model = os.environ.get("SPIKE_LLM_MODEL", "")
    embedding_model = os.environ.get("SPIKE_LLM_EMBEDDING_MODEL", model)
    if not (base_url and api_key and model):
        print("live probe requested but SPIKE_LLM_BASE_URL/SPIKE_LLM_API_KEY/"
              "SPIKE_LLM_MODEL unset; skipping live section")
        return
    llm_config = LLMConfig(api_key=api_key, model=model, base_url=base_url, small_model=model)
    llm_client = OpenAIGenericClient(config=llm_config, structured_output_mode="json_object")
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(api_key=api_key, base_url=base_url,
                                    embedding_model=embedding_model)
    )
    tmp = Path(tempfile.mkdtemp(prefix="spike_graphiti_live_"))
    graphiti = Graphiti(
        graph_driver=KuzuDriver(db=str(tmp / "user_memory")),
        llm_client=llm_client,
        embedder=embedder,
    )
    try:
        await graphiti.build_indices_and_constraints()
        await graphiti.add_episode(
            name="spike-live",
            episode_body="The user prefers pytest over unittest.",
            source_description="discord-chat",
            reference_time=__import__("datetime").datetime.now().astimezone(),
            source=__import__("graphiti_core.nodes", fromlist=["EpisodeType"]).EpisodeType.message,
            group_id="primary-user",
        )
        edges = await graphiti.search("testing framework preference", group_ids=["primary-user"])
        print(json.dumps([{"fact": e.fact, "valid_at": str(e.valid_at)} for e in edges], indent=2))
    finally:
        await graphiti.close()


def main() -> int:
    """Run the spike."""
    if REOPEN_MARKER in sys.argv:
        return 0 if _child_read_back(sys.argv[sys.argv.index(REOPEN_MARKER) + 1]) >= 0 else 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="also exercise add_episode/search against a real endpoint")
    args = parser.parse_args()

    _print_versions()
    _print_signatures()
    _prove_persistence()
    if args.live:
        asyncio.run(_live_probe())
    else:
        print("\n(live LLM probe skipped; pass --live plus SPIKE_LLM_* env vars to enable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
