"""Offline inspector for the ResearchRadar SQLite research memory store.

Prints compact table counts plus optional paper, project, and ingestion-run
details. No Discord, no network, no LLM: safe to run against any local copy.
"""

from __future__ import annotations

import argparse
import re
import sys

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from research_radar.config import get_settings
from research_radar.storage.database import Database, create_database

# (display label, physical table name). Table names are module constants,
# never user input, so they are safe to interpolate into COUNT queries.
_COUNT_TABLES: tuple[tuple[str, str], ...] = (
    ("papers", "papers"),
    ("paper_sources", "paper_sources"),
    ("paper_cards", "paper_cards"),
    ("document_artifacts", "document_artifacts"),
    ("ingestion_runs", "ingestion_runs"),
    ("provider_retrievals", "provider_retrievals"),
    ("projects", "projects"),
    ("project_papers", "project_papers"),
    ("gap_candidates", "gap_candidates"),
    ("critic_reviews", "gap_reviews"),
)

_MIN_RUNS = 1
_MAX_RUNS = 50
_QUERY_PREVIEW_CHARS = 60


def table_counts(db: Database) -> list[tuple[str, int]]:
    """Return row counts for the core research-store tables in display order."""

    counts: list[tuple[str, int]] = []
    with db.session_factory() as session:
        for label, table in _COUNT_TABLES:
            total = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            counts.append((label, int(total)))
    return counts


def print_table_counts(db: Database) -> None:
    """Print one aligned two-column table of research-store row counts."""

    label_width = max(len(label) for label, _ in _COUNT_TABLES)
    print(f"{'table':<{label_width}}  {'rows':>8}")
    for label, count in table_counts(db):
        print(f"{label:<{label_width}}  {count:>8}")


def print_paper_details(db: Database, paper_id: str) -> bool:
    """Print stored details for one paper; return False when the id is unknown."""

    with db.session_factory() as session:
        row = session.execute(
            text(
                "SELECT title, publication_year, doi FROM papers "
                "WHERE id = :paper_id OR canonical_key = :paper_id"
            ),
            {"paper_id": paper_id},
        ).first()
        if row is None:
            print("not found")
            return False

        title, year, doi = row
        print(f"title: {title}")
        print(f"year: {year if year is not None else 'unknown'}")
        print(f"doi: {doi if doi else 'none'}")

        sources = session.execute(
            text(
                "SELECT provider, external_id FROM paper_sources "
                "WHERE paper_id = :paper_id ORDER BY provider, external_id"
            ),
            {"paper_id": paper_id},
        ).all()
        print("sources:")
        if not sources:
            print("  none")
        for provider, external_id in sources:
            print(f"  {provider}: {external_id}")

        has_card = (
            session.execute(
                text("SELECT 1 FROM paper_cards WHERE paper_id = :paper_id"),
                {"paper_id": paper_id},
            ).first()
            is not None
        )
        print(f"paper_card: {'yes' if has_card else 'no'}")

        artifacts = session.execute(
            text(
                "SELECT artifact_type, sha256, byte_size FROM document_artifacts "
                "WHERE paper_id = :paper_id ORDER BY artifact_type, sha256"
            ),
            {"paper_id": paper_id},
        ).all()
        print("document_artifacts:")
        if not artifacts:
            print("  none")
        for artifact_type, sha256, byte_size in artifacts:
            print(f"  {artifact_type} {str(sha256)[:12]} {byte_size}B")
    return True


def print_project_details(db: Database, project_name: str) -> bool:
    """Print stored details for one project; return False when it is unknown."""

    normalized = re.sub(r"[^a-z0-9]+", "-", project_name.strip().lower()).strip("-")
    with db.session_factory() as session:
        row = session.execute(
            text(
                "SELECT id, name, goal FROM projects "
                "WHERE name = :name OR normalized_name = :normalized"
            ),
            {"name": project_name, "normalized": normalized},
        ).first()
        if row is None:
            print("not found")
            return False

        project_id, name, goal = row
        paper_count = session.execute(
            text("SELECT COUNT(*) FROM project_papers WHERE project_id = :project_id"),
            {"project_id": project_id},
        ).scalar_one()
        gap_count = session.execute(
            text("SELECT COUNT(*) FROM project_gaps WHERE project_id = :project_id"),
            {"project_id": project_id},
        ).scalar_one()

        print(f"name: {name}")
        print(f"goal: {goal if goal else 'none'}")
        print(f"linked papers: {int(paper_count)}")
        print(f"linked gaps: {int(gap_count)}")
    return True


def print_latest_runs(db: Database, limit: int) -> None:
    """Print the most recent ingestion runs, newest first."""

    bounded = max(_MIN_RUNS, min(limit, _MAX_RUNS))
    with db.session_factory() as session:
        rows = session.execute(
            text(
                "SELECT started_at, status, query, discovered_count, canonical_count "
                "FROM ingestion_runs ORDER BY started_at DESC LIMIT :limit"
            ),
            {"limit": bounded},
        ).all()
        if not rows:
            print("no ingestion runs recorded")
            return
        for started_at, status, query_value, discovered, canonical in rows:
            preview = str(query_value)
            if len(preview) > _QUERY_PREVIEW_CHARS:
                preview = f"{preview[:_QUERY_PREVIEW_CHARS - 1]}…"
            summary = (
                f"{started_at} | {status} | {preview} "
                f"| discovered={discovered} | canonical={canonical}"
            )
            print(summary)


def inspect_store(
    db: Database,
    *,
    paper_id: str | None = None,
    project: str | None = None,
    latest_runs: int = 0,
) -> int:
    """Run the requested inspections against one open database handle."""

    print_table_counts(db)
    if paper_id is not None:
        print()
        if not print_paper_details(db, paper_id):
            return 1
    if project is not None:
        print()
        if not print_project_details(db, project):
            return 1
    if latest_runs > 0:
        print()
        print_latest_runs(db, latest_runs)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the offline inspector's command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Inspect the ResearchRadar SQLite research memory store."
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLite database URL (default: DATABASE_URL from settings)",
    )
    parser.add_argument("--paper-id", default=None, help="Show details for one paper id")
    parser.add_argument("--project", default=None, help="Show details for one project name")
    parser.add_argument(
        "--latest-runs",
        type=int,
        default=0,
        help=f"Show the N most recent ingestion runs ({_MIN_RUNS}..{_MAX_RUNS})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point that maps failures to safe exit codes without tracebacks."""

    args = _parse_args(argv)
    database_url = args.database_url or get_settings().database_url
    try:
        db = create_database(database_url)
        db.initialize_schema()
        exit_code = inspect_store(
            db,
            paper_id=args.paper_id,
            project=args.project,
            latest_runs=args.latest_runs,
        )
        db.dispose()
    except SQLAlchemyError:
        print("A database error occurred while inspecting the research store.", file=sys.stderr)
        raise SystemExit(2) from None
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
