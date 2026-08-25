"""Tests for SQLite schema migration on legacy databases."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text

from research_radar.models import Paper, PaperCard, StructuredEvidence
from research_radar.storage.database import Database
from research_radar.storage.ingestion_repository import IngestionRepository
from research_radar.storage.migrations import run_migrations
from research_radar.storage.repositories import ResearchRepository
from research_radar.storage.tables import Base

_TABLE_NAME_QUERY = "SELECT name FROM sqlite_master WHERE type='table'"


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def test_sqlite_migration_preserves_legacy_data_and_adds_columns(tmp_path_factory: object) -> None:
    db_file = tmp_path_factory.mktemp("db") / "legacy_test.db"  # type: ignore[attr-defined]
    db_url = f"sqlite:///{db_file}"
    now_str = _utc_now().isoformat()

    db = Database.create(db_url)

    # 1. Manually create legacy schema WITHOUT tasks, modalities, evaluation_conditions
    with db.engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE papers (
                    id VARCHAR(36) PRIMARY KEY,
                    canonical_key VARCHAR(512) NOT NULL,
                    normalized_title VARCHAR(512) NOT NULL,
                    title TEXT NOT NULL,
                    abstract TEXT,
                    authors JSON,
                    publication_year INTEGER,
                    venue VARCHAR(255),
                    doi VARCHAR(255),
                    url TEXT,
                    citation_count INTEGER,
                    primary_source VARCHAR(64) NOT NULL,
                    first_discovered_at DATETIME NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                );
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE paper_cards (
                    paper_id VARCHAR(36) PRIMARY KEY REFERENCES papers(id),
                    problem TEXT,
                    motivation TEXT,
                    contributions JSON,
                    methods JSON,
                    datasets JSON,
                    metrics JSON,
                    main_claims JSON,
                    limitations JSON,
                    future_work JSON,
                    failure_cases JSON,
                    source_url TEXT,
                    document_sha256 VARCHAR(64),
                    selected_sections JSON,
                    llm_provider VARCHAR(128),
                    llm_model VARCHAR(255),
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                );
                """
            )
        )

        # 2. Insert old Paper and PaperCard row
        conn.execute(
            text(
                "INSERT INTO papers (id, canonical_key, normalized_title, title, "
                "primary_source, first_discovered_at, created_at, updated_at) "
                f"VALUES ('p-legacy', 'k-legacy', 'legacy title', 'Legacy Title', 'arxiv', "
                f"'{now_str}', '{now_str}', '{now_str}');"
            )
        )
        conn.execute(
            text(
                "INSERT INTO paper_cards (paper_id, problem, contributions, methods, "
                "created_at, updated_at) "
                f"VALUES ('p-legacy', 'Legacy Problem', '[\"contrib1\"]', '[\"method1\"]', "
                f"'{now_str}', '{now_str}');"
            )
        )

    # Verify new columns do not exist yet
    with db.engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(paper_cards);")).fetchall()}
        assert "tasks" not in cols
        assert "modalities" not in cols
        assert "evaluation_conditions" not in cols

    # 3. Run database initialization/migration
    db.initialize_schema()

    # 4. Verify old row still exists
    repo = ResearchRepository(db)
    legacy_paper = repo.get_paper("p-legacy")
    assert legacy_paper is not None
    assert legacy_paper.title == "Legacy Title"

    legacy_card = repo.get_paper_card("p-legacy")
    assert legacy_card is not None
    assert legacy_card.problem == "Legacy Problem"

    # 5. Verify new columns now exist
    with db.engine.connect() as conn:
        migrated_cols = {
            row[1] for row in conn.execute(text("PRAGMA table_info(paper_cards);")).fetchall()
        }
        assert "tasks" in migrated_cols
        assert "modalities" in migrated_cols
        assert "evaluation_conditions" in migrated_cols

    # 6. Verify new StructuredEvidence can be written/read
    p2 = Paper(id="p2", title="New Title", source="arxiv")
    new_pid = repo.upsert_merged_paper(p2)
    card_new = PaperCard(
        paper_id=new_pid,
        tasks=[StructuredEvidence(value="Segmentation", status="observed")],
        modalities=[StructuredEvidence(value="MRI", status="observed")],
        evaluation_conditions=[
            StructuredEvidence(value="Scanner Shift", status="explicitly_absent")
        ],
    )
    repo.upsert_paper_card(card_new)

    read_card = repo.get_paper_card(new_pid)
    assert read_card is not None
    assert len(read_card.tasks) == 1
    assert read_card.tasks[0].value == "Segmentation"
    assert read_card.evaluation_conditions[0].status == "explicitly_absent"

    # 7. Run migration again and verify no-op (idempotent)
    run_migrations(db.engine)

    db.dispose()


def test_v1_schema_gains_document_and_ingestion_tables_without_data_loss(
    tmp_path_factory: object,
) -> None:
    """A pre-phase database gains the new tables while keeping its papers."""

    db_file = tmp_path_factory.mktemp("db") / "v1_schema.db"  # type: ignore[attr-defined]
    db_url = f"sqlite:///{db_file}"

    # Build a database at the V1 schema: the declarative metadata as it was
    # before this phase, i.e. every table except the three new ones.
    first = Database.create(db_url)
    legacy_tables = [
        table
        for name, table in Base.metadata.tables.items()
        if name not in {"document_artifacts", "ingestion_runs", "provider_retrievals"}
    ]
    Base.metadata.create_all(first.engine, tables=legacy_tables)
    run_migrations(first.engine)

    repository = ResearchRepository(first)
    paper_id = repository.upsert_merged_paper(
        Paper(
            id="openalex:W1",
            title="Legacy paper retained across migration",
            abstract="Original abstract.",
            authors=["A. Author"],
            publication_year=2020,
            venue="Journal",
            doi="10.1000/legacy",
            url="https://example.org/legacy",
            citation_count=3,
            source="openalex",
            external_ids={"openalex": "W1", "doi": "10.1000/legacy"},
        )
    )

    with first.engine.connect() as conn:
        names = set(conn.execute(text(_TABLE_NAME_QUERY)).scalars())
    assert "document_artifacts" not in names
    assert "ingestion_runs" not in names
    first.dispose()

    # Reopening the same file applies the current schema additively.
    migrated = Database.create(db_url)
    migrated.initialize_schema()

    with migrated.engine.connect() as conn:
        names = set(conn.execute(text(_TABLE_NAME_QUERY)).scalars())
        versions = set(conn.execute(text("SELECT version FROM schema_migrations")).scalars())
    assert {"document_artifacts", "ingestion_runs", "provider_retrievals"} <= names
    assert {1, 2} <= versions

    stored = ResearchRepository(migrated).get_paper(paper_id)
    assert stored is not None
    assert stored.title == "Legacy paper retained across migration"
    assert stored.doi == "10.1000/legacy"

    # The new tables are usable immediately and start empty.
    ingestion = IngestionRepository(migrated)
    assert ingestion.count_artifacts() == 0
    assert ingestion.count_ingestion_runs() == 0
    run = ingestion.start_ingestion_run(
        query="post migration query",
        requested_limit=5,
        providers=["openalex"],
    )
    assert ingestion.complete_ingestion_run(
        run.id, discovered_count=1, canonical_count=1
    ) is not None
    migrated.dispose()


def test_running_migrations_twice_is_idempotent(tmp_path_factory: object) -> None:
    """Re-running schema initialization must not fail or duplicate markers."""

    db_file = tmp_path_factory.mktemp("db") / "idempotent.db"  # type: ignore[attr-defined]
    db = Database.create(f"sqlite:///{db_file}")
    db.initialize_schema()
    db.initialize_schema()
    db.initialize_schema()

    with db.engine.connect() as conn:
        rows = conn.execute(
            text("SELECT version, COUNT(*) FROM schema_migrations GROUP BY version")
        )
        counts = dict(rows.all())
    assert counts == {1: 1, 2: 1}
    db.dispose()
