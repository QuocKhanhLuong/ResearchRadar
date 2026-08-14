"""Tests for SQLite schema migration on legacy databases."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text

from research_radar.models import Paper, PaperCard, StructuredEvidence
from research_radar.storage.database import Database
from research_radar.storage.migrations import run_migrations
from research_radar.storage.repositories import ResearchRepository


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
