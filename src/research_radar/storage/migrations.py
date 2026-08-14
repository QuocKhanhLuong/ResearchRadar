"""Lightweight schema migrations for SQLite database backwards compatibility."""

from __future__ import annotations

import logging

from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)


def run_migrations(engine: Engine) -> None:
    """Safely apply incremental SQLite schema updates to existing databases."""

    with engine.begin() as conn:
        # Ensure schema migrations tracking table exists
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, "
                "applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ");"
            )
        )

        # Check if paper_cards table exists before migrating missing columns
        res = conn.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='paper_cards';"
            )
        ).fetchone()

        if res is not None:
            columns_res = conn.execute(text("PRAGMA table_info(paper_cards);")).fetchall()
            col_names = {row[1] for row in columns_res}

            if "tasks" not in col_names:
                logger.info("Migrating paper_cards table: adding 'tasks' column")
                conn.execute(text("ALTER TABLE paper_cards ADD COLUMN tasks JSON DEFAULT '[]';"))

            if "modalities" not in col_names:
                logger.info("Migrating paper_cards table: adding 'modalities' column")
                conn.execute(
                    text("ALTER TABLE paper_cards ADD COLUMN modalities JSON DEFAULT '[]';")
                )

            if "evaluation_conditions" not in col_names:
                logger.info(
                    "Migrating paper_cards table: adding 'evaluation_conditions' column"
                )
                conn.execute(
                    text(
                        "ALTER TABLE paper_cards "
                        "ADD COLUMN evaluation_conditions JSON DEFAULT '[]';"
                    )
                )

        conn.execute(text("INSERT OR IGNORE INTO schema_migrations (version) VALUES (1);"))
