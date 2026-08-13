"""SQLite engine and schema lifecycle for the one-process application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from research_radar.storage.tables import Base


@dataclass(slots=True)
class Database:
    """Own a SQLAlchemy engine and short-lived-session factory.

    Repositories create and close a new ``Session`` for each method.  The
    factory is therefore safe to hand to a service that calls synchronous
    storage work through ``asyncio.to_thread``.
    """

    engine: Engine
    session_factory: sessionmaker[Session]

    @classmethod
    def create(cls, database_url: str) -> Database:
        """Create a database handle without changing schema state yet."""

        return create_database(database_url)

    def initialize_schema(self) -> None:
        """Create the initial V1 schema when it does not already exist."""

        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        """Release database connections during application shutdown."""

        self.engine.dispose()


DatabaseTarget = Database | Engine


def create_database(database_url: str) -> Database:
    """Build a configured SQLAlchemy database handle from a URL.

    SQLite is the V1 default.  File-backed URLs create only their configured
    parent directory; no alternate or implicit data location is selected.
    """

    url = make_url(database_url)
    _ensure_sqlite_parent_directory(url)

    engine_kwargs: dict[str, object] = {"pool_pre_ping": True}
    if url.get_backend_name() == "sqlite":
        engine_kwargs["connect_args"] = {"check_same_thread": False}
        if _is_memory_sqlite(url):
            engine_kwargs["poolclass"] = StaticPool

    engine = create_engine(url, **engine_kwargs)
    if url.get_backend_name() == "sqlite":
        event.listen(engine, "connect", _configure_sqlite_connection)

    return Database(
        engine=engine,
        session_factory=sessionmaker(bind=engine, expire_on_commit=False),
    )


def initialize_schema(target: DatabaseTarget) -> None:
    """Initialize schema for a ``Database`` or raw SQLAlchemy engine."""

    engine = target.engine if isinstance(target, Database) else target
    Base.metadata.create_all(engine)


def _ensure_sqlite_parent_directory(url: URL) -> None:
    if url.get_backend_name() != "sqlite" or _is_memory_sqlite(url):
        return

    database_name = url.database
    if database_name is None:
        return
    Path(database_name).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _is_memory_sqlite(url: URL) -> bool:
    return url.database in (None, "", ":memory:")


def _configure_sqlite_connection(dbapi_connection: object, _: object) -> None:
    """Enable SQLite behavior appropriate to a short-transaction process."""

    cursor = dbapi_connection.cursor()  # type: ignore[union-attr]
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        # WAL improves reader/writer coexistence for a file database.  SQLite
        # may decline it for an in-memory DB, which is harmless for tests.
        cursor.execute("PRAGMA journal_mode=WAL")
    finally:
        cursor.close()
