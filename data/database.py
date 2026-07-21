"""
data/database.py
────────────────
SQLAlchemy engine + session factory for the SQLite database.

Exposes:
  - engine        → SQLAlchemy Engine (use for raw SQL via text())
  - SessionLocal  → session factory for ORM queries
  - get_db()      → FastAPI dependency that yields a session
  - init_db()     → run schema.sql and create all tables
  - get_connection() → context manager for raw sqlite3 access
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.settings import get_settings

settings = get_settings()

# ── SQLAlchemy engine ─────────────────────────────────────────────────────────

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # required for SQLite + FastAPI
    echo=settings.is_development,               # log SQL in dev mode
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, _connection_record):
    """Apply performance and integrity pragmas on every new connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("PRAGMA journal_mode = WAL")
    cursor.execute("PRAGMA synchronous = NORMAL")
    cursor.close()


# ── Session factory ───────────────────────────────────────────────────────────

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ── SQLAlchemy declarative base (for ORM models if added later) ───────────────

class Base(DeclarativeBase):
    pass


# ── FastAPI dependency ────────────────────────────────────────────────────────

def get_db():
    """
    FastAPI dependency — yields a SQLAlchemy Session, always closes it.

    Usage in a router:
        @router.get("/skus")
        def list_skus(db: Session = Depends(get_db)):
            ...
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Schema initialiser ────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Execute data/schema.sql to create all tables if they don't exist.
    Safe to call multiple times (uses IF NOT EXISTS).

    Called automatically by seed_data.py and at app startup.
    """
    schema_path = Path(__file__).parent / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    sql = schema_path.read_text()
    with engine.begin() as conn:
        # SQLite doesn't support multi-statement executemany via SQLAlchemy text()
        # so we split on semicolons and run each statement individually.
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        for stmt in statements:
            conn.execute(text(stmt))

    print(f"✅ Database initialised: {settings.database_url}")


# ── Raw sqlite3 context manager (for bulk inserts in seed_data.py) ────────────

@contextmanager
def get_connection():
    """
    Yield a raw sqlite3 connection for bulk inserts / executemany.
    Much faster than SQLAlchemy for seed operations.

    Usage:
        with get_connection() as conn:
            conn.executemany("INSERT INTO skus ...", rows)
            conn.commit()
    """
    db_path = settings.database_url.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
    finally:
        conn.close()
