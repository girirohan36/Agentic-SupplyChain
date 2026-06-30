"""
api/dependencies.py
────────────────────
Shared FastAPI dependency functions.

Exposes:
  - get_db()         → yields SQLAlchemy Session (reuses data/database.py)
  - get_settings()   → returns cached Settings singleton
  - valid_run_id()   → validates run_id path parameter exists in DB
  - PaginationParams → common limit/offset query params
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Query, status

from config.settings import get_settings as _get_settings, Settings

# ── Settings ──────────────────────────────────────────────────────────────────

def get_settings() -> Settings:
    return _get_settings()

SettingsDep = Annotated[Settings, Depends(get_settings)]


# ── Database session ──────────────────────────────────────────────────────────

def get_db_path() -> Path:
    settings = _get_settings()
    return Path(settings.database_url.replace("sqlite:///", ""))


def get_raw_conn():
    """
    Yield a raw sqlite3 connection for read-heavy dashboard queries.
    Faster than SQLAlchemy for simple SELECT operations.
    """
    db_path = get_db_path()
    conn    = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
    finally:
        conn.close()

RawConnDep = Annotated[sqlite3.Connection, Depends(get_raw_conn)]


# ── Run ID validator ──────────────────────────────────────────────────────────

def valid_run_id(
    run_id: str,
    conn: sqlite3.Connection = Depends(get_raw_conn),
) -> str:
    """
    Path-parameter dependency that verifies a run_id exists in workflow_runs.
    Raises HTTP 404 if not found.
    """
    row = conn.execute(
        "SELECT run_id FROM workflow_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow run '{run_id}' not found.",
        )
    return run_id

ValidRunId = Annotated[str, Depends(valid_run_id)]


# ── Pagination ────────────────────────────────────────────────────────────────

class PaginationParams:
    def __init__(
        self,
        limit:  int = Query(20, ge=1, le=100, description="Max records to return"),
        offset: int = Query(0,  ge=0,         description="Records to skip"),
    ):
        self.limit  = limit
        self.offset = offset

PaginationDep = Annotated[PaginationParams, Depends(PaginationParams)]
