"""
data/__init__.py
────────────────
Data layer package.

Quick-start:
    from data.database import init_db, get_connection, engine
    from data.seed_data import seed
"""

from data.database import engine, get_connection, get_db, init_db

__all__ = ["engine", "get_connection", "get_db", "init_db"]
