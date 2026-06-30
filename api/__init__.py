"""
api/__init__.py
───────────────
FastAPI application package.

    from api.main import app
    uvicorn api.main:app --reload
"""
from api.main import app

__all__ = ["app"]
