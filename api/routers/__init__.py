"""
api/routers/__init__.py
────────────────────────
Router sub-package — exports both routers for api/main.py inclusion.

    from api.routers import dashboard, workflow
    app.include_router(workflow.router)
    app.include_router(dashboard.router)
"""
from api.routers import dashboard, workflow

__all__ = ["dashboard", "workflow"]
