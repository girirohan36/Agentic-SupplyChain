"""
api/main.py
────────────
FastAPI application factory for the Supply Demand Execution API.

Features:
  - Lifespan context manager → initialises DB on startup
  - CORS configured for Streamlit dashboard (localhost:8501)
  - Global HTTP exception + validation error handlers
  - Rich OpenAPI docs at /docs (Swagger) and /redoc
  - Structured health check endpoint
  - Routers: /workflow and /dashboard
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import dashboard, workflow
from config.settings import get_settings

settings = get_settings()

# ── Application start time (for uptime reporting) ─────────────────────────────
_START_TIME = time.time()


# ═══════════════════════════════════════════════════════════════════
# Lifespan — runs at startup and shutdown
# ═══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup:
      - Initialise SQLite database (create tables if they don't exist)
      - Log ready message

    Shutdown:
      - Clean up any background resources
    """
    print("🚀 Supply Demand API starting up...")

    # Init DB
    db_path     = Path(settings.database_url.replace("sqlite:///", ""))
    schema_path = Path("data/schema.sql")

    if schema_path.exists():
        schema_sql = schema_path.read_text()
        conn       = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")

        # Smart semicolon split (respect parenthesis depth)
        statements: list[str] = []
        depth, buf = 0, []
        for ch in schema_sql:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == ";" and depth == 0:
                s = "".join(buf).strip()
                if s:
                    statements.append(s)
                buf = []
            else:
                buf.append(ch)
        for stmt in statements:
            try:
                conn.execute(stmt)
            except Exception:
                pass
        conn.commit()
        conn.close()
        print(f"✅ Database ready: {db_path}")
    else:
        print(f"⚠️  Schema file not found at {schema_path} — skipping DB init")

    print("✅ Supply Demand API ready at http://localhost:8000")
    print("   Docs:  http://localhost:8000/docs")
    print("   ReDoc: http://localhost:8000/redoc")

    yield  # ← Application runs here

    print("👋 Supply Demand API shutting down...")


# ═══════════════════════════════════════════════════════════════════
# App factory
# ═══════════════════════════════════════════════════════════════════

def create_app() -> FastAPI:
    app = FastAPI(
        title="Supply Demand Execution — Agentic AI API",
        description="""
## 🏭 Supply Demand Execution — Agentic AI

End-to-end multi-agent supply chain workflow API powered by **LangGraph** and **GPT-4o**.

### Agents
| Agent | Role |
|-------|------|
| Orchestrator | Plans and routes the workflow |
| Demand Forecast | Prophet + LLM time-series forecasting |
| Supply Planning | EOQ, reorder point, safety stock |
| Inventory | Health scoring, expiry risk, KPIs |
| Procurement | PO generation, multi-supplier, split orders |
| Fulfillment | Priority routing, dispatch, backorders |
| Exception Handler | Cascading detection, auto-escalation, notifications |

### Quick start
1. `POST /workflow/run` with `{"sku_ids": ["SKU-001", "SKU-011"]}`
2. `GET /workflow/{run_id}/stream` to watch live agent events
3. `GET /dashboard/kpis` for warehouse health overview
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name":  "Supply Chain AI Team",
            "email": "ops@supply-demand-ai.example.com",
        },
        license_info={"name": "MIT"},
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",    # Streamlit default
            "http://localhost:3000",    # React dev (if added later)
            "http://localhost:8000",    # Same-origin API calls
            "*",                        # Allow all in dev (tighten in prod)
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start    = time.time()
        response = await call_next(request)
        response.headers["X-Process-Time"] = f"{(time.time() - start) * 1000:.1f}ms"
        return response

    # ── Global error handlers ─────────────────────────────────────────────────
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "Not found", "path": str(request.url.path)},
        )

    @app.exception_handler(500)
    async def server_error_handler(request: Request, exc):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    # ── Core endpoints ────────────────────────────────────────────────────────
    @app.get(
        "/",
        tags=["Core"],
        summary="API info",
        response_description="API name, version, and key endpoint links",
    )
    async def root() -> dict:
        return {
            "name":    "Supply Demand Execution — Agentic AI API",
            "version": "1.0.0",
            "status":  "operational",
            "uptime_seconds": round(time.time() - _START_TIME, 1),
            "endpoints": {
                "docs":       "/docs",
                "redoc":      "/redoc",
                "openapi":    "/openapi.json",
                "health":     "/health",
                "workflow":   "/workflow",
                "dashboard":  "/dashboard",
            },
        }

    @app.get(
        "/health",
        tags=["Core"],
        summary="Health check",
        response_description="Database connectivity and app status",
    )
    async def health_check() -> dict:
        import sqlite3 as _sq
        from pathlib import Path as _P

        db_ok   = False
        db_info = {}
        try:
            db_path = _P(settings.database_url.replace("sqlite:///", ""))
            conn    = _sq.connect(str(db_path))
            row     = conn.execute(
                "SELECT COUNT(*) as skus FROM skus"
            ).fetchone()
            runs    = conn.execute(
                "SELECT COUNT(*) FROM workflow_runs"
            ).fetchone()
            conn.close()
            db_ok   = True
            db_info = {
                "path":         str(db_path),
                "sku_count":    row[0],
                "run_count":    runs[0],
            }
        except Exception as exc:
            db_info = {"error": str(exc)}

        return {
            "status":          "healthy" if db_ok else "degraded",
            "database":        "ok"       if db_ok else "error",
            "database_detail": db_info,
            "uptime_seconds":  round(time.time() - _START_TIME, 1),
            "environment":     settings.app_env,
        }

    # ── Include routers ───────────────────────────────────────────────────────
    app.include_router(workflow.router)
    app.include_router(dashboard.router)

    return app


# ── App instance ───────────────────────────────────────────────────────────────
app = create_app()


# ── Entry point (for python api/main.py) ──────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )
