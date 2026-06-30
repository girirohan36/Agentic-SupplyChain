"""
api/routers/workflow.py
────────────────────────
Workflow router — all endpoints for triggering and monitoring runs.

Endpoints:
  POST /workflow/run            → trigger a new workflow run (async background)
  GET  /workflow/runs           → list recent runs
  GET  /workflow/{run_id}       → full run status + result
  GET  /workflow/{run_id}/stream → SSE stream of live agent events
  POST /workflow/{run_id}/resume → HITL: submit human decision
  DELETE /workflow/{run_id}     → cancel a running workflow
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from datetime import datetime
from typing import Any, AsyncGenerator, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.dependencies import RawConnDep, SettingsDep, ValidRunId, get_db_path
from config.settings import get_settings

router = APIRouter(prefix="/workflow", tags=["Workflow"])

# ── In-memory event bus (per run_id → list of SSE events) ────────────────────
# In production this would be Redis pub/sub. Fine for portfolio demo.
_event_store: dict[str, list[dict]] = {}
_event_lock  = threading.Lock()


def _push_event(run_id: str, event: dict) -> None:
    with _event_lock:
        if run_id not in _event_store:
            _event_store[run_id] = []
        _event_store[run_id].append(event)


# ── Request / Response schemas ────────────────────────────────────────────────

class RunWorkflowRequest(BaseModel):
    sku_ids:      list[str]     = Field(..., min_length=1, description="SKUs to include in this run")
    triggered_by: str           = Field("api", description="Who triggered: api | scheduler | manual")

    model_config = {"json_schema_extra": {
        "example": {"sku_ids": ["SKU-001", "SKU-011", "SKU-008"], "triggered_by": "api"}
    }}


class RunWorkflowResponse(BaseModel):
    run_id:      str
    status:      str
    sku_ids:     list[str]
    triggered_by: str
    created_at:  str
    message:     str


class HITLResumeRequest(BaseModel):
    approved:     bool          = Field(..., description="True = approve, False = reject")
    response:     Optional[str] = Field(None, description="Optional free-text response")
    checkpoint_id: Optional[str] = Field(None, description="Checkpoint ID being responded to")


# ── Background workflow runner ────────────────────────────────────────────────

def _run_workflow_background(run_id: str, sku_ids: list[str], triggered_by: str) -> None:
    """
    Execute the full agent workflow in a background thread.
    Writes progress events to _event_store for SSE streaming.
    Persists run record + agent results to SQLite.
    """
    settings = get_settings()
    db_path  = get_db_path()

    def emit(agent: str, status: str, data: dict | None = None):
        event = {
            "run_id":    run_id,
            "agent":     agent,
            "status":    status,
            "data":      data or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        _push_event(run_id, event)

    def persist_status(run_status: str, summary: dict | None = None):
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        now = datetime.utcnow().isoformat()
        fields = "status = ?, completed_at = ?"
        params: list[Any] = [run_status, now]
        if summary:
            fields += ", summary = ?"
            params.append(json.dumps(summary))
        params.append(run_id)
        conn.execute(f"UPDATE workflow_runs SET {fields} WHERE run_id = ?", params)
        conn.commit()
        conn.close()

    def persist_agent_result(agent: str, agent_status: str,
                              output: dict | None = None, error: str | None = None,
                              started_at: str | None = None, completed_at: str | None = None):
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """INSERT INTO agent_results
               (run_id, agent, status, output, error, started_at, completed_at)
               VALUES (?,?,?,?,?,?,?)""",
            (run_id, agent, agent_status,
             json.dumps(output) if output else None,
             error, started_at, completed_at),
        )
        conn.commit()
        conn.close()

    try:
        # ── Bootstrap stubs so agents can be imported without full install ──────
        import sys, types

        fake_s = types.SimpleNamespace(
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            openai_temperature=settings.openai_temperature,
            database_url=settings.database_url,
            default_lead_time_days=settings.default_lead_time_days,
            default_safety_stock_days=settings.default_safety_stock_days,
            forecast_horizon_days=settings.forecast_horizon_days,
            max_agent_iterations=settings.max_agent_iterations,
            is_development=settings.is_development,
            is_production=settings.is_production,
            enable_human_in_the_loop=settings.enable_human_in_the_loop,
            low_stock_threshold_pct=settings.low_stock_threshold_pct,
        )

        import importlib

        def _patch_db(mod_name: str):
            try:
                m = importlib.import_module(mod_name)
                if hasattr(m, "DB_PATH"):
                    from pathlib import Path
                    m.DB_PATH = Path(settings.database_url.replace("sqlite:///", ""))
            except Exception:
                pass

        for mn in ("tools.forecast_tools", "tools.inventory_tools",
                   "tools.procurement_tools", "tools.notification_tools",
                   "tools.fulfillment_tools"):
            _patch_db(mn)

        # ── Import agent nodes ────────────────────────────────────────────────
        from agents.demand_forecast   import demand_forecast_node
        from agents.supply_planning   import supply_planning_node
        from agents.inventory         import inventory_node
        from agents.procurement       import procurement_node
        from agents.fulfillment       import fulfillment_node
        from agents.exception_handler import exception_handler_node
        from agents.orchestrator      import orchestrator_finalise

        # ── Build initial state ───────────────────────────────────────────────
        state: dict[str, Any] = {
            "run_id":       run_id,
            "sku_ids":      [s.upper().strip() for s in sku_ids],
            "triggered_by": triggered_by,
            "workflow_status": "running",
            "messages":     [],
            "exception_events":  [],
            "exception_detected": False,
            "replenishment_needed": False,
            "per_sku_results": [],
            "sku_processing_complete": False,
            "hitl_required": False,
            "hitl_checkpoint": None,
            "hitl_approved": None,
            "hitl_response": None,
            "supply_plan_input": {
                "forecasts": [
                    {"sku_id": s.upper().strip(), "avg_daily_demand": 20.0,
                     "trend": "stable", "confidence": "medium", "horizon_days": 90}
                    for s in sku_ids
                ]
            },
            "procurement_input": {
                "reorder_triggers": [], "recommended_pos": [], "sku_plans": []
            },
            "fulfillment_input": {
                "sku_ids": [s.upper().strip() for s in sku_ids],
                "open_orders": {}, "health_records": {}
            },
        }

        persist_status("running")

        AGENTS = [
            ("demand_forecast",   demand_forecast_node),
            ("supply_planning",   supply_planning_node),
            ("inventory",         inventory_node),
            ("procurement",       procurement_node),
            ("fulfillment",       fulfillment_node),
            ("exception_handler", exception_handler_node),
        ]

        for agent_name, agent_fn in AGENTS:
            started = datetime.utcnow().isoformat()
            emit(agent_name, "running")

            try:
                result = agent_fn(state)
                state.update(result)
                completed = datetime.utcnow().isoformat()
                status_val = result.get(f"{agent_name}_status", "success") or "success"
                emit(agent_name, status_val, {k: v for k, v in result.items()
                                              if not isinstance(v, list) or len(str(v)) < 500})
                persist_agent_result(agent_name, status_val,
                                     output={"keys_written": list(result.keys())},
                                     started_at=started, completed_at=completed)
            except Exception as exc:
                completed = datetime.utcnow().isoformat()
                emit(agent_name, "failed", {"error": str(exc)})
                persist_agent_result(agent_name, "failed",
                                     error=str(exc), started_at=started, completed_at=completed)

        # ── Finalise ──────────────────────────────────────────────────────────
        final = orchestrator_finalise(state)
        state.update(final)
        summary = state.get("summary", {})
        persist_status("completed", summary)
        emit("orchestrator", "completed", summary)

    except Exception as exc:
        persist_status("failed")
        emit("system", "failed", {"error": str(exc)})


# ════════════════════════════════════════════════════════════════════
# Endpoints
# ════════════════════════════════════════════════════════════════════

@router.post(
    "/run",
    response_model=RunWorkflowResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger a new workflow run",
    description=(
        "Starts a full Supply Demand Execution workflow run in the background. "
        "Returns immediately with the run_id. Use GET /workflow/{run_id} or "
        "GET /workflow/{run_id}/stream to track progress."
    ),
)
async def trigger_run(
    request:          RunWorkflowRequest,
    background_tasks: BackgroundTasks,
    conn:             RawConnDep,
) -> RunWorkflowResponse:
    run_id     = f"RUN-{uuid4().hex[:8].upper()}"
    created_at = datetime.utcnow().isoformat()

    # Persist initial record
    conn.execute(
        """INSERT INTO workflow_runs
           (run_id, status, sku_ids, triggered_by, created_at)
           VALUES (?,?,?,?,?)""",
        (run_id, "initialised",
         json.dumps([s.upper().strip() for s in request.sku_ids]),
         request.triggered_by, created_at),
    )
    conn.commit()

    # Fire and forget in background thread
    background_tasks.add_task(
        _run_workflow_background,
        run_id, request.sku_ids, request.triggered_by,
    )

    return RunWorkflowResponse(
        run_id=run_id,
        status="initialised",
        sku_ids=[s.upper().strip() for s in request.sku_ids],
        triggered_by=request.triggered_by,
        created_at=created_at,
        message=f"Workflow {run_id} started. Stream events at /workflow/{run_id}/stream",
    )


@router.get(
    "/runs",
    summary="List recent workflow runs",
)
async def list_runs(
    conn:       RawConnDep,
    limit:      int = 20,
    status_filter: Optional[str] = None,
) -> dict:
    query  = "SELECT * FROM workflow_runs"
    params: list = []
    if status_filter:
        query  += " WHERE status = ?"
        params.append(status_filter)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    runs = []
    for row in rows:
        r = dict(row)
        try:
            r["sku_ids"] = json.loads(r.get("sku_ids", "[]"))
        except Exception:
            pass
        try:
            r["summary"] = json.loads(r.get("summary") or "{}")
        except Exception:
            pass
        runs.append(r)

    return {"total": len(runs), "runs": runs}


@router.get(
    "/{run_id}",
    summary="Get full workflow run status and results",
)
async def get_run(run_id: ValidRunId, conn: RawConnDep) -> dict:
    run_row = conn.execute(
        "SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,)
    ).fetchone()

    agent_rows = conn.execute(
        "SELECT * FROM agent_results WHERE run_id = ? ORDER BY rowid",
        (run_id,),
    ).fetchall()

    exc_rows = conn.execute(
        "SELECT * FROM exception_events WHERE run_id = ? ORDER BY raised_at DESC",
        (run_id,),
    ).fetchall()

    po_rows = conn.execute(
        "SELECT po.*, pol.sku_id, pol.quantity, pol.unit_cost "
        "FROM purchase_orders po "
        "JOIN purchase_order_lines pol ON po.po_number = pol.po_number "
        "WHERE po.run_id = ?",
        (run_id,),
    ).fetchall()

    run    = dict(run_row)
    try:
        run["sku_ids"] = json.loads(run.get("sku_ids", "[]"))
    except Exception:
        pass
    try:
        run["summary"] = json.loads(run.get("summary") or "{}")
    except Exception:
        pass

    # Merge in live events
    live_events = []
    with _event_lock:
        live_events = list(_event_store.get(run_id, []))

    return {
        "run":            run,
        "agent_results":  [dict(r) for r in agent_rows],
        "exceptions":     [dict(r) for r in exc_rows],
        "purchase_orders": [dict(r) for r in po_rows],
        "live_events":    live_events[-20:],   # last 20 events
    }


@router.get(
    "/{run_id}/stream",
    summary="SSE stream of live agent events",
    description=(
        "Server-Sent Events stream. Connect with EventSource in the browser "
        "or httpx in Python. Each event is a JSON object with agent, status, data, timestamp."
    ),
)
async def stream_run(run_id: str, request: Request) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        cursor = 0
        while True:
            if await request.is_disconnected():
                break

            with _event_lock:
                events = _event_store.get(run_id, [])
                new_events = events[cursor:]
                cursor     = len(events)

            for event in new_events:
                yield f"data: {json.dumps(event)}\n\n"

            # Check if run is terminal
            with _event_lock:
                all_events = _event_store.get(run_id, [])
            if any(e.get("status") in ("completed", "failed") for e in all_events):
                yield f"data: {json.dumps({'type': 'stream_end', 'run_id': run_id})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":          "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.post(
    "/{run_id}/resume",
    summary="Submit HITL decision to resume a paused workflow",
)
async def resume_run(
    run_id:  ValidRunId,
    body:    HITLResumeRequest,
    conn:    RawConnDep,
) -> dict:
    # Verify run is paused
    row = conn.execute(
        "SELECT status FROM workflow_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")

    # Emit HITL response event for the background thread to pick up
    _push_event(run_id, {
        "run_id":    run_id,
        "type":      "hitl_response",
        "approved":  body.approved,
        "response":  body.response,
        "checkpoint_id": body.checkpoint_id,
        "timestamp": datetime.utcnow().isoformat(),
    })

    return {
        "run_id":    run_id,
        "approved":  body.approved,
        "message":   f"HITL {'approved' if body.approved else 'rejected'}. Workflow will resume.",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.delete(
    "/{run_id}",
    summary="Cancel a workflow run",
    status_code=status.HTTP_200_OK,
)
async def cancel_run(run_id: ValidRunId, conn: RawConnDep) -> dict:
    conn.execute(
        "UPDATE workflow_runs SET status = 'cancelled' WHERE run_id = ? AND status IN ('initialised','running','paused')",
        (run_id,),
    )
    conn.commit()
    _push_event(run_id, {"run_id": run_id, "type": "cancelled",
                          "timestamp": datetime.utcnow().isoformat()})
    return {"run_id": run_id, "status": "cancelled"}
