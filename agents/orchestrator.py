"""
agents/orchestrator.py
───────────────────────
The Orchestrator Agent — master controller of the supply-demand workflow.

Responsibilities:
  1. PLAN    — on entry, decide which agents need to run and in what order
  2. ROUTE   — after each agent, decide the next step via conditional edges
  3. FINALISE — once all agents are done, compile the run summary

The Orchestrator runs as TWO nodes in the graph:
  - "orchestrator_plan"     → START → this node → demand_forecast
  - "orchestrator_finalise" → all agents done → this node → END

This two-node design keeps the planning and summary logic clean and
gives the graph a clear entry and exit point.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from graph.state import WorkflowState, is_all_agents_done

settings = get_settings()

# ── LLM singleton (lazy — only instantiated when agent runs) ──────────────────
_llm: ChatOpenAI | None = None

def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            api_key=settings.openai_api_key,
        )
    return _llm


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — orchestrator_plan
# ─────────────────────────────────────────────────────────────────────────────

def orchestrator_plan(state: WorkflowState) -> dict[str, Any]:
    """
    LangGraph node: orchestrator_plan
    ──────────────────────────────────
    Entry point of the graph. Analyses the incoming run request and
    produces a structured execution plan.

    Reads:
      state["run_id"], state["sku_ids"]

    Writes:
      state["orchestrator_plan"]   — LLM-generated execution plan
      state["workflow_status"]     — "running"
      state["demand_forecast_status"] — "pending" (reset for clarity)

    Returns a partial state dict (LangGraph merges it).
    """
    run_id  = state.get("run_id", "UNKNOWN")
    sku_ids = state.get("sku_ids", [])

    system_prompt = """You are the Orchestrator of a Supply Demand Execution AI system.
Your job is to analyse an incoming workflow run request and produce a clear, structured
execution plan in JSON.

The available agents are (in default execution order):
  1. demand_forecast   — forecasts demand for each SKU using Prophet + LLM
  2. supply_planning   — runs EOQ and reorder point calculations
  3. inventory         — scores inventory health, flags critical SKUs
  4. procurement       — generates Purchase Orders for SKUs that need replenishment
  5. fulfillment       — routes open orders to available stock
  6. exception_handler — resolves any disruptions, sends alerts

Return a JSON object with this exact shape:
{
  "execution_order": ["demand_forecast", "supply_planning", "inventory", "procurement", "fulfillment"],
  "priority_skus": ["SKU-001", ...],  // SKUs that likely need urgent attention
  "notes": "brief plain-English rationale"
}"""

    user_prompt = f"""New workflow run:
  run_id:  {run_id}
  sku_ids: {json.dumps(sku_ids)}

Produce the execution plan JSON."""

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        # Parse the JSON plan from the response
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        plan = json.loads(raw.strip())
    except Exception as exc:
        # Graceful fallback — proceed with default ordering
        plan = {
            "execution_order": [
                "demand_forecast", "supply_planning", "inventory",
                "procurement", "fulfillment",
            ],
            "priority_skus": sku_ids,
            "notes": f"LLM plan generation failed ({exc}); using default order.",
        }

    return {
        "orchestrator_plan":        plan,
        "workflow_status":          "running",
        "demand_forecast_status":   "pending",
        "supply_plan_status":       "pending",
        "inventory_status":         "pending",
        "procurement_status":       "pending",
        "fulfillment_status":       "pending",
        "exception_status":         "pending",
        "exception_detected":       False,
        "replenishment_needed":     False,
        "run_complete":             False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — orchestrator_finalise
# ─────────────────────────────────────────────────────────────────────────────

def orchestrator_finalise(state: WorkflowState) -> dict[str, Any]:
    """
    LangGraph node: orchestrator_finalise
    ──────────────────────────────────────
    Final node before END. Aggregates all agent results into a
    human-readable run summary and marks the workflow complete.

    Reads:
      All agent result and status fields, exception_events

    Writes:
      state["summary"]          — structured run summary dict
      state["workflow_status"]  — "completed"
      state["run_complete"]     — True
    """
    sku_ids          = state.get("sku_ids", [])
    exceptions       = state.get("exception_events", [])
    inv_result       = state.get("inventory_result") or {}
    procurement_res  = state.get("procurement_result") or {}
    forecast_res     = state.get("demand_forecast_result") or {}
    supply_res       = state.get("supply_plan_result") or {}

    # Count issued POs
    issued_pos = procurement_res.get("issued_pos", [])
    po_count   = len(issued_pos)
    po_value   = sum(po.get("total_value", 0) for po in issued_pos)

    # Count critical inventory SKUs
    inv_records      = inv_result.get("records", [])
    critical_skus    = [r["sku_id"] for r in inv_records
                        if r.get("health_status") in ("critical", "stock_out")]
    reorder_triggers = inv_result.get("reorder_triggers", [])

    # Unresolved exceptions
    unresolved = [e for e in exceptions if not e.get("resolved")]

    summary = {
        "run_id":             state.get("run_id"),
        "completed_at":       datetime.utcnow().isoformat(),
        "skus_processed":     len(sku_ids),
        "forecast_horizon":   forecast_res.get("horizon_days"),
        "reorder_triggers":   reorder_triggers,
        "critical_sku_count": len(critical_skus),
        "critical_skus":      critical_skus,
        "pos_issued":         po_count,
        "po_total_value_usd": round(po_value, 2),
        "total_exceptions":   len(exceptions),
        "unresolved_exceptions": len(unresolved),
        "agent_statuses": {
            "demand_forecast":   state.get("demand_forecast_status"),
            "supply_planning":   state.get("supply_plan_status"),
            "inventory":         state.get("inventory_status"),
            "procurement":       state.get("procurement_status"),
            "fulfillment":       state.get("fulfillment_status"),
            "exception_handler": state.get("exception_status"),
        },
    }

    return {
        "summary":          summary,
        "workflow_status":  "completed",
        "run_complete":     True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routing helpers (used as conditional edge functions in workflow.py)
# ─────────────────────────────────────────────────────────────────────────────

def route_after_inventory(state: WorkflowState) -> str:
    """
    Conditional edge: after the Inventory Agent.
    If any SKU triggered a reorder → go to Procurement.
    Otherwise skip straight to Fulfillment.
    """
    if state.get("replenishment_needed"):
        return "procurement"
    return "fulfillment"


def route_after_procurement(state: WorkflowState) -> str:
    """
    Conditional edge: after the Procurement Agent.
    If HITL is enabled and POs were generated → pause for human approval.
    Otherwise continue to Fulfillment.
    """
    if (
        settings.enable_human_in_the_loop
        and state.get("hitl_required")
    ):
        return "hitl_interrupt"
    return "fulfillment"


def route_after_fulfillment(state: WorkflowState) -> str:
    """
    Conditional edge: after the Fulfillment Agent.
    If any exception was detected → go to Exception Handler.
    Otherwise finalise.
    """
    if state.get("exception_detected"):
        return "exception_handler"
    return "orchestrator_finalise"


def route_after_exception(state: WorkflowState) -> str:
    """
    Conditional edge: after the Exception Handler.
    Always routes to finalise (exception handler resolves in-place).
    """
    return "orchestrator_finalise"
