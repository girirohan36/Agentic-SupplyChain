"""
graph/workflow.py  (Phase 4 — fan-out/fan-in parallel SKU processing)
───────────────────────────────────────────────────────────────────────
Supply Demand Execution — LangGraph StateGraph

Phase 4 adds a parallel fan-out/fan-in pattern using the LangGraph Send API:

  START → orchestrator_plan → demand_forecast → supply_planning
       → sku_dispatcher ──┬──→ sku_worker[SKU-001] ──┐
                          ├──→ sku_worker[SKU-002] ──┤
                          ├──→    …×20 SKUs…          ├──→ sku_aggregator
                          └──→ sku_worker[SKU-020] ──┘
       → inventory → procurement → fulfillment
       → exception_handler → orchestrator_finalise → END

The sku_worker node runs inventory scoring + supply planning for one SKU.
The sku_aggregator fan-in node merges all per-SKU results.

Usage:
    from graph.workflow import build_graph
    from graph.state import initial_state
    from graph.checkpointer import get_checkpointer, thread_config

    graph = build_graph(get_checkpointer())
    result = graph.invoke(
        initial_state("RUN-001", ["SKU-001", "SKU-002"]),
        config=thread_config("RUN-001"),
    )
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from agents.demand_forecast   import demand_forecast_node
from agents.exception_handler import exception_handler_node
from agents.fulfillment        import fulfillment_node
from agents.inventory          import inventory_node
from agents.orchestrator       import (
    orchestrator_finalise,
    orchestrator_plan,
    route_after_exception,
    route_after_fulfillment,
    route_after_inventory,
    route_after_procurement,
)
from agents.procurement        import procurement_node
from agents.supply_planning    import supply_planning_node
from graph.state               import WorkflowState

# ── Node name constants ────────────────────────────────────────────────────────
NODE_ORCHESTRATOR_PLAN      = "orchestrator_plan"
NODE_DEMAND_FORECAST        = "demand_forecast"
NODE_SUPPLY_PLANNING        = "supply_planning"
NODE_SKU_DISPATCHER         = "sku_dispatcher"
NODE_SKU_WORKER             = "sku_worker"
NODE_SKU_AGGREGATOR         = "sku_aggregator"
NODE_INVENTORY              = "inventory"
NODE_PROCUREMENT            = "procurement"
NODE_FULFILLMENT            = "fulfillment"
NODE_EXCEPTION_HANDLER      = "exception_handler"
NODE_ORCHESTRATOR_FINALISE  = "orchestrator_finalise"


# ═══════════════════════════════════════════════════════════════════
# Phase 4 — Fan-out / Fan-in nodes
# ═══════════════════════════════════════════════════════════════════

def sku_dispatcher(state: WorkflowState) -> list[Any]:
    """
    Fan-out dispatcher node.

    Uses LangGraph's Send API to spawn one sku_worker node per SKU
    in state["sku_ids"]. Each worker receives a sub-state containing
    only its own SKU.

    Returns a list of Send() objects — LangGraph executes them in parallel.
    """
    try:
        from langgraph.types import Send
    except ImportError:
        # Fallback for older langgraph versions — return empty list
        # (workflow will skip fan-out and go straight to inventory)
        return []

    sku_ids      = state.get("sku_ids", [])
    supply_plans = state.get("supply_plan_result", {}).get("sku_plans", [])
    plan_map     = {p["sku_id"]: p for p in supply_plans}
    forecasts    = state.get("supply_plan_input", {}).get("forecasts", [])
    fc_map       = {f["sku_id"]: f for f in forecasts}

    sends = []
    for sku_id in sku_ids:
        per_sku_state = {
            "run_id":      state.get("run_id"),
            "sku_ids":     [sku_id],
            "current_sku": sku_id,
            "supply_plan_input": {
                "forecasts": [fc_map.get(sku_id, {"sku_id": sku_id, "avg_daily_demand": 10.0})]
            },
            "procurement_input": {
                "sku_plans":         [plan_map.get(sku_id, {"sku_id": sku_id, "needs_replenishment": False})],
                "reorder_triggers":  [sku_id] if plan_map.get(sku_id, {}).get("needs_replenishment") else [],
                "recommended_pos":   [],
            },
            "exception_events":  [],
            "exception_detected": False,
        }
        sends.append(Send(NODE_SKU_WORKER, per_sku_state))

    return sends


def sku_worker(state: WorkflowState) -> dict[str, Any]:
    """
    Per-SKU worker node (runs in parallel via fan-out).

    Runs inventory scoring for a single SKU and writes
    its result into state["per_sku_results"] for the aggregator.

    Each parallel instance has its own isolated state slice.
    """
    sku_id  = state.get("current_sku") or (state.get("sku_ids") or ["UNKNOWN"])[0]
    run_id  = state.get("run_id", "UNKNOWN")

    # Run inventory scoring for this single SKU
    inv_result = inventory_node(state)
    records    = inv_result.get("inventory_result", {}).get("records", [])
    record     = records[0] if records else {"sku_id": sku_id, "health_status": "healthy", "health_score": 1.0}

    return {
        "per_sku_results": [record],   # appended to list by reducer
    }


def sku_aggregator(state: WorkflowState) -> dict[str, Any]:
    """
    Fan-in aggregator node.

    Waits for all sku_worker nodes to complete, merges per_sku_results
    into a single InventorySnapshot, then sets inventory_result in state.
    """
    per_sku = state.get("per_sku_results", [])
    run_id  = state.get("run_id", "UNKNOWN")

    from datetime import datetime

    critical_count = sum(1 for r in per_sku if r.get("health_status") in ("critical", "stock_out"))
    at_risk_count  = sum(1 for r in per_sku if r.get("health_status") == "at_risk")
    healthy_count  = sum(1 for r in per_sku if r.get("health_status") == "healthy")
    reorder_triggers = [r["sku_id"] for r in per_sku if r.get("reorder_triggered")]

    inventory_result = {
        "run_id":            run_id,
        "generated_at":      datetime.utcnow().isoformat(),
        "location_id":       "DC-01",
        "records":           per_sku,
        "total_skus":        len(per_sku),
        "critical_count":    critical_count,
        "at_risk_count":     at_risk_count,
        "healthy_count":     healthy_count,
        "reorder_triggers":  reorder_triggers,
        "inventory_commentary": f"Parallel assessment complete: {critical_count} critical, {at_risk_count} at-risk.",
        "warnings":          [],
    }

    open_orders = {r["sku_id"]: r.get("on_hand", 0.0) * 0.1 for r in per_sku}

    return {
        "inventory_result":    inventory_result,
        "inventory_status":    "success",
        "sku_processing_complete": True,
        "replenishment_needed": bool(reorder_triggers),
        "fulfillment_input": {
            "sku_ids":        state.get("sku_ids", []),
            "open_orders":    open_orders,
            "health_records": {r["sku_id"]: r for r in per_sku},
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Conditional edge routers
# ═══════════════════════════════════════════════════════════════════

def _route_after_supply_planning(state: WorkflowState) -> list | str:
    """
    After supply_planning, decide between:
      - fan-out path (sku_dispatcher → parallel workers)
      - direct inventory path (fallback for older langgraph / single SKU)
    """
    try:
        from langgraph.types import Send  # noqa: F401
        # Fan-out available → use dispatcher
        return NODE_SKU_DISPATCHER
    except ImportError:
        return NODE_INVENTORY


def _route_after_inventory(state: WorkflowState) -> str:
    dest = route_after_inventory(state)
    return NODE_PROCUREMENT if dest == "procurement" else NODE_FULFILLMENT


def _route_after_procurement(state: WorkflowState) -> str:
    dest = route_after_procurement(state)
    if dest == "hitl_interrupt":
        return END
    return NODE_FULFILLMENT


def _route_after_fulfillment(state: WorkflowState) -> str:
    dest = route_after_fulfillment(state)
    return NODE_EXCEPTION_HANDLER if dest == "exception_handler" else NODE_ORCHESTRATOR_FINALISE


def _route_after_exception(state: WorkflowState) -> str:
    return NODE_ORCHESTRATOR_FINALISE


# ═══════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════

def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> Any:
    """
    Build and compile the Phase 4 Supply Demand Execution StateGraph.

    Topology:
      START → orchestrator_plan → demand_forecast → supply_planning
            → [sku_dispatcher → sku_worker×N → sku_aggregator]  ← parallel fan-out
            → inventory (direct, uses aggregated result)
            → [conditional] → procurement → [conditional] → fulfillment
            → [conditional] → exception_handler → orchestrator_finalise → END

    Args:
        checkpointer: MemorySaver (dev) or SqliteSaver (prod).
                      Pass None to run without state persistence.
    Returns:
        Compiled LangGraph runnable.
    """
    graph = StateGraph(WorkflowState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    graph.add_node(NODE_ORCHESTRATOR_PLAN,     orchestrator_plan)
    graph.add_node(NODE_DEMAND_FORECAST,       demand_forecast_node)
    graph.add_node(NODE_SUPPLY_PLANNING,       supply_planning_node)
    graph.add_node(NODE_SKU_DISPATCHER,        sku_dispatcher)
    graph.add_node(NODE_SKU_WORKER,            sku_worker)
    graph.add_node(NODE_SKU_AGGREGATOR,        sku_aggregator)
    graph.add_node(NODE_INVENTORY,             inventory_node)
    graph.add_node(NODE_PROCUREMENT,           procurement_node)
    graph.add_node(NODE_FULFILLMENT,           fulfillment_node)
    graph.add_node(NODE_EXCEPTION_HANDLER,     exception_handler_node)
    graph.add_node(NODE_ORCHESTRATOR_FINALISE, orchestrator_finalise)

    # ── Linear edges ───────────────────────────────────────────────────────────
    graph.add_edge(START,                   NODE_ORCHESTRATOR_PLAN)
    graph.add_edge(NODE_ORCHESTRATOR_PLAN,  NODE_DEMAND_FORECAST)
    graph.add_edge(NODE_DEMAND_FORECAST,    NODE_SUPPLY_PLANNING)

    # ── Fan-out: supply_planning → dispatcher → workers → aggregator ───────────
    graph.add_edge(NODE_SUPPLY_PLANNING,    NODE_SKU_DISPATCHER)
    graph.add_conditional_edges(
        NODE_SKU_DISPATCHER,
        sku_dispatcher,                     # returns list[Send] objects
        [NODE_SKU_WORKER],                  # possible target nodes
    )
    graph.add_edge(NODE_SKU_WORKER,         NODE_SKU_AGGREGATOR)

    # ── Fan-in: aggregator → inventory (reads aggregated result) ──────────────
    graph.add_edge(NODE_SKU_AGGREGATOR,     NODE_INVENTORY)

    # ── Conditional edges ──────────────────────────────────────────────────────
    graph.add_conditional_edges(
        NODE_INVENTORY,
        _route_after_inventory,
        {NODE_PROCUREMENT: NODE_PROCUREMENT, NODE_FULFILLMENT: NODE_FULFILLMENT},
    )
    graph.add_conditional_edges(
        NODE_PROCUREMENT,
        _route_after_procurement,
        {END: END, NODE_FULFILLMENT: NODE_FULFILLMENT},
    )
    graph.add_conditional_edges(
        NODE_FULFILLMENT,
        _route_after_fulfillment,
        {NODE_EXCEPTION_HANDLER: NODE_EXCEPTION_HANDLER,
         NODE_ORCHESTRATOR_FINALISE: NODE_ORCHESTRATOR_FINALISE},
    )
    graph.add_conditional_edges(
        NODE_EXCEPTION_HANDLER,
        _route_after_exception,
        {NODE_ORCHESTRATOR_FINALISE: NODE_ORCHESTRATOR_FINALISE},
    )
    graph.add_edge(NODE_ORCHESTRATOR_FINALISE, END)

    graph.set_entry_point(NODE_ORCHESTRATOR_PLAN)

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)


# ═══════════════════════════════════════════════════════════════════
# Quick-run helper
# ═══════════════════════════════════════════════════════════════════

def run_workflow(sku_ids: list[str], triggered_by: str = "manual") -> dict[str, Any]:
    """
    Convenience function: build graph with MemorySaver and run synchronously.

    Usage:
        from graph.workflow import run_workflow
        result = run_workflow(["SKU-001", "SKU-002"])
        print(result["summary"])
    """
    from uuid import uuid4
    from graph.checkpointer import get_memory_checkpointer, thread_config
    from graph.state import initial_state

    run_id = f"RUN-{uuid4().hex[:8].upper()}"
    state  = initial_state(run_id=run_id, sku_ids=sku_ids, triggered_by=triggered_by)
    config = thread_config(run_id)

    graph  = build_graph(get_memory_checkpointer())
    return graph.invoke(state, config=config)
