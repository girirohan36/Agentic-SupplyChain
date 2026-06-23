"""
graph/state.py
──────────────
LangGraph shared state — the single source of truth that flows
through every node in the multi-agent graph.

Design principles:
  1. ALL agents read from and write to this TypedDict.
  2. Fields are Optional so nodes only populate what they own.
  3. Immutable-by-convention: each node returns a *partial* dict
     update; LangGraph merges it (reducer pattern).
  4. Serialisable: all nested objects are plain dicts (not Pydantic
     models) so the built-in MemorySaver checkpointer can pickle them.
     We convert Pydantic → dict on write, dict → Pydantic on read
     using the helper functions at the bottom of this file.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# ─────────────────────────────────────────────────────────────────────────────
# WorkflowState — the LangGraph TypedDict
# ─────────────────────────────────────────────────────────────────────────────

class WorkflowState(TypedDict, total=False):
    """
    Shared state passed between every node in the supply-demand graph.

    Naming convention:
      - *_input  → data fed INTO an agent
      - *_result → data produced BY an agent (dict-serialised Pydantic model)
      - messages → LangChain message history (tool calls, LLM responses)
      - flags    → boolean control signals for conditional edges
    """

    # ── Run metadata ──────────────────────────────────────────────────────────
    run_id:        str           # unique run identifier e.g. "RUN-A1B2C3D4"
    triggered_by:  str           # "api" | "scheduler" | "manual" | "test"
    sku_ids:       list[str]     # SKUs in scope for this run
    current_sku:   str           # SKU currently being processed (loop control)

    # ── LangChain message thread (add_messages reducer auto-appends) ──────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Orchestrator ──────────────────────────────────────────────────────────
    orchestrator_plan:    Optional[dict[str, Any]]  # {agent: action, priority, rationale}
    workflow_status:      str                        # mirrors WorkflowStatus enum value

    # ── Demand Forecast Agent ─────────────────────────────────────────────────
    demand_forecast_input:   Optional[dict[str, Any]]   # serialised ForecastRequest
    demand_forecast_result:  Optional[dict[str, Any]]   # serialised ForecastResult
    demand_forecast_status:  Optional[str]               # AgentStatus value

    # ── Supply Planning Agent ─────────────────────────────────────────────────
    supply_plan_input:    Optional[dict[str, Any]]   # serialised EOQInput
    supply_plan_result:   Optional[dict[str, Any]]   # serialised SupplyPlanResult
    supply_plan_status:   Optional[str]

    # ── Inventory Agent ───────────────────────────────────────────────────────
    inventory_input:   Optional[dict[str, Any]]   # {sku_ids, location_id}
    inventory_result:  Optional[dict[str, Any]]   # serialised InventorySnapshot
    inventory_status:  Optional[str]

    # ── Procurement Agent ─────────────────────────────────────────────────────
    procurement_input:   Optional[dict[str, Any]]   # {reorder_triggers, recommended_pos}
    procurement_result:  Optional[dict[str, Any]]   # {issued_pos: [...]}
    procurement_status:  Optional[str]

    # ── Fulfillment Agent ─────────────────────────────────────────────────────
    fulfillment_input:   Optional[dict[str, Any]]   # {open_orders, available_stock}
    fulfillment_result:  Optional[dict[str, Any]]   # {routed_orders, dispatch_list}
    fulfillment_status:  Optional[str]

    # ── Exception Agent ───────────────────────────────────────────────────────
    exception_events:    list[dict[str, Any]]   # list of serialised ExceptionEvent
    exception_result:    Optional[dict[str, Any]]  # {resolved, escalated, notifications}
    exception_status:    Optional[str]

    # ── Human-in-the-Loop ─────────────────────────────────────────────────────
    hitl_required:      bool                    # True → graph pauses at __interrupt__
    hitl_checkpoint:    Optional[dict[str, Any]]  # serialised HITLCheckpoint
    hitl_response:      Optional[dict[str, Any]]  # human's response payload
    hitl_approved:      Optional[bool]            # True=approved, False=rejected

    # ── Control flags (used by conditional edges) ─────────────────────────────
    replenishment_needed:  bool   # True → route to Procurement after Supply Planning
    exception_detected:    bool   # True → route to Exception Agent
    fulfillment_ready:     bool   # True → route to Fulfillment Agent
    run_complete:          bool   # True → route to END node

    # ── Phase 4 — Per-SKU parallel processing accumulator ────────────────────
    # Each sku_worker node writes a partial result here.
    # The sku_aggregator fan-in node merges them into inventory_result.
    per_sku_results: list[dict[str, Any]]   # accumulated by add_messages reducer
    sku_processing_complete: bool            # True when all sku_worker nodes done

    # ── Aggregated outputs ────────────────────────────────────────────────────
    # Populated by Orchestrator at end of run — surfaced by API
    summary: Optional[dict[str, Any]]   # {skus_processed, pos_issued, exceptions, duration}


# ─────────────────────────────────────────────────────────────────────────────
# Default state factory
# ─────────────────────────────────────────────────────────────────────────────

def initial_state(
    run_id:       str,
    sku_ids:      list[str],
    triggered_by: str = "api",
) -> WorkflowState:
    """
    Build the initial WorkflowState for a new run.
    Pass this to graph.invoke() or graph.astream().

    Usage:
        from graph.state import initial_state
        state = initial_state(run_id="RUN-ABC123", sku_ids=["SKU-001", "SKU-002"])
        result = await graph.ainvoke(state, config={"configurable": {"thread_id": run_id}})
    """
    return WorkflowState(
        run_id=run_id,
        triggered_by=triggered_by,
        sku_ids=[s.upper().strip() for s in sku_ids],
        current_sku=sku_ids[0].upper().strip() if sku_ids else "",
        messages=[],
        workflow_status="initialised",

        # Agent I/O — all None until each agent runs
        orchestrator_plan=None,
        demand_forecast_input=None,
        demand_forecast_result=None,
        demand_forecast_status="pending",
        supply_plan_input=None,
        supply_plan_result=None,
        supply_plan_status="pending",
        inventory_input=None,
        inventory_result=None,
        inventory_status="pending",
        procurement_input=None,
        procurement_result=None,
        procurement_status="pending",
        fulfillment_input=None,
        fulfillment_result=None,
        fulfillment_status="pending",
        exception_events=[],
        exception_result=None,
        exception_status="pending",

        # HITL
        hitl_required=False,
        hitl_checkpoint=None,
        hitl_response=None,
        hitl_approved=None,

        # Phase 4 parallel accumulator
        per_sku_results=[],
        sku_processing_complete=False,

        # Control flags
        replenishment_needed=False,
        exception_detected=False,
        fulfillment_ready=False,
        run_complete=False,

        summary=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# State helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_agent_status(state: WorkflowState, agent: str) -> str:
    """
    Return the current status string for a named agent.

    Args:
        state: current WorkflowState
        agent: one of 'demand_forecast' | 'supply_plan' | 'inventory' |
               'procurement' | 'fulfillment' | 'exception'
    Returns:
        AgentStatus value string, or 'unknown'
    """
    key = f"{agent}_status"
    return state.get(key, "unknown")  # type: ignore[arg-type]


def add_exception(
    state: WorkflowState,
    event: dict[str, Any],
) -> dict[str, Any]:
    """
    Return a state patch that appends a new exception event.

    Usage inside a node:
        patch = add_exception(state, exception_event.model_dump())
        return {**patch, "exception_detected": True}
    """
    existing = list(state.get("exception_events") or [])
    existing.append(event)
    return {"exception_events": existing, "exception_detected": True}


def get_forecast_result(state: WorkflowState) -> Optional[dict[str, Any]]:
    """Safely retrieve the demand forecast result dict."""
    return state.get("demand_forecast_result")


def get_inventory_result(state: WorkflowState) -> Optional[dict[str, Any]]:
    """Safely retrieve the inventory snapshot dict."""
    return state.get("inventory_result")


def get_supply_plan_result(state: WorkflowState) -> Optional[dict[str, Any]]:
    """Safely retrieve the supply plan result dict."""
    return state.get("supply_plan_result")


def is_all_agents_done(state: WorkflowState) -> bool:
    """
    True when every core agent has a terminal status
    (success | failed | skipped).
    Used by the Orchestrator to decide if the run is complete.
    """
    terminal = {"success", "failed", "skipped"}
    core_agents = [
        "demand_forecast_status",
        "supply_plan_status",
        "inventory_status",
        "procurement_status",
        "fulfillment_status",
    ]
    return all(state.get(k, "pending") in terminal for k in core_agents)  # type: ignore[arg-type]
