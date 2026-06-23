"""
agents/fulfillment.py  (Phase 5 — enhanced)
─────────────────────────────────────────────
Fulfillment Agent — LangGraph node with:
  1. Priority-sorted dispatch (critical stockouts first)
  2. route_to_best_dc for multi-location routing
  3. create_dispatch persists every shipment to DB
  4. manage_backorder for held orders with ETA
  5. calculate_fill_rate KPI surfaced in result
  6. Low fill rate (<80%) triggers an ExceptionEvent
  7. ReAct loop with FULFILLMENT_TOOLS + fallback

Node name in graph: "fulfillment"
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.demand_forecast import _run_react_loop
from config.settings import get_settings
from graph.state import WorkflowState, add_exception
from tools.fulfillment_tools import (
    FULFILLMENT_TOOLS,
    calculate_fill_rate,
    create_dispatch,
    get_open_orders,
    manage_backorder,
    route_to_best_dc,
    score_order_priority,
)

settings = get_settings()

FULFILLMENT_TOOL_MAP: dict[str, Any] = {
    "get_open_orders":       get_open_orders,
    "score_order_priority":  score_order_priority,
    "route_to_best_dc":      route_to_best_dc,
    "create_dispatch":       create_dispatch,
    "manage_backorder":      manage_backorder,
    "calculate_fill_rate":   calculate_fill_rate,
}

FILL_RATE_THRESHOLD = 80.0   # below this → exception event


def _get_llm_with_tools() -> Any:
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.0,
        api_key=settings.openai_api_key,
    )
    return llm.bind_tools(FULFILLMENT_TOOLS)


def _build_system_prompt(sku_ids: list[str], run_id: str) -> str:
    return f"""You are the Fulfillment Agent in a Supply Demand Execution AI system.

Your job is to fulfil open customer orders for SKUs: {sku_ids}
Run ID: {run_id}

You have access to these tools:
- get_open_orders:       Load recent demand as open orders needing fulfillment
- score_order_priority:  Score 0-100 priority for an order (higher = fulfil first)
- route_to_best_dc:      Select optimal DC for a SKU order
- create_dispatch:       Record a fulfilled dispatch (decrements stock)
- manage_backorder:      Record an unfulfillable order with ETA
- calculate_fill_rate:   Compute % of demand fulfilled this run

Fulfillment strategy:
1. Call get_open_orders to see what needs to be fulfilled
2. For each SKU, call score_order_priority to get a priority score
3. Sort by priority (highest first) — critical stockouts go first
4. For each order, call route_to_best_dc to find available stock
5. If stock available: call create_dispatch to ship
6. If stock insufficient: call manage_backorder with ETA
7. At the end, call calculate_fill_rate for the run KPI

Return JSON:
{{
  "routed_orders": [
    {{"sku_id": "SKU-XXX", "qty_needed": 100, "qty_fulfilled": 100,
      "status": "fulfilled|partial|held", "dc": "DC-01",
      "priority_score": 75, "dispatch_id": "DISP-XXXXXXXX"}}
  ],
  "held_orders": [
    {{"sku_id": "SKU-XXX", "qty_held": 50, "reason": "...", "eta_date": "YYYY-MM-DD"}}
  ],
  "dispatch_list": [{{"sku_id": "SKU-XXX", "qty": 100, "dc": "DC-01", "date": "YYYY-MM-DD"}}],
  "fill_rate_pct": <float>,
  "fill_rate_grade": "excellent|good|acceptable|poor",
  "fulfillment_commentary": "<2-sentence summary>"
}}"""


def fulfillment_node(state: WorkflowState) -> dict[str, Any]:
    """
    LangGraph node: fulfillment (Phase 5 enhanced)
    ────────────────────────────────────────────────
    Priority-sorted, multi-DC fulfillment with DB write-back,
    fill rate tracking, and low fill rate exception generation.
    """
    ff_input      = state.get("fulfillment_input") or {}
    sku_ids       = ff_input.get("sku_ids") or state.get("sku_ids", [])
    run_id        = state.get("run_id", "UNKNOWN")
    exception_patch: dict[str, Any] = {}

    # ── HITL rejection guard ───────────────────────────────────────────────────
    if state.get("hitl_required") and state.get("hitl_approved") is False:
        return {
            "fulfillment_result": {
                "note":          "Fulfillment skipped — HITL rejected PO approval.",
                "routed_orders": [], "held_orders": [], "dispatch_list": [],
                "fill_rate_pct": 0.0, "fill_rate_grade": "poor",
            },
            "fulfillment_status": "skipped",
        }

    try:
        llm_with_tools = _get_llm_with_tools()
        messages = [
            SystemMessage(content=_build_system_prompt(sku_ids, run_id)),
            HumanMessage(content=(
                f"Fulfil orders for SKUs: {sku_ids}\n"
                f"Health records available: {list((ff_input.get('health_records') or {}).keys())}\n"
                f"Today: {datetime.utcnow().date()}"
            )),
        ]
        messages, final_content = _run_react_loop(
            llm_with_tools, messages, FULFILLMENT_TOOL_MAP
        )
        raw = final_content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result_data = json.loads(raw.strip())

    except Exception as exc:
        result_data = _fallback_fulfillment(state, sku_ids, run_id)
        result_data["_fallback_reason"] = str(exc)

    routed_orders = result_data.get("routed_orders", [])
    held_orders   = result_data.get("held_orders",   [])
    dispatch_list = result_data.get("dispatch_list", [])
    fill_rate     = result_data.get("fill_rate_pct", 100.0)
    fill_grade    = result_data.get("fill_rate_grade", "good")

    # ── Low fill rate exception ────────────────────────────────────────────────
    if fill_rate < FILL_RATE_THRESHOLD:
        exc_event = {
            "run_id":         run_id,
            "raised_by":      "fulfillment",
            "exception_type": "supply_disruption",
            "severity":       "high" if fill_rate < 50 else "warning",
            "sku_id":         None,
            "supplier_id":    None,
            "description": (
                f"Low fill rate: {fill_rate:.1f}% (threshold {FILL_RATE_THRESHOLD:.0f}%). "
                f"{len(held_orders)} orders on backorder. "
                f"Grade: {fill_grade}."
            ),
            "context": {
                "fill_rate_pct":  fill_rate,
                "held_orders":    len(held_orders),
                "routed_orders":  len(routed_orders),
            },
            "resolved": False,
        }
        exception_patch = add_exception(state, exc_event)

    # ── Exception for individual held orders ──────────────────────────────────
    for held in held_orders:
        exc_event = {
            "run_id":         run_id,
            "raised_by":      "fulfillment",
            "exception_type": "stockout_imminent",
            "severity":       "warning",
            "sku_id":         held.get("sku_id"),
            "supplier_id":    None,
            "description": (
                f"{held.get('sku_id')}: {held.get('qty_held', 0):.0f} units on backorder. "
                f"Reason: {held.get('reason', 'Insufficient stock')}. "
                f"ETA: {held.get('eta_date', 'Unknown')}."
            ),
            "context": held,
            "resolved": False,
        }
        exception_patch = add_exception(state, exc_event)

    fulfillment_result = {
        "routed_orders":         routed_orders,
        "held_orders":           held_orders,
        "dispatch_list":         dispatch_list,
        "total_fulfilled":       len([o for o in routed_orders if o.get("status") == "fulfilled"]),
        "total_partial":         len([o for o in routed_orders if o.get("status") == "partial"]),
        "total_held":            len(held_orders),
        "fill_rate_pct":         fill_rate,
        "fill_rate_grade":       fill_grade,
        "fulfillment_commentary": result_data.get("fulfillment_commentary", ""),
        "generated_at":          datetime.utcnow().isoformat(),
    }

    has_exception = bool(held_orders) or bool(exception_patch) or state.get("exception_detected", False)

    return {
        **exception_patch,
        "fulfillment_result":  fulfillment_result,
        "fulfillment_status":  "success",
        "exception_detected":  has_exception,
    }


def _fallback_fulfillment(
    state:   WorkflowState,
    sku_ids: list[str],
    run_id:  str,
) -> dict:
    """
    Direct tool calls when LLM is unavailable.
    Phase 5: uses score_order_priority, route_to_best_dc, create_dispatch, manage_backorder.
    """
    ff_input     = state.get("fulfillment_input") or {}
    health_map   = ff_input.get("health_records", {})
    open_orders  = ff_input.get("open_orders", {})

    # If no open_orders from upstream, load from DB
    if not open_orders:
        orders_data = get_open_orders(sku_ids=sku_ids, days_back=7)
        for order in orders_data.get("orders", []):
            open_orders[order["sku_id"]] = order["total_qty"]

    routed_orders: list[dict] = []
    held_orders:   list[dict] = []
    dispatch_list: list[dict] = []
    total_demanded  = 0.0
    total_dispatched = 0.0

    # Score and sort: highest priority first
    scored: list[dict] = []
    for sku_id in sku_ids:
        qty_needed    = open_orders.get(sku_id, 0.0)
        if qty_needed <= 0:
            continue
        health_rec    = health_map.get(sku_id, {})
        health_status = health_rec.get("health_status", "healthy")
        dos           = health_rec.get("days_of_supply", 30.0)
        priority      = score_order_priority(
            sku_id=sku_id, qty_needed=qty_needed,
            health_status=health_status, days_of_supply=dos,
        )
        scored.append({
            "sku_id":         sku_id,
            "qty_needed":     qty_needed,
            "priority_score": priority.get("priority_score", 0),
            "health_status":  health_status,
        })

    scored.sort(key=lambda x: x["priority_score"], reverse=True)

    for item in scored:
        sku_id    = item["sku_id"]
        qty_needed = item["qty_needed"]
        total_demanded += qty_needed

        # Route to best DC
        route = route_to_best_dc(sku_id=sku_id, qty_needed=qty_needed)

        if not route.get("routable"):
            # No stock anywhere
            bo = manage_backorder(
                sku_id=sku_id, qty_held=qty_needed,
                reason="No stock at any location",
            )
            held_orders.append({
                "sku_id":   sku_id,
                "qty_held": qty_needed,
                "reason":   "No stock at any location",
                "eta_date": bo.get("eta_date"),
            })
            continue

        dc        = route.get("dc", "DC-01")
        available = route.get("available", 0.0)
        qty_disp  = min(available, qty_needed)

        if qty_disp > 0:
            # Create dispatch
            disp = create_dispatch(
                sku_id=sku_id, location_id=dc,
                qty_dispatched=qty_disp, channel="online",
                reference_doc=f"RUN-{run_id}",
            )
            dispatch_id = disp.get("dispatch_id", "DISP-UNKNOWN")
            total_dispatched += qty_disp
            dispatch_list.append({
                "sku_id":    sku_id,
                "qty":       qty_disp,
                "dc":        dc,
                "date":      str(datetime.utcnow().date()),
            })
        else:
            dispatch_id = None

        shortfall = qty_needed - qty_disp
        status    = "fulfilled" if shortfall <= 0 else ("partial" if qty_disp > 0 else "held")

        routed_orders.append({
            "sku_id":        sku_id,
            "qty_needed":    qty_needed,
            "qty_fulfilled": qty_disp,
            "status":        status,
            "dc":            dc,
            "priority_score": item["priority_score"],
            "dispatch_id":   dispatch_id,
        })

        if shortfall > 0:
            bo = manage_backorder(
                sku_id=sku_id, qty_held=shortfall,
                reason=f"Partial stock — {shortfall:.0f} units short",
            )
            held_orders.append({
                "sku_id":   sku_id,
                "qty_held": shortfall,
                "reason":   f"Partial stock — {shortfall:.0f} units short",
                "eta_date": bo.get("eta_date"),
            })

    # Fill rate
    fill_rate = round(total_dispatched / total_demanded * 100, 1) if total_demanded > 0 else 100.0
    fill_grade = ("excellent"  if fill_rate >= 95 else
                  "good"       if fill_rate >= 85 else
                  "acceptable" if fill_rate >= 70 else "poor")

    return {
        "routed_orders": routed_orders,
        "held_orders":   held_orders,
        "dispatch_list": dispatch_list,
        "fill_rate_pct": fill_rate,
        "fill_rate_grade": fill_grade,
        "fulfillment_commentary": (
            f"Fallback fulfillment: {len(dispatch_list)} dispatches, "
            f"{len(held_orders)} backorders. Fill rate: {fill_rate:.1f}% ({fill_grade})."
        ),
    }
