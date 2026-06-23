"""
agents/supply_planning.py  (Phase 3 · Part 2 — ReAct upgrade)
──────────────────────────────────────────────────────────────
Supply Planning Agent — LangGraph node with ReAct tool-calling loop.

Architecture:
  1. LLM receives ForecastResult from state (avg_daily per SKU)
  2. LLM.bind_tools(INVENTORY_TOOLS) → calls tools to gather live data
     - get_stock_level       → current on-hand position
     - compute_eoq           → optimal order quantity
     - calculate_reorder_point → when to trigger procurement
     - score_inventory_health  → overall health before recommending
  3. LLM reasons about urgency and builds supply plan
  4. Structured JSON result written back to WorkflowState

Node name in graph: "supply_planning"
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agents.demand_forecast import _run_react_loop
from config.settings import get_settings
from graph.state import WorkflowState
from tools.inventory_tools import (
    INVENTORY_TOOLS,
    calculate_reorder_point,
    compute_eoq,
    get_stock_level,
    score_inventory_health,
)

settings = get_settings()

SUPPLY_TOOL_MAP: dict[str, Any] = {
    "get_stock_level":          get_stock_level,
    "compute_eoq":              compute_eoq,
    "calculate_reorder_point":  calculate_reorder_point,
    "score_inventory_health":   score_inventory_health,
}


def _get_llm_with_tools() -> Any:
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.0,
        api_key=settings.openai_api_key,
    )
    return llm.bind_tools(INVENTORY_TOOLS)


def _build_system_prompt(sku_ids: list[str]) -> str:
    return f"""You are the Supply Planning Agent in a Supply Demand Execution AI system.

Your job is to determine replenishment requirements for: {sku_ids}

You have access to these tools:
- get_stock_level:          Get current on-hand, reserved, in-transit inventory
- compute_eoq:              Calculate Economic Order Quantity (optimal order size)
- calculate_reorder_point:  Determine at what stock level to trigger a new order
- score_inventory_health:   Get composite 0-1 health score and status

Instructions:
1. For each SKU:
   a. Call get_stock_level to see current position
   b. Call score_inventory_health to assess risk
   c. Call calculate_reorder_point using avg_daily_demand from the forecast
   d. Call compute_eoq to determine optimal order quantity
2. Determine urgency: critical (<3 days supply), high (below safety stock),
   medium (below reorder point), low (healthy)
3. Return a JSON supply plan with this exact structure:

{{
  "sku_plans": [
    {{
      "sku_id": "SKU-XXX",
      "on_hand": <float>,
      "available": <float>,
      "reorder_point": <float>,
      "safety_stock": <float>,
      "eoq": <float>,
      "urgency": "critical|high|medium|low",
      "needs_replenishment": <bool>,
      "recommended_order_qty": <float>,
      "rationale": "<1 sentence reason>"
    }}
  ],
  "replenishment_needed": <bool>,
  "reorder_triggers": ["SKU-XXX", ...],
  "supply_commentary": "<brief ops summary>"
}}"""


def supply_planning_node(state: WorkflowState) -> dict[str, Any]:
    """
    LangGraph node: supply_planning
    ─────────────────────────────────
    ReAct-upgraded supply planning agent. LLM reasons about current
    stock positions and forecast data to build a replenishment plan.
    """
    sku_ids    = state.get("sku_ids", [])
    plan_input = state.get("supply_plan_input") or {}
    forecasts  = plan_input.get("forecasts", [])
    run_id     = state.get("run_id", "UNKNOWN")

    # Build avg_daily lookup from upstream forecasts
    avg_daily_map = {
        f["sku_id"]: f.get("avg_daily_demand", 10.0)
        for f in forecasts
    }

    forecast_summary = "\n".join([
        f"  {f['sku_id']}: avg_daily={f.get('avg_daily_demand', 10.0):.1f}, "
        f"trend={f.get('trend', 'unknown')}, confidence={f.get('confidence', 'medium')}"
        for f in forecasts
    ]) or "  No forecast data available — use defaults."

    try:
        llm_with_tools = _get_llm_with_tools()
        messages = [
            SystemMessage(content=_build_system_prompt(sku_ids)),
            HumanMessage(content=(
                f"Build a supply plan for SKUs: {sku_ids}\n\n"
                f"Demand forecast summary:\n{forecast_summary}\n\n"
                f"Avg daily demand per SKU: {avg_daily_map}\n"
                f"Safety stock target: {settings.default_safety_stock_days} days\n"
                f"Default lead time: {settings.default_lead_time_days} days"
            )),
        ]

        messages, final_content = _run_react_loop(
            llm_with_tools, messages, SUPPLY_TOOL_MAP
        )

        raw = final_content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result_data = json.loads(raw.strip())

    except Exception as exc:
        result_data = _fallback_supply_plan(sku_ids, avg_daily_map)
        result_data["_fallback_reason"] = str(exc)

    # ── Build recommended PO drafts from supply plan ───────────────────────────
    sku_plans        = result_data.get("sku_plans", [])
    reorder_triggers = result_data.get("reorder_triggers", [])
    recommended_pos: list[dict] = []

    for plan in sku_plans:
        if plan.get("needs_replenishment"):
            order_qty = plan.get("recommended_order_qty", plan.get("eoq", 100.0))
            recommended_pos.append({
                "po_number":     f"PO-{uuid4().hex[:8].upper()}",
                "supplier_id":   "SUP-AUTO",   # resolved by procurement agent
                "supplier_name": "",
                "status":        "draft",
                "lines": [{
                    "line_number": 1,
                    "sku_id":      plan["sku_id"],
                    "description": plan["sku_id"],
                    "quantity":    round(order_qty, 1),
                    "unit_cost":   0.0,           # filled in by procurement agent
                }],
                "notes": f"Urgency: {plan.get('urgency')}. {plan.get('rationale', '')}",
                "issued_date": str(datetime.utcnow().date()),
                "created_by_agent": "supply_planning",
            })

    supply_plan_result = {
        "generated_at":       datetime.utcnow().isoformat(),
        "sku_plans":          sku_plans,
        "reorder_triggers":   reorder_triggers,
        "recommended_pos":    recommended_pos,
        "total_pos":          len(recommended_pos),
        "supply_commentary":  result_data.get("supply_commentary", ""),
        "warnings":           [],
    }

    return {
        "supply_plan_result":    supply_plan_result,
        "supply_plan_status":    "success",
        "replenishment_needed":  bool(reorder_triggers),
        "procurement_input": {
            "reorder_triggers": reorder_triggers,
            "recommended_pos":  recommended_pos,
            "sku_plans":        sku_plans,
        },
        "inventory_input": {
            "sku_ids":     sku_ids,
            "location_id": "DC-01",
            "sku_plans":   sku_plans,
        },
    }


def _fallback_supply_plan(
    sku_ids: list[str],
    avg_daily_map: dict[str, float],
) -> dict:
    """Direct tool calls when LLM is unavailable."""
    import math
    import sqlite3
    from pathlib import Path

    DB_PATH = Path("data/supply_demand.db")
    sku_plans: list[dict]  = []
    reorder_triggers: list[str] = []

    for sku_id in sku_ids:
        avg_daily  = avg_daily_map.get(sku_id, 10.0)
        stock      = get_stock_level(sku_id=sku_id)
        on_hand    = stock.get("on_hand", 0.0)
        available  = stock.get("available", 0.0)
        in_transit = stock.get("in_transit", 0.0)

        rop_data   = calculate_reorder_point(
            sku_id=sku_id,
            avg_daily_demand=avg_daily,
            safety_stock_days=settings.default_safety_stock_days,
        )
        eoq_data   = compute_eoq(
            sku_id=sku_id,
            annual_demand=avg_daily * 365,
        )

        rop    = rop_data.get("reorder_point", 50.0)
        ss     = rop_data.get("safety_stock", 20.0)
        eoq    = eoq_data.get("eoq", 100.0)
        dos    = available / avg_daily if avg_daily > 0 else 999.0
        needs  = (on_hand + in_transit) <= rop

        urgency = (
            "critical" if dos < 3  else
            "high"     if available < ss  else
            "medium"   if needs    else
            "low"
        )
        if needs:
            reorder_triggers.append(sku_id)

        sku_plans.append({
            "sku_id":                sku_id,
            "on_hand":               on_hand,
            "available":             available,
            "reorder_point":         rop,
            "safety_stock":          ss,
            "eoq":                   eoq,
            "urgency":               urgency,
            "needs_replenishment":   needs,
            "recommended_order_qty": eoq,
            "rationale":             f"On-hand {on_hand:.0f} vs reorder point {rop:.0f}.",
        })

    return {
        "sku_plans":           sku_plans,
        "replenishment_needed": bool(reorder_triggers),
        "reorder_triggers":    reorder_triggers,
        "supply_commentary":   (
            f"Fallback supply plan. {len(reorder_triggers)}/{len(sku_ids)} "
            "SKUs need replenishment."
        ),
    }
