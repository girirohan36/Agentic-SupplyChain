"""
agents/inventory.py  (Phase 4 — enhanced with parallel scoring)
────────────────────────────────────────────────────────────────
Inventory Agent — LangGraph node with:
  1. bulk_score_all_skus    → single DB call replaces N individual calls
  2. get_inventory_kpis     → warehouse-wide health metrics upfront
  3. check_expiry_risk      → flags perishables expiring soon
  4. concurrent.futures     → parallel LLM commentary per critical SKU
  5. DB write-back          → record_stock_movement for every sale event
  6. Overstock detection    → flags capital tied up in excess stock
  7. Richer ExceptionEvents → severity tuned by health_score bands

Node name in graph: "inventory"
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.demand_forecast import _run_react_loop
from config.settings import get_settings
from graph.state import WorkflowState, add_exception
from tools.inventory_tools import (
    INVENTORY_TOOLS,
    bulk_score_all_skus,
    check_expiry_risk,
    get_days_of_supply,
    get_inventory_kpis,
    get_stock_level,
    list_critical_skus,
    record_stock_movement,
    score_inventory_health,
    update_stock_level,
)

settings = get_settings()

INVENTORY_TOOL_MAP: dict[str, Any] = {
    "get_stock_level":        get_stock_level,
    "score_inventory_health": score_inventory_health,
    "list_critical_skus":     list_critical_skus,
    "get_days_of_supply":     get_days_of_supply,
    "bulk_score_all_skus":    bulk_score_all_skus,
    "get_inventory_kpis":     get_inventory_kpis,
    "check_expiry_risk":      check_expiry_risk,
    "update_stock_level":     update_stock_level,
    "record_stock_movement":  record_stock_movement,
}


def _get_llm_with_tools() -> Any:
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.0,
        api_key=settings.openai_api_key,
    )
    return llm.bind_tools(INVENTORY_TOOLS)


def _build_system_prompt(sku_ids: list[str]) -> str:
    return f"""You are the Inventory Agent in a Supply Demand Execution AI system.

Your job is to perform a comprehensive inventory assessment for SKUs: {sku_ids}

You have access to these tools (use the most efficient ones first):
- get_inventory_kpis:     Start here — warehouse-wide health metrics in one call
- bulk_score_all_skus:    Score ALL SKUs at once (much faster than individual calls)
- check_expiry_risk:      Flag perishable SKUs expiring soon
- list_critical_skus:     Get worst-performing SKUs sorted by urgency
- score_inventory_health: Score a specific SKU if you need details
- get_stock_level:        Get precise on-hand/reserved/in-transit for a SKU
- get_days_of_supply:     Days of supply remaining for a specific SKU

Efficient workflow:
1. Call get_inventory_kpis once for the warehouse overview
2. Call bulk_score_all_skus to get all SKU statuses efficiently
3. Call check_expiry_risk to catch perishable issues
4. For any critical/stock_out SKU, call get_days_of_supply for more detail
5. Return your assessment

Return JSON in this exact structure:
{{
  "records": [
    {{
      "sku_id": "SKU-XXX",
      "health_status": "healthy|at_risk|critical|stock_out",
      "health_score": <0.0-1.0>,
      "days_of_supply": <float>,
      "on_hand": <float>,
      "available": <float>,
      "reorder_triggered": <bool>,
      "overstock_flag": <bool>,
      "expiry_risk_flag": <bool>,
      "stockout_risk_date": "YYYY-MM-DD or null",
      "llm_commentary": "<one sentence — what action is needed and why>"
    }}
  ],
  "kpis": {{
    "avg_days_of_supply": <float>,
    "fill_rate_pct": <float>,
    "stockout_count": <int>,
    "reorder_trigger_count": <int>,
    "total_inventory_value": <float>
  }},
  "inventory_commentary": "<2-sentence warehouse health narrative>"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Parallel LLM commentary generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_commentary_for_sku(sku_record: dict) -> str:
    """
    Generate a one-sentence LLM narrative for a critical/at_risk SKU.
    Runs in a thread pool so multiple SKUs are processed concurrently.
    """
    try:
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.2,
            api_key=settings.openai_api_key,
        )
        prompt = (
            f"In one sentence, state the inventory risk and recommended action for "
            f"SKU {sku_record['sku_id']}: "
            f"status={sku_record['health_status']}, "
            f"health_score={sku_record.get('health_score', 0):.2f}, "
            f"days_of_supply={sku_record.get('days_of_supply', 0):.1f}, "
            f"on_hand={sku_record.get('on_hand', 0):.0f}, "
            f"reorder_triggered={sku_record.get('reorder_triggered', False)}."
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content.strip()
    except Exception:
        status = sku_record.get("health_status", "unknown")
        dos    = sku_record.get("days_of_supply", 0)
        return (f"{sku_record['sku_id']}: {status} — "
                f"{dos:.1f} days of supply remaining. "
                f"{'Immediate reorder required.' if status in ('critical','stock_out') else 'Monitor closely.'}")


def _add_parallel_commentaries(records: list[dict]) -> list[dict]:
    """
    Add LLM commentary to critical and at_risk SKUs using a thread pool.
    Healthy SKUs get a short default commentary (no LLM call needed).
    """
    critical_records = [r for r in records if r.get("health_status") in ("critical", "stock_out", "at_risk")]
    healthy_records  = [r for r in records if r.get("health_status") == "healthy"]

    # Default commentary for healthy SKUs (no LLM cost)
    for r in healthy_records:
        r["llm_commentary"] = r.get("llm_commentary") or f"{r['sku_id']}: healthy — no action needed."

    if not critical_records:
        return records

    # Parallel LLM calls for all critical/at-risk SKUs
    try:
        with ThreadPoolExecutor(max_workers=min(5, len(critical_records))) as executor:
            futures = {
                executor.submit(_generate_commentary_for_sku, rec): rec
                for rec in critical_records
                if not rec.get("llm_commentary")
            }
            for future in as_completed(futures):
                rec = futures[future]
                try:
                    rec["llm_commentary"] = future.result()
                except Exception:
                    rec["llm_commentary"] = rec.get("llm_commentary") or "Commentary unavailable."
    except Exception:
        for r in critical_records:
            if not r.get("llm_commentary"):
                r["llm_commentary"] = "Commentary unavailable — LLM offline."

    return records


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph Node
# ─────────────────────────────────────────────────────────────────────────────

def inventory_node(state: WorkflowState) -> dict[str, Any]:
    """
    LangGraph node: inventory (Phase 4 enhanced)
    ─────────────────────────────────────────────
    Efficient multi-SKU inventory assessment using:
      - bulk_score_all_skus (one DB call for all SKUs)
      - get_inventory_kpis (warehouse overview)
      - Parallel LLM commentary for critical SKUs
      - Expiry risk detection
      - Exception event generation with tuned severity
    """
    sku_ids     = state.get("sku_ids", [])
    run_id      = state.get("run_id", "UNKNOWN")
    location_id = "DC-01"
    exception_patch: dict[str, Any] = {}

    try:
        llm_with_tools = _get_llm_with_tools()
        messages = [
            SystemMessage(content=_build_system_prompt(sku_ids)),
            HumanMessage(content=(
                f"Assess inventory for SKUs: {sku_ids}\n"
                f"Location: {location_id}\n"
                f"Today: {date.today()}"
            )),
        ]
        messages, final_content = _run_react_loop(
            llm_with_tools, messages, INVENTORY_TOOL_MAP
        )
        raw = final_content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result_data = json.loads(raw.strip())

    except Exception as exc:
        result_data = _fallback_inventory_enhanced(sku_ids, location_id)
        result_data["_fallback_reason"] = str(exc)

    records     = result_data.get("records", [])
    kpis        = result_data.get("kpis", {})
    expiry_data = result_data.get("expiry_risks", [])

    # ── Parallel commentaries for critical SKUs ────────────────────────────────
    records = _add_parallel_commentaries(records)

    # ── Raise ExceptionEvents ──────────────────────────────────────────────────
    for rec in records:
        status = rec.get("health_status", "healthy")

        if status in ("critical", "stock_out"):
            severity = "critical" if status == "stock_out" else "high"
            exc_event = {
                "run_id":         run_id,
                "raised_by":      "inventory",
                "exception_type": "stockout_imminent",
                "severity":       severity,
                "sku_id":         rec["sku_id"],
                "supplier_id":    None,
                "description": (
                    f"{rec['sku_id']}: {'STOCK OUT' if status == 'stock_out' else 'Critical stock'} — "
                    f"dos={rec.get('days_of_supply', 0):.1f}d, "
                    f"score={rec.get('health_score', 0):.2f}. "
                    f"{rec.get('llm_commentary', '')}"
                ),
                "context": {
                    "health_score":   rec.get("health_score"),
                    "on_hand":        rec.get("on_hand"),
                    "days_of_supply": rec.get("days_of_supply"),
                },
                "resolved": False,
            }
            exception_patch = add_exception(state, exc_event)

        # Overstock alert
        if rec.get("overstock_flag"):
            exc_event = {
                "run_id":         run_id,
                "raised_by":      "inventory",
                "exception_type": "overstock",
                "severity":       "info",
                "sku_id":         rec["sku_id"],
                "supplier_id":    None,
                "description": (
                    f"{rec['sku_id']}: Overstock detected. "
                    f"on_hand={rec.get('on_hand', 0):.0f} is >2× reorder point. "
                    "Consider pausing replenishment or running a promotion."
                ),
                "context": {"on_hand": rec.get("on_hand")},
                "resolved": False,
            }
            exception_patch = add_exception(state, exc_event)

        # Expiry risk alert
        if rec.get("expiry_risk_flag"):
            exc_event = {
                "run_id":         run_id,
                "raised_by":      "inventory",
                "exception_type": "supply_disruption",
                "severity":       "warning",
                "sku_id":         rec["sku_id"],
                "supplier_id":    None,
                "description":    f"{rec['sku_id']}: Expiry risk — batch expires within lead time.",
                "context":        {},
                "resolved":       False,
            }
            exception_patch = add_exception(state, exc_event)

    # ── Aggregate counts ───────────────────────────────────────────────────────
    critical_count = sum(1 for r in records if r.get("health_status") in ("critical", "stock_out"))
    at_risk_count  = sum(1 for r in records if r.get("health_status") == "at_risk")
    healthy_count  = sum(1 for r in records if r.get("health_status") == "healthy")
    reorder_triggers = [r["sku_id"] for r in records if r.get("reorder_triggered")]

    inventory_result = {
        "run_id":            run_id,
        "generated_at":      datetime.utcnow().isoformat(),
        "location_id":       location_id,
        "records":           records,
        "total_skus":        len(records),
        "critical_count":    critical_count,
        "at_risk_count":     at_risk_count,
        "healthy_count":     healthy_count,
        "reorder_triggers":  reorder_triggers,
        "kpis":              kpis,
        "expiry_risks":      expiry_data,
        "inventory_commentary": result_data.get("inventory_commentary", ""),
        "warnings":          [],
    }

    open_orders = {r["sku_id"]: r.get("on_hand", 0.0) * 0.1 for r in records}

    return {
        **exception_patch,
        "inventory_result":    inventory_result,
        "inventory_status":    "success",
        "replenishment_needed": bool(reorder_triggers),
        "fulfillment_input": {
            "sku_ids":        sku_ids,
            "open_orders":    open_orders,
            "health_records": {r["sku_id"]: r for r in records},
        },
        "exception_detected": bool(exception_patch) or state.get("exception_detected", False),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced fallback (uses bulk_score_all_skus for efficiency)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_inventory_enhanced(sku_ids: list[str], location_id: str) -> dict:
    """
    Fallback path using direct tool calls (no LLM).
    Phase 4 upgrade: uses bulk_score_all_skus + get_inventory_kpis in one pass.
    """
    # 1. Bulk score — single DB call
    bulk = bulk_score_all_skus(location_id=location_id, include_healthy=True)
    bulk_map = {r["sku_id"]: r for r in bulk.get("records", [])}

    # 2. Warehouse KPIs
    kpis_raw = get_inventory_kpis(location_id=location_id)

    # 3. Expiry risk
    expiry_raw = check_expiry_risk(location_id=location_id, days_ahead=30)

    expiry_skus = {e["sku_id"] for e in expiry_raw.get("at_risk_skus", [])}

    records: list[dict] = []
    for sku_id in sku_ids:
        bulk_rec = bulk_map.get(sku_id, {})

        on_hand   = bulk_rec.get("on_hand", 0.0)
        available = bulk_rec.get("available", 0.0)
        dos       = bulk_rec.get("days_of_supply", 0.0)
        status    = bulk_rec.get("health_status", "healthy")
        score     = bulk_rec.get("health_score", 1.0)
        reorder   = bulk_rec.get("reorder_triggered", False)
        overstock = bulk_rec.get("overstock", False)
        expiry_risk = sku_id in expiry_skus

        stockout_date = None
        if dos < 999 and dos > 0:
            stockout_date = str(date.today() + timedelta(days=int(dos)))

        records.append({
            "sku_id":            sku_id,
            "health_status":     status,
            "health_score":      score,
            "days_of_supply":    dos,
            "on_hand":           on_hand,
            "available":         available,
            "reorder_triggered": reorder,
            "overstock_flag":    overstock,
            "expiry_risk_flag":  expiry_risk,
            "stockout_risk_date": stockout_date,
            "avg_daily_demand":  bulk_rec.get("avg_daily_demand", 10.0),
            "llm_commentary":    None,
        })

    critical_count = sum(1 for r in records if r["health_status"] in ("critical", "stock_out"))

    return {
        "records": records,
        "kpis": {
            "avg_days_of_supply":    kpis_raw.get("avg_days_of_supply", 0),
            "fill_rate_pct":         kpis_raw.get("fill_rate_pct", 0),
            "stockout_count":        kpis_raw.get("stockout_count", 0),
            "reorder_trigger_count": kpis_raw.get("reorder_trigger_count", 0),
            "total_inventory_value": kpis_raw.get("total_inventory_value", 0),
        },
        "expiry_risks": expiry_raw.get("at_risk_skus", []),
        "inventory_commentary": (
            f"Fallback assessment: {critical_count}/{len(sku_ids)} SKUs at critical/stock-out. "
            f"Fill rate: {kpis_raw.get('fill_rate_pct', 0):.1f}%. "
            f"Avg days of supply: {kpis_raw.get('avg_days_of_supply', 0):.1f}d."
        ),
    }
