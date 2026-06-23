"""
agents/procurement.py  (Phase 4 — enhanced with multi-supplier logic)
───────────────────────────────────────────────────────────────────────
Procurement Agent — LangGraph node upgraded with:
  1. compare_suppliers     → full vendor comparison matrix before deciding
  2. calculate_split_order → splits critical orders across 2 suppliers
  3. Contract vs spot pricing → uses contracted_cost if available
  4. Urgency-based routing  → critical = split order, others = single PO
  5. Richer HITL context    → surfaces supplier comparison in approval request

Node name in graph: "procurement"
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.demand_forecast import _run_react_loop
from config.settings import get_settings
from graph.checkpointer import build_interrupt_payload
from graph.state import WorkflowState
from tools.notification_tools import NOTIFICATION_TOOLS, send_alert
from tools.procurement_tools import (
    PROCUREMENT_TOOLS,
    calculate_split_order,
    compare_suppliers,
    find_best_supplier,
    generate_purchase_order,
    get_po_status,
    list_open_pos,
    submit_po,
)

settings = get_settings()

ALL_PROCUREMENT_TOOLS = PROCUREMENT_TOOLS + NOTIFICATION_TOOLS
HITL_VALUE_THRESHOLD  = 5_000.0

PROCUREMENT_TOOL_MAP: dict[str, Any] = {
    "find_best_supplier":      find_best_supplier,
    "compare_suppliers":       compare_suppliers,
    "calculate_split_order":   calculate_split_order,
    "generate_purchase_order": generate_purchase_order,
    "submit_po":               submit_po,
    "get_po_status":           get_po_status,
    "list_open_pos":           list_open_pos,
    "send_alert":              send_alert,
}


def _get_llm_with_tools() -> Any:
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.0,
        api_key=settings.openai_api_key,
    )
    return llm.bind_tools(ALL_PROCUREMENT_TOOLS)


def _build_system_prompt(hitl_enabled: bool) -> str:
    hitl_note = (
        "IMPORTANT: Do NOT submit POs with total_value ≥ $5,000 — leave as draft and "
        "mark hitl_required=true in your result."
        if hitl_enabled else
        "You may submit all POs immediately."
    )
    return f"""You are the Procurement Agent in a Supply Demand Execution AI system.

You have access to these tools:
- list_open_pos:            Check existing open POs to avoid duplicates
- compare_suppliers:        Full comparison matrix for a SKU across all suppliers
- calculate_split_order:    Split a large/critical order across 2 suppliers
- find_best_supplier:       Quick single-winner supplier selection
- generate_purchase_order:  Create a PO (saved as draft)
- submit_po:                Submit a draft PO to the supplier
- get_po_status:            Check a PO's current status
- send_alert:               Queue an alert for critical situations

Procurement strategy:
1. Call list_open_pos first to avoid duplicate orders
2. For each SKU needing replenishment:
   a. If urgency = 'critical': use calculate_split_order then generate two POs
   b. Otherwise: use compare_suppliers to justify choice, then generate one PO
3. Submit POs under $5,000 immediately
4. {hitl_note}

Return JSON:
{{
  "issued_pos": [
    {{
      "po_number": "PO-XXXXXXXX",
      "sku_id": "SKU-XXX",
      "supplier_id": "SUP-XXX",
      "quantity": <float>,
      "total_value": <float>,
      "status": "submitted|draft",
      "expected_date": "YYYY-MM-DD",
      "urgency": "critical|high|medium|low",
      "is_split_order": <bool>
    }}
  ],
  "skipped": [],
  "hitl_required": <bool>,
  "hitl_po_numbers": [],
  "total_value": <float>,
  "procurement_commentary": "<summary>"
}}"""


def procurement_node(state: WorkflowState) -> dict[str, Any]:
    """
    LangGraph node: procurement (Phase 4 enhanced)
    ────────────────────────────────────────────────
    Autonomous multi-supplier procurement with:
      - Supplier comparison matrix before every PO
      - Split orders for critical urgency SKUs
      - HITL trigger for high-value POs
    """
    proc_input  = state.get("procurement_input") or {}
    sku_plans   = proc_input.get("sku_plans", [])
    run_id      = state.get("run_id", "UNKNOWN")
    hitl_enabled = settings.enable_human_in_the_loop

    needs_replenishment = [p for p in sku_plans if p.get("needs_replenishment")]

    if not needs_replenishment:
        return {
            "procurement_result": {
                "issued_pos": [], "skipped": [], "total_value": 0.0,
                "hitl_required": False,
                "procurement_commentary": "No replenishment needed.",
            },
            "procurement_status": "success",
            "hitl_required": False,
        }

    plans_summary = "\n".join([
        f"  {p['sku_id']}: urgency={p.get('urgency','medium')}, "
        f"qty={p.get('recommended_order_qty', p.get('eoq', 100)):.0f}, "
        f"on_hand={p.get('on_hand', 0):.0f}"
        for p in needs_replenishment
    ])

    try:
        llm_with_tools = _get_llm_with_tools()
        messages = [
            SystemMessage(content=_build_system_prompt(hitl_enabled)),
            HumanMessage(content=(
                f"Generate POs for:\n{plans_summary}\n\n"
                f"Run ID: {run_id}\n"
                f"HITL threshold: ${HITL_VALUE_THRESHOLD:,.0f}\n"
                f"Today: {datetime.utcnow().date()}"
            )),
        ]
        messages, final_content = _run_react_loop(
            llm_with_tools, messages, PROCUREMENT_TOOL_MAP
        )
        raw = final_content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result_data = json.loads(raw.strip())

    except Exception as exc:
        result_data = _fallback_procurement_enhanced(needs_replenishment, run_id, hitl_enabled)
        result_data["_fallback_reason"] = str(exc)

    # ── HITL checkpoint ────────────────────────────────────────────────────────
    hitl_required   = result_data.get("hitl_required", False)
    hitl_po_numbers = result_data.get("hitl_po_numbers", [])
    hitl_checkpoint = None

    if hitl_required and hitl_po_numbers:
        hitl_value = sum(
            po.get("total_value", 0)
            for po in result_data.get("issued_pos", [])
            if po.get("po_number") in hitl_po_numbers
        )
        hitl_checkpoint = build_interrupt_payload(
            run_id=run_id,
            agent="procurement",
            prompt=(
                f"{len(hitl_po_numbers)} PO(s) require approval. "
                f"Total: ${hitl_value:,.2f} (threshold: ${HITL_VALUE_THRESHOLD:,.0f}).\n"
                f"POs: {', '.join(hitl_po_numbers)}"
            ),
            context={
                "po_numbers":  hitl_po_numbers,
                "total_value": hitl_value,
                "threshold":   HITL_VALUE_THRESHOLD,
            },
        )

    issued_pos = result_data.get("issued_pos", [])
    procurement_result = {
        "issued_pos":      issued_pos,
        "skipped":         result_data.get("skipped", []),
        "total_pos":       len(issued_pos),
        "total_value":     result_data.get("total_value", 0.0),
        "hitl_triggered":  hitl_required,
        "hitl_po_numbers": hitl_po_numbers,
        "split_orders":    [po for po in issued_pos if po.get("is_split_order")],
        "procurement_commentary": result_data.get("procurement_commentary", ""),
        "generated_at":    datetime.utcnow().isoformat(),
    }

    return {
        "procurement_result": procurement_result,
        "procurement_status": "success",
        "hitl_required":      hitl_required,
        "hitl_checkpoint":    hitl_checkpoint,
    }


def _fallback_procurement_enhanced(
    needs_replenishment: list[dict],
    run_id: str,
    hitl_enabled: bool,
) -> dict:
    """
    Direct tool calls when LLM is unavailable.
    Phase 4: uses compare_suppliers + calculate_split_order for critical SKUs.
    """
    issued_pos:      list[dict] = []
    hitl_po_numbers: list[str]  = []
    total_value      = 0.0

    for plan in needs_replenishment:
        sku_id  = plan["sku_id"]
        qty     = plan.get("recommended_order_qty", plan.get("eoq", 100.0))
        urgency = plan.get("urgency", "medium")

        # ── Critical: try split order ──────────────────────────────────────────
        if urgency == "critical":
            split = calculate_split_order(sku_id=sku_id, total_quantity=qty)

            if split.get("feasible"):
                # Place two POs
                for order_key in ("primary_order", "secondary_order"):
                    order = split[order_key]
                    po = generate_purchase_order(
                        sku_id=sku_id,
                        quantity=order["quantity"],
                        supplier_id=order["supplier_id"],
                        notes=f"Split order ({order_key}) — urgency=critical. {split.get('rationale','')}",
                    )
                    if po.get("persisted"):
                        po_val = po.get("total_value", 0.0)
                        total_value += po_val
                        needs_hitl   = hitl_enabled and po_val >= HITL_VALUE_THRESHOLD
                        if not needs_hitl:
                            submit_po(po_number=po["po_number"])
                            status = "submitted"
                        else:
                            hitl_po_numbers.append(po["po_number"])
                            status = "draft"
                        issued_pos.append({
                            "po_number":    po["po_number"],
                            "sku_id":       sku_id,
                            "supplier_id":  order["supplier_id"],
                            "quantity":     order["quantity"],
                            "total_value":  po_val,
                            "status":       status,
                            "expected_date": order.get("expected_date"),
                            "urgency":      urgency,
                            "is_split_order": True,
                        })
                    # Alert for critical
                    send_alert(
                        severity="critical",
                        message=f"Critical split-order PO generated for {sku_id}",
                        sku_id=sku_id,
                        run_id=run_id,
                    )
                continue  # skip single-PO logic below

        # ── Non-critical: single PO with best supplier ─────────────────────────
        comparison = compare_suppliers(sku_id=sku_id, required_quantity=qty)
        supplier_id = comparison.get("recommended")

        po = generate_purchase_order(sku_id=sku_id, quantity=qty, supplier_id=supplier_id)
        if not po.get("persisted"):
            continue

        po_val     = po.get("total_value", 0.0)
        total_value += po_val
        needs_hitl  = hitl_enabled and po_val >= HITL_VALUE_THRESHOLD

        if not needs_hitl:
            submit_po(po_number=po["po_number"])
            status = "submitted"
        else:
            hitl_po_numbers.append(po["po_number"])
            status = "draft"

        if urgency in ("critical", "high"):
            send_alert(
                severity="high",
                message=f"PO {po['po_number']} generated for {sku_id} (urgency={urgency})",
                sku_id=sku_id,
                run_id=run_id,
            )

        issued_pos.append({
            "po_number":    po["po_number"],
            "sku_id":       sku_id,
            "supplier_id":  supplier_id,
            "quantity":     qty,
            "total_value":  po_val,
            "status":       status,
            "expected_date": po.get("expected_date"),
            "urgency":      urgency,
            "is_split_order": False,
        })

    split_count = sum(1 for po in issued_pos if po.get("is_split_order"))

    return {
        "issued_pos":        issued_pos,
        "skipped":           [],
        "hitl_required":     bool(hitl_po_numbers),
        "hitl_po_numbers":   hitl_po_numbers,
        "total_value":       round(total_value, 2),
        "procurement_commentary": (
            f"Fallback procurement: {len(issued_pos)} POs, "
            f"{split_count} split orders, "
            f"{len(hitl_po_numbers)} pending HITL."
        ),
    }
