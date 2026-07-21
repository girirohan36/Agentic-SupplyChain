"""
tools/procurement_tools.py  (Phase 4 — extended)
─────────────────────────────────────────────────
All Phase 3 tools retained + 2 new tools:

  NEW in Phase 4:
  - compare_suppliers      → full comparison matrix for all suppliers of a SKU
  - calculate_split_order  → split a large order across 2 suppliers for risk reduction
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from uuid import uuid4

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config.settings import get_settings

settings = get_settings()
DB_PATH  = Path("data/supply_demand.db")


# ── Arg schemas ───────────────────────────────────────────────────────────────

class GetSupplierInfoArgs(BaseModel):
    supplier_id: str = Field(...)

class FindBestSupplierArgs(BaseModel):
    sku_id:             str   = Field(...)
    weight_reliability: float = Field(0.50)
    weight_cost:        float = Field(0.30)
    weight_lead_time:   float = Field(0.20)

class GeneratePurchaseOrderArgs(BaseModel):
    sku_id:      str           = Field(...)
    quantity:    float         = Field(...)
    supplier_id: Optional[str] = Field(None)
    notes:       Optional[str] = Field(None)

class SubmitPOArgs(BaseModel):
    po_number: str = Field(...)

class GetPOStatusArgs(BaseModel):
    po_number: str = Field(...)

class ListOpenPOsArgs(BaseModel):
    supplier_id: Optional[str] = Field(None)
    limit:       int            = Field(20)

class CancelPOArgs(BaseModel):
    po_number: str = Field(...)
    reason:    str = Field("")

# Phase 4 new schemas
class CompareSuppliersArgs(BaseModel):
    sku_id:             str   = Field(..., description="SKU to source")
    required_quantity:  float = Field(..., description="Units needed — used to calculate total cost per supplier")
    weight_reliability: float = Field(0.40, description="Reliability weighting")
    weight_cost:        float = Field(0.35, description="Total cost weighting")
    weight_lead_time:   float = Field(0.25, description="Lead time weighting")

class CalculateSplitOrderArgs(BaseModel):
    sku_id:          str   = Field(..., description="SKU to order")
    total_quantity:  float = Field(..., description="Total units needed across both suppliers")
    primary_split:   float = Field(0.70, description="Fraction to place with primary supplier (0.5–0.9)")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_supplier(supplier_id: str) -> dict | None:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row  = conn.execute("SELECT * FROM suppliers WHERE supplier_id = ?", (supplier_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def _load_sku(sku_id: str) -> dict | None:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row  = conn.execute("SELECT * FROM skus WHERE sku_id = ?", (sku_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ════════════════════════════════════════════════════════════════════
# PHASE 3 TOOLS (retained)
# ════════════════════════════════════════════════════════════════════

@tool(args_schema=GetSupplierInfoArgs)
def get_supplier_info(supplier_id: str) -> dict:
    """Retrieve full supplier record from vendor master. Returns name, country, lead_time, reliability, status."""
    supplier = _load_supplier(supplier_id.upper().strip())
    if not supplier:
        return {"error": f"Supplier {supplier_id} not found", "found": False}
    return {**supplier, "found": True}


@tool(args_schema=FindBestSupplierArgs)
def find_best_supplier(sku_id: str, weight_reliability: float = 0.50,
                        weight_cost: float = 0.30, weight_lead_time: float = 0.20) -> dict:
    """Rank all eligible suppliers for a SKU by composite score (reliability, cost, lead time)."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT s.* FROM suppliers s
               JOIN sku_suppliers ss ON s.supplier_id = ss.supplier_id
               WHERE ss.sku_id = ? AND s.status NOT IN ('blacklisted','on_hold')""",
            (sku_id.upper().strip(),),
        ).fetchall()
        conn.close()
        if not rows:
            return {"sku_id": sku_id, "error": "No eligible suppliers"}
        suppliers  = [dict(r) for r in rows]
        costs      = [s["unit_cost"]      for s in suppliers]
        leads      = [s["lead_time_days"] for s in suppliers]
        min_c, max_c = min(costs), max(costs)
        min_l, max_l = min(leads), max(leads)
        ranked = []
        for s in suppliers:
            nc = (s["unit_cost"]      - min_c) / (max_c - min_c + 1e-9)
            nl = (s["lead_time_days"] - min_l) / (max_l - min_l + 1e-9)
            score = (s["reliability_score"] * weight_reliability
                     + (1 - nc) * weight_cost + (1 - nl) * weight_lead_time)
            ranked.append({**s, "composite_score": round(score, 4)})
        ranked.sort(key=lambda x: x["composite_score"], reverse=True)
        return {"sku_id": sku_id, "best_supplier": ranked[0], "all_ranked": ranked,
                "selection_rationale": (
                    f"Selected {ranked[0]['name']} (score {ranked[0]['composite_score']:.3f}) — "
                    f"reliability {ranked[0]['reliability_score']:.0%}, "
                    f"lead {ranked[0]['lead_time_days']}d, cost ${ranked[0]['unit_cost']:.2f}")}
    except Exception as exc:
        return {"sku_id": sku_id, "error": str(exc)}


@tool(args_schema=GeneratePurchaseOrderArgs)
def generate_purchase_order(sku_id: str, quantity: float,
                             supplier_id: Optional[str] = None,
                             notes: Optional[str] = None) -> dict:
    """Generate and persist a draft Purchase Order. Auto-selects best supplier if supplier_id not given."""
    try:
        sku_id = sku_id.upper().strip()
        if supplier_id:
            supplier = _load_supplier(supplier_id.upper().strip())
            if not supplier: return {"error": f"Supplier {supplier_id} not found"}
        else:
            best = find_best_supplier(sku_id=sku_id)
            if "error" in best: return {"error": best["error"]}
            supplier    = best["best_supplier"]
            supplier_id = supplier["supplier_id"]
        if supplier.get("status") == "blacklisted":
            return {"error": f"Supplier {supplier_id} is blacklisted"}
        sku        = _load_sku(sku_id) or {}
        unit_cost  = supplier.get("unit_cost", sku.get("unit_cost", 10.0))
        min_qty    = supplier.get("min_order_qty", 0.0)
        order_qty  = max(quantity, min_qty)
        lead_time  = supplier.get("lead_time_days", settings.default_lead_time_days)
        exp_date   = str(date.today() + timedelta(days=lead_time))
        po_number  = f"PO-{uuid4().hex[:8].upper()}"
        line_total = round(order_qty * unit_cost, 2)
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT OR IGNORE INTO purchase_orders "
            "(po_number,supplier_id,status,total_value,issued_date,expected_date,notes,created_by_agent) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (po_number, supplier_id, "draft", line_total, str(date.today()),
             exp_date, notes or "Auto-generated", "procurement_agent"),
        )
        conn.execute(
            "INSERT OR IGNORE INTO purchase_order_lines "
            "(po_number,line_number,sku_id,description,quantity,unit_cost,line_total) "
            "VALUES (?,?,?,?,?,?,?)",
            (po_number, 1, sku_id, sku.get("name", sku_id), round(order_qty, 1), unit_cost, line_total),
        )
        conn.commit(); conn.close()
        return {"po_number": po_number, "supplier_id": supplier_id,
                "supplier_name": supplier.get("name", ""), "status": "draft",
                "issued_date": str(date.today()), "expected_date": exp_date,
                "lines": [{"line_number": 1, "sku_id": sku_id,
                            "description": sku.get("name", sku_id),
                            "quantity": round(order_qty, 1), "unit_cost": unit_cost,
                            "line_total": line_total}],
                "total_value": line_total, "persisted": True}
    except Exception as exc:
        return {"sku_id": sku_id, "error": str(exc), "persisted": False}


@tool(args_schema=SubmitPOArgs)
def submit_po(po_number: str) -> dict:
    """Change PO status from draft → submitted. Only call after obtaining required approvals."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        result = conn.execute(
            "UPDATE purchase_orders SET status='submitted' WHERE po_number=? AND status='draft'",
            (po_number,),
        )
        conn.commit()
        if result.rowcount == 0:
            conn.close()
            return {"po_number": po_number, "success": False, "error": "PO not found or not draft"}
        row = conn.execute(
            "SELECT po_number,status,total_value,expected_date FROM purchase_orders WHERE po_number=?",
            (po_number,),
        ).fetchone()
        conn.close()
        return {"po_number": po_number, "success": True, "new_status": "submitted",
                "total_value": row[2] if row else None,
                "expected_date": row[3] if row else None}
    except Exception as exc:
        return {"po_number": po_number, "success": False, "error": str(exc)}


@tool(args_schema=GetPOStatusArgs)
def get_po_status(po_number: str) -> dict:
    """Look up a Purchase Order by number. Returns full PO header + line items."""
    try:
        conn = sqlite3.connect(str(DB_PATH)); conn.row_factory = sqlite3.Row
        po   = conn.execute("SELECT * FROM purchase_orders WHERE po_number=?", (po_number,)).fetchone()
        if not po: conn.close(); return {"po_number": po_number, "found": False}
        lines = conn.execute("SELECT * FROM purchase_order_lines WHERE po_number=?", (po_number,)).fetchall()
        conn.close()
        return {**dict(po), "found": True, "lines": [dict(l) for l in lines]}
    except Exception as exc:
        return {"po_number": po_number, "error": str(exc), "found": False}


@tool(args_schema=ListOpenPOsArgs)
def list_open_pos(supplier_id: Optional[str] = None, limit: int = 20) -> dict:
    """List all draft or submitted POs. Use before generating new POs to avoid duplicates."""
    try:
        conn   = sqlite3.connect(str(DB_PATH)); conn.row_factory = sqlite3.Row
        query  = "SELECT * FROM purchase_orders WHERE status IN ('draft','submitted')"
        params: list = []
        if supplier_id:
            query += " AND supplier_id=?"; params.append(supplier_id.upper().strip())
        query += " ORDER BY issued_date DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall(); conn.close()
        return {"total_open": len(rows), "purchase_orders": [dict(r) for r in rows]}
    except Exception as exc:
        return {"error": str(exc), "purchase_orders": []}


@tool(args_schema=CancelPOArgs)
def cancel_po(po_number: str, reason: str = "") -> dict:
    """Cancel a draft or submitted PO. Cannot cancel shipped or received POs."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        row  = conn.execute("SELECT status FROM purchase_orders WHERE po_number=?", (po_number,)).fetchone()
        if not row: conn.close(); return {"po_number": po_number, "success": False, "error": "Not found"}
        if row[0] in ("shipped", "received"):
            conn.close()
            return {"po_number": po_number, "success": False, "error": f"Cannot cancel {row[0]} PO"}
        conn.execute(
            "UPDATE purchase_orders SET status='cancelled', notes=notes||? WHERE po_number=?",
            (f" | Cancelled: {reason}" if reason else " | Cancelled", po_number),
        )
        conn.commit(); conn.close()
        return {"po_number": po_number, "success": True, "new_status": "cancelled"}
    except Exception as exc:
        return {"po_number": po_number, "success": False, "error": str(exc)}


# ════════════════════════════════════════════════════════════════════
# PHASE 4 — NEW TOOLS
# ════════════════════════════════════════════════════════════════════

@tool(args_schema=CompareSuppliersArgs)
def compare_suppliers(
    sku_id:             str,
    required_quantity:  float,
    weight_reliability: float = 0.40,
    weight_cost:        float = 0.35,
    weight_lead_time:   float = 0.25,
) -> dict:
    """
    Produce a full comparison matrix for all suppliers of a SKU.

    Unlike find_best_supplier (which just returns the winner), this tool
    returns a detailed side-by-side comparison including:
      - total_cost = required_quantity × unit_cost
      - delivery_date = today + lead_time_days
      - risk_score = 1 - reliability_score
      - composite_score (same formula as find_best_supplier)
      - recommendation: 'use', 'backup', 'avoid'

    Use this when the Procurement Agent needs to justify its supplier choice
    or when considering a split order across multiple vendors.
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT s.*, ss.is_primary, ss.contracted_cost
               FROM suppliers s
               JOIN sku_suppliers ss ON s.supplier_id = ss.supplier_id
               WHERE ss.sku_id = ? AND s.status NOT IN ('blacklisted')
               ORDER BY ss.is_primary DESC""",
            (sku_id.upper().strip(),),
        ).fetchall()
        conn.close()

        if not rows:
            return {"sku_id": sku_id, "error": "No suppliers found for this SKU"}

        suppliers = [dict(r) for r in rows]
        costs     = [s.get("contracted_cost") or s["unit_cost"] for s in suppliers]
        leads     = [s["lead_time_days"] for s in suppliers]
        min_c, max_c = min(costs), max(costs)
        min_l, max_l = min(leads), max(leads)

        comparison: list[dict] = []
        for s in suppliers:
            eff_cost    = s.get("contracted_cost") or s["unit_cost"]
            total_cost  = round(required_quantity * eff_cost, 2)
            nc          = (eff_cost            - min_c) / (max_c - min_c + 1e-9)
            nl          = (s["lead_time_days"] - min_l) / (max_l - min_l + 1e-9)
            composite   = round(
                s["reliability_score"] * weight_reliability
                + (1 - nc) * weight_cost
                + (1 - nl) * weight_lead_time,
                4,
            )
            delivery_date = str(date.today() + timedelta(days=s["lead_time_days"]))

            status = s.get("status", "active")
            recommendation = (
                "avoid"  if status in ("on_hold", "blacklisted") else
                "use"    if composite >= 0.70 else
                "backup" if composite >= 0.45 else
                "avoid"
            )

            comparison.append({
                "supplier_id":     s["supplier_id"],
                "name":            s["name"],
                "country":         s["country"],
                "unit_cost":       eff_cost,
                "total_cost":      total_cost,
                "lead_time_days":  s["lead_time_days"],
                "delivery_date":   delivery_date,
                "reliability":     s["reliability_score"],
                "risk_score":      round(1.0 - s["reliability_score"], 3),
                "is_primary":      bool(s.get("is_primary")),
                "status":          status,
                "composite_score": composite,
                "recommendation":  recommendation,
            })

        comparison.sort(key=lambda x: x["composite_score"], reverse=True)

        return {
            "sku_id":            sku_id,
            "required_quantity": required_quantity,
            "supplier_count":    len(comparison),
            "recommended":       comparison[0]["supplier_id"],
            "comparison":        comparison,
            "cheapest":          min(comparison, key=lambda x: x["total_cost"])["supplier_id"],
            "fastest":           min(comparison, key=lambda x: x["lead_time_days"])["supplier_id"],
            "most_reliable":     max(comparison, key=lambda x: x["reliability"])["supplier_id"],
        }

    except Exception as exc:
        return {"sku_id": sku_id, "error": str(exc)}


@tool(args_schema=CalculateSplitOrderArgs)
def calculate_split_order(
    sku_id:         str,
    total_quantity: float,
    primary_split:  float = 0.70,
) -> dict:
    """
    Calculate a split purchase order across two suppliers to reduce supply risk.

    Split ordering is used when:
      - A single supplier cannot fulfil the full quantity
      - Supply disruption risk is high (low reliability score)
      - Urgency is critical and parallel delivery is needed

    Args:
        sku_id:         SKU to source
        total_quantity: total units needed
        primary_split:  fraction going to primary supplier (0.5–0.9, default 0.7)

    Returns two draft PO recommendations with quantities, costs, dates,
    and an explanation of the risk-reduction rationale.

    NOTE: This tool only calculates — it does not generate POs.
          Call generate_purchase_order twice with the returned quantities to
          actually create the POs.
    """
    primary_split = max(0.50, min(0.90, primary_split))  # clamp to valid range

    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT s.*, ss.is_primary, ss.contracted_cost
               FROM suppliers s
               JOIN sku_suppliers ss ON s.supplier_id = ss.supplier_id
               WHERE ss.sku_id = ? AND s.status NOT IN ('blacklisted','on_hold')
               ORDER BY ss.is_primary DESC""",
            (sku_id.upper().strip(),),
        ).fetchall()
        conn.close()

        eligible = [dict(r) for r in rows]
        if len(eligible) < 2:
            return {
                "sku_id":    sku_id,
                "feasible":  False,
                "reason":    f"Need ≥2 eligible suppliers, found {len(eligible)}",
                "fallback":  "Place full order with single supplier",
            }

        primary   = eligible[0]   # highest is_primary score
        secondary = eligible[1]

        qty_primary   = round(total_quantity * primary_split, 1)
        qty_secondary = round(total_quantity * (1 - primary_split), 1)

        cost_p = round(qty_primary   * (primary.get("contracted_cost") or primary["unit_cost"]), 2)
        cost_s = round(qty_secondary * (secondary.get("contracted_cost") or secondary["unit_cost"]), 2)

        date_p = str(date.today() + timedelta(days=primary["lead_time_days"]))
        date_s = str(date.today() + timedelta(days=secondary["lead_time_days"]))

        # First delivery arrival date
        first_arrival = date_p if primary["lead_time_days"] <= secondary["lead_time_days"] else date_s

        return {
            "sku_id":          sku_id,
            "feasible":        True,
            "total_quantity":  total_quantity,
            "total_cost":      round(cost_p + cost_s, 2),
            "first_arrival":   first_arrival,
            "primary_order": {
                "supplier_id":   primary["supplier_id"],
                "supplier_name": primary["name"],
                "quantity":      qty_primary,
                "unit_cost":     primary.get("contracted_cost") or primary["unit_cost"],
                "total_cost":    cost_p,
                "expected_date": date_p,
                "lead_time":     primary["lead_time_days"],
                "reliability":   primary["reliability_score"],
            },
            "secondary_order": {
                "supplier_id":   secondary["supplier_id"],
                "supplier_name": secondary["name"],
                "quantity":      qty_secondary,
                "unit_cost":     secondary.get("contracted_cost") or secondary["unit_cost"],
                "total_cost":    cost_s,
                "expected_date": date_s,
                "lead_time":     secondary["lead_time_days"],
                "reliability":   secondary["reliability_score"],
            },
            "rationale": (
                f"Split {primary_split:.0%}/{1-primary_split:.0%} across "
                f"{primary['name']} and {secondary['name']}. "
                f"Combined reliability: {1-(1-primary['reliability_score'])*(1-secondary['reliability_score']):.1%}. "
                f"First delivery expected {first_arrival}."
            ),
            "next_steps": [
                f"Call generate_purchase_order(sku_id='{sku_id}', quantity={qty_primary}, "
                f"supplier_id='{primary['supplier_id']}')",
                f"Call generate_purchase_order(sku_id='{sku_id}', quantity={qty_secondary}, "
                f"supplier_id='{secondary['supplier_id']}')",
            ],
        }

    except Exception as exc:
        return {"sku_id": sku_id, "feasible": False, "error": str(exc)}


# ── Tool registry ─────────────────────────────────────────────────────────────

PROCUREMENT_TOOLS = [
    get_supplier_info,
    find_best_supplier,
    generate_purchase_order,
    submit_po,
    get_po_status,
    list_open_pos,
    cancel_po,
    # Phase 4 additions
    compare_suppliers,
    calculate_split_order,
]
