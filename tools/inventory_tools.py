"""
tools/inventory_tools.py  (Phase 4 — extended with 5 new tools)
────────────────────────────────────────────────────────────────
All Phase 3 tools retained + 5 new tools added below:

  NEW in Phase 4:
  - update_stock_level       → write new stock snapshot to DB
  - record_stock_movement    → persist a transaction (receipt/sale/write-off)
  - bulk_score_all_skus      → score every SKU in one call, sorted by urgency
  - get_inventory_kpis       → warehouse-wide KPIs (turn rate, DOS, fill rate)
  - check_expiry_risk        → flag perishable SKUs expiring within lead time
"""

from __future__ import annotations

import math
import sqlite3
import statistics
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

class GetStockLevelArgs(BaseModel):
    sku_id:      str = Field(..., description="SKU identifier (e.g. 'SKU-001')")
    location_id: str = Field("DC-01", description="Warehouse location code")

class ComputeEOQArgs(BaseModel):
    sku_id:         str   = Field(..., description="SKU to compute EOQ for")
    annual_demand:  float = Field(..., description="Forecasted annual demand in units")
    ordering_cost:  float = Field(100.0, description="Cost to place one purchase order (USD)")
    holding_cost_pct: float = Field(0.25, description="Annual holding cost as fraction of unit cost")

class CalculateReorderPointArgs(BaseModel):
    sku_id:            str            = Field(..., description="SKU identifier")
    avg_daily_demand:  Optional[float] = Field(None)
    lead_time_days:    Optional[int]   = Field(None)
    safety_stock_days: int             = Field(14)

class ScoreInventoryHealthArgs(BaseModel):
    sku_id:           str            = Field(..., description="SKU to score")
    location_id:      str            = Field("DC-01")
    avg_daily_demand: Optional[float] = Field(None)

class ListCriticalSKUsArgs(BaseModel):
    location_id:       str   = Field("DC-01")
    max_results:       int   = Field(20)
    min_dos_threshold: float = Field(7.0)

class GetDaysOfSupplyArgs(BaseModel):
    sku_id:           str            = Field(...)
    location_id:      str            = Field("DC-01")
    avg_daily_demand: Optional[float] = Field(None)

# ── Phase 4 new arg schemas ───────────────────────────────────────────────────

class UpdateStockLevelArgs(BaseModel):
    sku_id:      str            = Field(..., description="SKU to update")
    location_id: str            = Field("DC-01")
    on_hand:     Optional[float] = Field(None, description="New on-hand quantity")
    reserved:    Optional[float] = Field(None, description="New reserved quantity")
    in_transit:  Optional[float] = Field(None, description="New in-transit quantity")

class RecordStockMovementArgs(BaseModel):
    sku_id:        str            = Field(...)
    movement_type: str            = Field(..., description="receipt|sale|adjustment|transfer_in|transfer_out|return|write_off")
    quantity:      float          = Field(..., description="Absolute quantity moved (always positive)")
    location_id:   str            = Field("DC-01")
    reference_doc: Optional[str]  = Field(None)
    notes:         Optional[str]  = Field(None)

class BulkScoreAllSKUsArgs(BaseModel):
    location_id:    str = Field("DC-01")
    include_healthy: bool = Field(False, description="Include healthy SKUs in result (default: critical/at_risk only)")

class GetInventoryKPIsArgs(BaseModel):
    location_id: str = Field("DC-01")

class CheckExpiryRiskArgs(BaseModel):
    location_id: str = Field("DC-01")
    days_ahead:  int = Field(30, description="Flag SKUs expiring within this many days")


# ── Internal DB helpers ───────────────────────────────────────────────────────

def _get_avg_daily_from_db(sku_id: str, days: int = 30) -> float:
    cutoff = str(date.today() - timedelta(days=days))
    conn   = sqlite3.connect(str(DB_PATH))
    row    = conn.execute(
        "SELECT AVG(quantity) FROM orders WHERE sku_id = ? AND order_date >= ?",
        (sku_id, cutoff),
    ).fetchone()
    conn.close()
    return float(row[0] or 10.0)


def _get_lead_time_from_db(sku_id: str) -> int:
    conn = sqlite3.connect(str(DB_PATH))
    row  = conn.execute(
        """SELECT s.lead_time_days FROM suppliers s
           JOIN sku_suppliers ss ON s.supplier_id = ss.supplier_id
           WHERE ss.sku_id = ? AND ss.is_primary = 1 LIMIT 1""",
        (sku_id,),
    ).fetchone()
    conn.close()
    return int(row[0]) if row else settings.default_lead_time_days


def _get_unit_cost_from_db(sku_id: str) -> float:
    conn = sqlite3.connect(str(DB_PATH))
    row  = conn.execute("SELECT unit_cost FROM skus WHERE sku_id = ?", (sku_id,)).fetchone()
    conn.close()
    return float(row[0]) if row else 10.0


def _compute_health_score(on_hand, reserved, reorder_pt, safety_stock, avg_daily):
    available   = max(0.0, on_hand - reserved)
    dos         = available / avg_daily if avg_daily > 0 else 999.0
    dos_score   = min(1.0, dos / 30.0)
    rop_score   = min(1.0, available / reorder_pt) if reorder_pt > 0 else 1.0
    ss_score    = min(1.0, available / safety_stock) if safety_stock > 0 else 1.0
    score       = round(0.50 * dos_score + 0.30 * rop_score + 0.20 * ss_score, 4)
    status      = ("stock_out" if on_hand <= 0 else
                   "critical"  if score < 0.40 else
                   "at_risk"   if score < 0.75 else "healthy")
    return score, status, available, dos


# ════════════════════════════════════════════════════════════════════
# PHASE 3 TOOLS (unchanged)
# ════════════════════════════════════════════════════════════════════

@tool(args_schema=GetStockLevelArgs)
def get_stock_level(sku_id: str, location_id: str = "DC-01") -> dict:
    """Get current inventory stock level for a SKU at a warehouse location.
    Returns on_hand, reserved, in_transit, available (ATP), total_supply."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        row  = conn.execute(
            "SELECT * FROM inventory_levels WHERE sku_id = ? AND location_id = ?",
            (sku_id.upper().strip(), location_id),
        ).fetchone()
        conn.close()
        if not row:
            return {"sku_id": sku_id, "location_id": location_id, "found": False,
                    "on_hand": 0.0, "available": 0.0, "error": "SKU not found"}
        r         = dict(row)
        available = max(0.0, r["on_hand"] - r.get("reserved", 0.0))
        return {**r, "found": True, "available": available,
                "total_supply": r["on_hand"] + r.get("in_transit", 0.0)}
    except Exception as exc:
        return {"sku_id": sku_id, "error": str(exc), "found": False}


@tool(args_schema=ComputeEOQArgs)
def compute_eoq(sku_id: str, annual_demand: float,
                ordering_cost: float = 100.0, holding_cost_pct: float = 0.25) -> dict:
    """Compute Economic Order Quantity. EOQ = sqrt(2DS/H). Returns eoq, annual_orders, days_between_orders, total_annual_cost."""
    try:
        unit_cost    = _get_unit_cost_from_db(sku_id)
        holding_cost = max(0.01, unit_cost * holding_cost_pct)
        eoq          = math.sqrt(2 * annual_demand * ordering_cost / holding_cost)
        orders_py    = annual_demand / eoq if eoq > 0 else 1.0
        total_cost   = (annual_demand / eoq * ordering_cost) + (eoq / 2 * holding_cost)
        return {"sku_id": sku_id, "annual_demand": round(annual_demand, 1),
                "eoq": round(eoq, 1), "ordering_cost": ordering_cost,
                "holding_cost_per_unit": round(holding_cost, 4),
                "annual_orders": round(orders_py, 2),
                "days_between_orders": round(365 / orders_py, 1),
                "total_annual_cost": round(total_cost, 2), "unit_cost": unit_cost}
    except Exception as exc:
        return {"sku_id": sku_id, "error": str(exc), "eoq": None}


@tool(args_schema=CalculateReorderPointArgs)
def calculate_reorder_point(sku_id: str, avg_daily_demand: Optional[float] = None,
                             lead_time_days: Optional[int] = None,
                             safety_stock_days: int = 14) -> dict:
    """Calculate reorder point. ROP = avg_daily × lead_time + safety_stock."""
    try:
        avg_daily    = avg_daily_demand or _get_avg_daily_from_db(sku_id)
        lead_time    = lead_time_days   or _get_lead_time_from_db(sku_id)
        safety_stock = round(avg_daily * safety_stock_days, 1)
        reorder_pt   = round(avg_daily * lead_time + safety_stock, 1)
        return {"sku_id": sku_id, "avg_daily_demand": round(avg_daily, 2),
                "lead_time_days": lead_time, "safety_stock_days": safety_stock_days,
                "safety_stock": safety_stock, "reorder_point": reorder_pt,
                "max_stock_level": round(reorder_pt + avg_daily * lead_time, 1),
                "interpretation": f"Order when stock hits {reorder_pt:.0f} units."}
    except Exception as exc:
        return {"sku_id": sku_id, "error": str(exc), "reorder_point": None}


@tool(args_schema=ScoreInventoryHealthArgs)
def score_inventory_health(sku_id: str, location_id: str = "DC-01",
                            avg_daily_demand: Optional[float] = None) -> dict:
    """Compute 0–1 inventory health score. Status: healthy(≥0.75) / at_risk(0.4–0.75) / critical(<0.4) / stock_out."""
    try:
        stock        = get_stock_level(sku_id=sku_id, location_id=location_id)
        if not stock.get("found"):
            return {"sku_id": sku_id, "health_score": 0.0, "health_status": "stock_out",
                    "error": "SKU not in inventory"}
        on_hand      = stock["on_hand"]
        reserved     = stock["reserved"]
        in_transit   = stock["in_transit"]
        reorder_pt   = stock.get("reorder_point") or 50.0
        safety_stock = stock.get("safety_stock")  or 20.0
        avg_daily    = avg_daily_demand or _get_avg_daily_from_db(sku_id)
        score, status, available, dos = _compute_health_score(
            on_hand, reserved, reorder_pt, safety_stock, avg_daily)
        return {"sku_id": sku_id, "location_id": location_id,
                "health_score": score, "health_status": status,
                "on_hand": on_hand, "available": available, "in_transit": in_transit,
                "days_of_supply": round(dos, 1),
                "reorder_triggered": (on_hand + in_transit) <= reorder_pt,
                "overstock": on_hand > reorder_pt * 2,
                "components": {
                    "dos_score": round(min(1.0, dos / 30.0), 4),
                    "rop_score": round(min(1.0, available / reorder_pt) if reorder_pt > 0 else 1.0, 4),
                    "ss_score":  round(min(1.0, available / safety_stock) if safety_stock > 0 else 1.0, 4),
                }}
    except Exception as exc:
        return {"sku_id": sku_id, "error": str(exc), "health_score": None}


@tool(args_schema=ListCriticalSKUsArgs)
def list_critical_skus(location_id: str = "DC-01", max_results: int = 20,
                        min_dos_threshold: float = 7.0) -> dict:
    """List all SKUs at critical or at-risk status, sorted by urgency (stock_out first)."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM inventory_levels WHERE location_id = ?", (location_id,)
        ).fetchall()
        conn.close()
        critical_skus = []
        for row in rows:
            r         = dict(row)
            sku_id    = r["sku_id"]
            on_hand   = r["on_hand"]
            reserved  = r.get("reserved", 0.0)
            in_transit = r.get("in_transit", 0.0)
            rop        = r.get("reorder_point") or 50.0
            available  = max(0.0, on_hand - reserved)
            avg_daily  = _get_avg_daily_from_db(sku_id, days=30)
            dos        = available / avg_daily if avg_daily > 0 else 999.0
            if on_hand <= 0 or dos < min_dos_threshold or (on_hand + in_transit) <= rop:
                status = ("stock_out" if on_hand <= 0 else
                          "critical"  if dos < 3   else "at_risk")
                critical_skus.append({"sku_id": sku_id, "status": status,
                                       "on_hand": on_hand, "available": available,
                                       "days_of_supply": round(dos, 1),
                                       "reorder_triggered": (on_hand + in_transit) <= rop})
        priority = {"stock_out": 0, "critical": 1, "at_risk": 2}
        critical_skus.sort(key=lambda x: priority.get(x["status"], 3))
        return {"location_id": location_id, "total_critical": len(critical_skus),
                "skus": critical_skus[:max_results]}
    except Exception as exc:
        return {"error": str(exc), "skus": []}


@tool(args_schema=GetDaysOfSupplyArgs)
def get_days_of_supply(sku_id: str, location_id: str = "DC-01",
                        avg_daily_demand: Optional[float] = None) -> dict:
    """Calculate days of supply remaining. Urgency: stock_out / critical(<7d) / urgent(<14d) / watch(<30d) / comfortable."""
    try:
        stock     = get_stock_level(sku_id=sku_id, location_id=location_id)
        available = stock.get("available", 0.0)
        avg_daily = avg_daily_demand or _get_avg_daily_from_db(sku_id)
        dos       = available / avg_daily if avg_daily > 0 else 999.0
        urgency   = ("stock_out"   if available <= 0 else
                     "critical"    if dos < 7    else
                     "urgent"      if dos < 14   else
                     "watch"       if dos < 30   else "comfortable")
        return {"sku_id": sku_id, "available": available,
                "avg_daily_demand": round(avg_daily, 2),
                "days_of_supply": round(dos, 1), "urgency": urgency,
                "stockout_date": str(date.today() + timedelta(days=int(dos))) if dos < 999 else None}
    except Exception as exc:
        return {"sku_id": sku_id, "error": str(exc), "days_of_supply": None}


# ════════════════════════════════════════════════════════════════════
# PHASE 4 — NEW TOOLS
# ════════════════════════════════════════════════════════════════════

@tool(args_schema=UpdateStockLevelArgs)
def update_stock_level(
    sku_id:      str,
    location_id: str            = "DC-01",
    on_hand:     Optional[float] = None,
    reserved:    Optional[float] = None,
    in_transit:  Optional[float] = None,
) -> dict:
    """
    Write updated stock quantities back to the inventory_levels table.

    Only provided fields are updated — pass None to leave a field unchanged.
    Use after receiving goods (update on_hand + in_transit), after sales
    (update reserved), or after a stock count reconciliation.

    Returns the updated stock level record.
    """
    try:
        updates: list[str] = ["last_updated = ?"]
        params:  list      = [datetime.utcnow().isoformat()]

        if on_hand    is not None: updates.append("on_hand = ?");   params.append(on_hand)
        if reserved   is not None: updates.append("reserved = ?");  params.append(reserved)
        if in_transit is not None: updates.append("in_transit = ?");params.append(in_transit)

        params += [sku_id.upper().strip(), location_id]

        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA foreign_keys = ON")
        result = conn.execute(
            f"UPDATE inventory_levels SET {', '.join(updates)} "
            f"WHERE sku_id = ? AND location_id = ?",
            params,
        )
        conn.commit()

        if result.rowcount == 0:
            conn.close()
            return {"success": False, "error": f"{sku_id} not found in inventory_levels"}

        # Return updated record
        conn.row_factory = sqlite3.Row
        row  = conn.execute(
            "SELECT * FROM inventory_levels WHERE sku_id = ? AND location_id = ?",
            (sku_id.upper().strip(), location_id),
        ).fetchone()
        conn.close()

        r         = dict(row)
        available = max(0.0, r["on_hand"] - r.get("reserved", 0.0))
        return {**r, "found": True, "available": available, "success": True,
                "total_supply": r["on_hand"] + r.get("in_transit", 0.0)}

    except Exception as exc:
        return {"sku_id": sku_id, "success": False, "error": str(exc)}


@tool(args_schema=RecordStockMovementArgs)
def record_stock_movement(
    sku_id:        str,
    movement_type: str,
    quantity:      float,
    location_id:   str           = "DC-01",
    reference_doc: Optional[str] = None,
    notes:         Optional[str] = None,
) -> dict:
    """
    Persist a stock movement transaction to the stock_movements ledger.

    Valid movement_type values:
      receipt      → goods received from supplier (increases on_hand)
      sale         → units sold / fulfilled (decreases on_hand)
      adjustment   → manual stock count correction
      transfer_in  → moved in from another location
      transfer_out → moved out to another location
      return       → customer return (increases on_hand)
      write_off    → damaged or expired goods removed

    Also updates the inventory_levels table to reflect the new balance.
    Returns the movement_id and updated stock balance.
    """
    VALID_TYPES = {"receipt", "sale", "adjustment", "transfer_in",
                   "transfer_out", "return", "write_off"}
    if movement_type not in VALID_TYPES:
        return {"error": f"Invalid movement_type '{movement_type}'. Valid: {sorted(VALID_TYPES)}"}
    if quantity <= 0:
        return {"error": "quantity must be positive"}

    OUTBOUND = {"sale", "transfer_out", "write_off"}
    delta    = -quantity if movement_type in OUTBOUND else quantity

    movement_id = str(uuid4())
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA foreign_keys = ON")

        # Insert movement record
        conn.execute(
            """INSERT INTO stock_movements
               (movement_id, sku_id, location_id, movement_type, quantity,
                reference_doc, notes, recorded_by, occurred_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (movement_id, sku_id.upper().strip(), location_id, movement_type,
             quantity, reference_doc, notes, "inventory_agent",
             datetime.utcnow().isoformat()),
        )

        # Update inventory_levels
        conn.execute(
            "UPDATE inventory_levels SET on_hand = MAX(0, on_hand + ?), last_updated = ? "
            "WHERE sku_id = ? AND location_id = ?",
            (delta, datetime.utcnow().isoformat(),
             sku_id.upper().strip(), location_id),
        )
        conn.commit()

        # Return updated balance
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT on_hand, reserved, in_transit FROM inventory_levels "
            "WHERE sku_id = ? AND location_id = ?",
            (sku_id.upper().strip(), location_id),
        ).fetchone()
        conn.close()

        new_balance = dict(row) if row else {}
        return {
            "success":       True,
            "movement_id":   movement_id,
            "sku_id":        sku_id.upper().strip(),
            "movement_type": movement_type,
            "quantity":      quantity,
            "delta":         delta,
            "new_on_hand":   new_balance.get("on_hand", 0.0),
            "new_available": max(0.0, new_balance.get("on_hand", 0.0)
                                    - new_balance.get("reserved", 0.0)),
        }

    except Exception as exc:
        return {"success": False, "movement_id": movement_id, "error": str(exc)}


@tool(args_schema=BulkScoreAllSKUsArgs)
def bulk_score_all_skus(
    location_id:     str  = "DC-01",
    include_healthy: bool = False,
) -> dict:
    """
    Score inventory health for ALL SKUs at a location in a single call.

    Returns a ranked list sorted by urgency (worst first).
    This is the primary tool for the Inventory Agent's initial triage —
    call this once instead of score_inventory_health for each SKU.

    Args:
        include_healthy: if False (default), only returns at_risk / critical / stock_out SKUs
        location_id:     warehouse to assess

    Returns:
        {
          "location_id": ...,
          "total_skus":  20,
          "stock_out":   2,
          "critical":    3,
          "at_risk":     4,
          "healthy":     11,
          "records": [sorted list of health records]
        }
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM inventory_levels WHERE location_id = ?", (location_id,)
        ).fetchall()
        conn.close()

        records: list[dict] = []
        counts = {"stock_out": 0, "critical": 0, "at_risk": 0, "healthy": 0}

        for row in rows:
            r          = dict(row)
            sku_id     = r["sku_id"]
            on_hand    = r["on_hand"]
            reserved   = r.get("reserved", 0.0)
            in_transit = r.get("in_transit", 0.0)
            rop        = r.get("reorder_point") or 50.0
            ss         = r.get("safety_stock")  or 20.0
            avg_daily  = _get_avg_daily_from_db(sku_id, days=30)

            score, status, available, dos = _compute_health_score(
                on_hand, reserved, rop, ss, avg_daily
            )
            counts[status] += 1

            if not include_healthy and status == "healthy":
                continue

            stockout_date = None
            if dos < 999 and avg_daily > 0:
                stockout_date = str(date.today() + timedelta(days=int(dos)))

            records.append({
                "sku_id":            sku_id,
                "health_status":     status,
                "health_score":      score,
                "on_hand":           on_hand,
                "available":         available,
                "in_transit":        in_transit,
                "days_of_supply":    round(dos, 1),
                "reorder_triggered": (on_hand + in_transit) <= rop,
                "overstock":         on_hand > rop * 2,
                "stockout_risk_date": stockout_date,
                "avg_daily_demand":  round(avg_daily, 2),
            })

        # Sort: stock_out → critical → at_risk → healthy
        priority = {"stock_out": 0, "critical": 1, "at_risk": 2, "healthy": 3}
        records.sort(key=lambda x: (priority.get(x["health_status"], 3), x["days_of_supply"]))

        reorder_triggers = [r["sku_id"] for r in records if r["reorder_triggered"]]

        return {
            "location_id":       location_id,
            "total_skus":        len(rows),
            "stock_out":         counts["stock_out"],
            "critical":          counts["critical"],
            "at_risk":           counts["at_risk"],
            "healthy":           counts["healthy"],
            "reorder_triggers":  reorder_triggers,
            "records":           records,
        }

    except Exception as exc:
        return {"error": str(exc), "records": []}


@tool(args_schema=GetInventoryKPIsArgs)
def get_inventory_kpis(location_id: str = "DC-01") -> dict:
    """
    Calculate warehouse-wide inventory KPIs in a single call.

    KPIs returned:
      - avg_days_of_supply       → average DoS across all SKUs
      - fill_rate_pct            → % of SKUs with on_hand > 0
      - stockout_count           → SKUs with zero on_hand
      - reorder_trigger_count    → SKUs at or below reorder point
      - overstock_count          → SKUs with on_hand > 2× reorder point
      - total_inventory_value    → on_hand units × unit_cost
      - avg_health_score         → average 0–1 health score
      - inventory_turn_estimate  → estimated annual turns (annual_demand / avg_on_hand)

    Use at the start of a workflow run for a quick warehouse health check
    before diving into per-SKU analysis.
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        inv_rows = conn.execute(
            "SELECT il.*, s.unit_cost FROM inventory_levels il "
            "JOIN skus s ON il.sku_id = s.sku_id "
            "WHERE il.location_id = ?", (location_id,)
        ).fetchall()
        conn.close()

        if not inv_rows:
            return {"error": "No inventory records found", "location_id": location_id}

        dos_list:      list[float] = []
        scores:        list[float] = []
        total_value                = 0.0
        stockout_count             = 0
        reorder_trigger_count      = 0
        overstock_count            = 0
        annual_demand_total        = 0.0
        on_hand_total              = 0.0

        for row in inv_rows:
            r          = dict(row)
            sku_id     = r["sku_id"]
            on_hand    = r["on_hand"]
            reserved   = r.get("reserved", 0.0)
            in_transit = r.get("in_transit", 0.0)
            rop        = r.get("reorder_point") or 50.0
            ss         = r.get("safety_stock")  or 20.0
            unit_cost  = r.get("unit_cost", 10.0)
            avg_daily  = _get_avg_daily_from_db(sku_id, days=30)

            score, status, available, dos = _compute_health_score(
                on_hand, reserved, rop, ss, avg_daily
            )

            dos_list.append(dos if dos < 999 else 30.0)
            scores.append(score)
            total_value        += on_hand * unit_cost
            annual_demand_total += avg_daily * 365
            on_hand_total       += on_hand

            if on_hand <= 0:
                stockout_count += 1
            if (on_hand + in_transit) <= rop:
                reorder_trigger_count += 1
            if on_hand > rop * 2:
                overstock_count += 1

        total_skus         = len(inv_rows)
        fill_rate          = round((total_skus - stockout_count) / total_skus * 100, 1)
        avg_dos            = round(statistics.mean(dos_list), 1) if dos_list else 0.0
        avg_health         = round(statistics.mean(scores), 3) if scores else 0.0
        inv_turn_estimate  = round(annual_demand_total / on_hand_total, 2) if on_hand_total > 0 else 0.0

        return {
            "location_id":            location_id,
            "total_skus":             total_skus,
            "avg_days_of_supply":     avg_dos,
            "fill_rate_pct":          fill_rate,
            "stockout_count":         stockout_count,
            "reorder_trigger_count":  reorder_trigger_count,
            "overstock_count":        overstock_count,
            "total_inventory_value":  round(total_value, 2),
            "avg_health_score":       avg_health,
            "inventory_turn_estimate": inv_turn_estimate,
            "generated_at":           datetime.utcnow().isoformat(),
        }

    except Exception as exc:
        return {"error": str(exc), "location_id": location_id}


@tool(args_schema=CheckExpiryRiskArgs)
def check_expiry_risk(location_id: str = "DC-01", days_ahead: int = 30) -> dict:
    """
    Identify perishable SKUs whose earliest batch expires within `days_ahead` days.

    A SKU is flagged when its expiry_date in inventory_levels falls before
    today + days_ahead AND on_hand > 0. This helps prevent write-off losses
    by prompting early sale or redistribution.

    Returns a list of at-risk SKUs with days_until_expiry and recommended action.
    """
    try:
        threshold = str(date.today() + timedelta(days=days_ahead))

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT il.sku_id, il.on_hand, il.expiry_date, s.shelf_life_days, s.name
               FROM inventory_levels il
               JOIN skus s ON il.sku_id = s.sku_id
               WHERE il.location_id = ?
                 AND il.expiry_date IS NOT NULL
                 AND il.expiry_date <= ?
                 AND il.on_hand > 0""",
            (location_id, threshold),
        ).fetchall()
        conn.close()

        at_risk: list[dict] = []
        for row in rows:
            r              = dict(row)
            expiry         = date.fromisoformat(r["expiry_date"])
            days_remaining = (expiry - date.today()).days

            action = (
                "URGENT: Redistribute or markdown immediately" if days_remaining <= 7 else
                "Promote or redistribute within 2 weeks"       if days_remaining <= 14 else
                "Monitor closely — plan markdown"
            )

            at_risk.append({
                "sku_id":          r["sku_id"],
                "name":            r["name"],
                "on_hand":         r["on_hand"],
                "expiry_date":     r["expiry_date"],
                "days_until_expiry": days_remaining,
                "recommended_action": action,
            })

        at_risk.sort(key=lambda x: x["days_until_expiry"])

        return {
            "location_id":    location_id,
            "days_ahead":     days_ahead,
            "at_risk_count":  len(at_risk),
            "at_risk_skus":   at_risk,
        }

    except Exception as exc:
        return {"error": str(exc), "at_risk_skus": []}


# ── Tool registries ───────────────────────────────────────────────────────────

INVENTORY_TOOLS = [
    get_stock_level,
    compute_eoq,
    calculate_reorder_point,
    score_inventory_health,
    list_critical_skus,
    get_days_of_supply,
    # Phase 4 additions
    update_stock_level,
    record_stock_movement,
    bulk_score_all_skus,
    get_inventory_kpis,
    check_expiry_risk,
]
