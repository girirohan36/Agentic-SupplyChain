"""
tools/fulfillment_tools.py
───────────────────────────
LangChain @tool definitions for the Fulfillment Agent.

Tools:
  - get_open_orders         → load unfulfilled demand per SKU
  - score_order_priority    → rank orders by urgency / volume
  - route_to_best_dc        → select optimal DC by stock + proximity
  - create_dispatch         → persist a dispatch record to DB
  - manage_backorder        → record a held order with ETA
  - calculate_fill_rate     → % demand fulfilled in a run
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

class GetOpenOrdersArgs(BaseModel):
    sku_ids:   Optional[list] = Field(None,  description="Filter by SKU list — None returns all SKUs with demand")
    days_back: int            = Field(7,     description="How many days of recent demand to treat as 'open' orders")
    channel:   Optional[str]  = Field(None,  description="Filter by channel: online, retail, wholesale")


class ScoreOrderPriorityArgs(BaseModel):
    sku_id:        str   = Field(..., description="SKU to score")
    qty_needed:    float = Field(..., description="Units needed")
    health_status: str   = Field("healthy", description="Inventory health status for this SKU")
    days_of_supply: float = Field(30.0,  description="Current days of supply")
    channel:       str   = Field("online", description="Sales channel")


class RouteToBestDCArgs(BaseModel):
    sku_id:          str            = Field(..., description="SKU to route")
    qty_needed:      float          = Field(..., description="Units needed")
    preferred_dc:    Optional[str]  = Field(None, description="Preferred DC code — if None, best DC selected automatically")


class CreateDispatchArgs(BaseModel):
    sku_id:        str            = Field(...)
    location_id:   str            = Field("DC-01")
    qty_dispatched: float         = Field(..., description="Units dispatched")
    channel:       str            = Field("default")
    reference_doc: Optional[str]  = Field(None, description="Order reference number")
    notes:         Optional[str]  = Field(None)


class ManageBackorderArgs(BaseModel):
    sku_id:        str            = Field(...)
    qty_held:      float          = Field(..., description="Units that cannot be fulfilled now")
    reason:        str            = Field("Insufficient stock")
    channel:       str            = Field("default")
    reference_doc: Optional[str]  = Field(None)
    eta_days:      Optional[int]  = Field(None, description="Estimated days until stock arrives")


class CalculateFillRateArgs(BaseModel):
    run_id:   str = Field(..., description="Workflow run ID to calculate fill rate for")
    sku_ids:  Optional[list] = Field(None, description="Limit to specific SKUs — None = all")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_available_stock(sku_id: str, location_id: str = "DC-01") -> float:
    conn = sqlite3.connect(str(DB_PATH))
    row  = conn.execute(
        "SELECT on_hand, reserved FROM inventory_levels WHERE sku_id = ? AND location_id = ?",
        (sku_id.upper().strip(), location_id),
    ).fetchone()
    conn.close()
    if not row:
        return 0.0
    return max(0.0, row[0] - row[1])


def _get_all_locations() -> list[str]:
    conn  = sqlite3.connect(str(DB_PATH))
    rows  = conn.execute("SELECT DISTINCT location_id FROM inventory_levels").fetchall()
    conn.close()
    return [r[0] for r in rows] or ["DC-01"]


# ═══════════════════════════════════════════════════════════════════
# Tool 1 — get_open_orders
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=GetOpenOrdersArgs)
def get_open_orders(
    sku_ids:   Optional[list] = None,
    days_back: int            = 7,
    channel:   Optional[str]  = None,
) -> dict:
    """
    Load recent customer demand as 'open orders' requiring fulfillment.

    Aggregates the last `days_back` days of orders per SKU (and optionally
    per channel) from the orders table. These represent the demand that the
    Fulfillment Agent needs to satisfy from current on-hand stock.

    Returns a list of open order records sorted by total quantity descending.
    Each record includes sku_id, total_qty_needed, avg_daily, and channel breakdown.

    Use this as the starting point for every fulfillment run.
    """
    try:
        cutoff = str(date.today() - timedelta(days=days_back))
        conn   = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        query  = """
            SELECT sku_id,
                   channel,
                   SUM(quantity)  AS total_qty,
                   AVG(quantity)  AS avg_daily,
                   COUNT(*)       AS order_days
            FROM orders
            WHERE order_date >= ?
        """
        params: list = [cutoff]

        if sku_ids:
            ph     = ",".join("?" * len(sku_ids))
            query += f" AND sku_id IN ({ph})"
            params.extend([s.upper().strip() for s in sku_ids])

        if channel:
            query  += " AND channel = ?"
            params.append(channel)

        query += " GROUP BY sku_id, channel ORDER BY total_qty DESC"
        rows   = conn.execute(query, params).fetchall()
        conn.close()

        # Aggregate per SKU across channels
        sku_totals: dict[str, dict] = {}
        for row in rows:
            sid = row["sku_id"]
            if sid not in sku_totals:
                sku_totals[sid] = {
                    "sku_id":        sid,
                    "total_qty":     0.0,
                    "channels":      {},
                    "days_back":     days_back,
                }
            sku_totals[sid]["total_qty"]            += row["total_qty"]
            sku_totals[sid]["channels"][row["channel"]] = {
                "qty":        round(row["total_qty"], 1),
                "avg_daily":  round(row["avg_daily"], 2),
                "order_days": row["order_days"],
            }

        orders = sorted(sku_totals.values(), key=lambda x: x["total_qty"], reverse=True)
        return {
            "days_back":   days_back,
            "total_skus":  len(orders),
            "total_units": round(sum(o["total_qty"] for o in orders), 1),
            "orders":      orders,
        }

    except Exception as exc:
        return {"error": str(exc), "orders": []}


# ═══════════════════════════════════════════════════════════════════
# Tool 2 — score_order_priority
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=ScoreOrderPriorityArgs)
def score_order_priority(
    sku_id:         str,
    qty_needed:     float,
    health_status:  str   = "healthy",
    days_of_supply: float = 30.0,
    channel:        str   = "online",
) -> dict:
    """
    Compute a 0–100 priority score for fulfilling an order.

    Priority is determined by three components:
      1. Inventory health urgency  (50% weight)
         stock_out=50, critical=40, at_risk=25, healthy=10
      2. Days-of-supply urgency    (30% weight)
         <3d=30, <7d=24, <14d=18, <30d=9, ≥30d=0
      3. Channel premium           (20% weight)
         online=10, retail=7, wholesale=4, default=2

    Score 80–100: fulfil immediately
    Score 50–79:  fulfil this run
    Score 20–49:  fulfil if stock allows
    Score 0–19:   defer if needed

    Use this to sort a list of orders before executing fulfillment.
    """
    # Component 1: inventory urgency
    health_map = {
        "stock_out": 50,
        "critical":  40,
        "at_risk":   25,
        "healthy":   10,
    }
    urgency_score = health_map.get(health_status, 10)

    # Component 2: days of supply
    if days_of_supply <= 3:
        dos_score = 30
    elif days_of_supply <= 7:
        dos_score = 24
    elif days_of_supply <= 14:
        dos_score = 18
    elif days_of_supply < 30:
        dos_score = 9
    else:
        dos_score = 0

    # Component 3: channel premium
    channel_map = {
        "online":    10,
        "retail":    7,
        "wholesale": 4,
        "default":   2,
    }
    channel_score = channel_map.get(channel.lower(), 2)

    total_score = urgency_score + dos_score + channel_score

    # Priority tier
    if total_score >= 80:
        tier = "CRITICAL — fulfil immediately"
    elif total_score >= 50:
        tier = "HIGH — fulfil this run"
    elif total_score >= 20:
        tier = "MEDIUM — fulfil if stock allows"
    else:
        tier = "LOW — defer if needed"

    return {
        "sku_id":         sku_id,
        "priority_score": total_score,
        "priority_tier":  tier,
        "qty_needed":     qty_needed,
        "components": {
            "inventory_urgency": urgency_score,
            "dos_urgency":       dos_score,
            "channel_premium":   channel_score,
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Tool 3 — route_to_best_dc
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=RouteToBestDCArgs)
def route_to_best_dc(
    sku_id:       str,
    qty_needed:   float,
    preferred_dc: Optional[str] = None,
) -> dict:
    """
    Select the optimal warehouse/DC for fulfilling an order.

    Evaluates all locations with stock for this SKU and selects the best
    based on: (1) can fully satisfy qty_needed, (2) highest available stock.
    A preferred_dc can be specified to override automatic selection.

    Returns the selected DC, available quantity there, and whether full
    or partial fulfillment is possible from that location.

    Use this before create_dispatch to confirm which DC to ship from.
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """SELECT location_id, on_hand, reserved,
                      MAX(0, on_hand - reserved) AS available
               FROM inventory_levels
               WHERE sku_id = ? AND on_hand > 0
               ORDER BY (on_hand - reserved) DESC""",
            (sku_id.upper().strip(),),
        ).fetchall()
        conn.close()

        if not rows:
            return {
                "sku_id":      sku_id,
                "routable":    False,
                "reason":      "No stock at any location",
                "qty_needed":  qty_needed,
            }

        locations = [dict(r) for r in rows]

        # If preferred_dc specified, check it first
        if preferred_dc:
            for loc in locations:
                if loc["location_id"] == preferred_dc:
                    available = loc["available"]
                    return {
                        "sku_id":       sku_id,
                        "routable":     available > 0,
                        "dc":           preferred_dc,
                        "available":    available,
                        "qty_needed":   qty_needed,
                        "can_fully_fill": available >= qty_needed,
                        "shortfall":    max(0.0, qty_needed - available),
                    }

        # Auto-select: prefer DCs that can fully satisfy the order
        full_fill = [l for l in locations if l["available"] >= qty_needed]
        selected  = full_fill[0] if full_fill else locations[0]  # fallback: most stock

        available = selected["available"]
        return {
            "sku_id":         sku_id,
            "routable":       True,
            "dc":             selected["location_id"],
            "available":      available,
            "qty_needed":     qty_needed,
            "can_fully_fill": available >= qty_needed,
            "qty_to_dispatch": min(available, qty_needed),
            "shortfall":      max(0.0, qty_needed - available),
            "all_locations":  [
                {"dc": l["location_id"], "available": l["available"]}
                for l in locations
            ],
        }

    except Exception as exc:
        return {"sku_id": sku_id, "error": str(exc), "routable": False}


# ═══════════════════════════════════════════════════════════════════
# Tool 4 — create_dispatch
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=CreateDispatchArgs)
def create_dispatch(
    sku_id:         str,
    location_id:    str            = "DC-01",
    qty_dispatched: float          = 0.0,
    channel:        str            = "default",
    reference_doc:  Optional[str]  = None,
    notes:          Optional[str]  = None,
) -> dict:
    """
    Record a fulfillment dispatch — stock has been allocated and will ship.

    Persists a stock_movement of type 'sale' (which decreases on_hand)
    and returns the dispatch record with dispatch_id, updated stock balance,
    and dispatch_date.

    Call this after confirming that stock is available from route_to_best_dc.
    Do NOT call this for backorders — use manage_backorder instead.
    """
    try:
        dispatch_id = f"DISP-{uuid4().hex[:8].upper()}"
        ref_doc     = reference_doc or dispatch_id

        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA foreign_keys = ON")

        # Record as stock movement (sale type)
        conn.execute(
            """INSERT INTO stock_movements
               (movement_id, sku_id, location_id, movement_type, quantity,
                reference_doc, notes, recorded_by, occurred_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (dispatch_id, sku_id.upper().strip(), location_id, "sale",
             qty_dispatched, ref_doc, notes or f"Dispatch for {channel}",
             "fulfillment_agent", datetime.utcnow().isoformat()),
        )

        # Decrement on_hand
        conn.execute(
            "UPDATE inventory_levels SET on_hand = MAX(0, on_hand - ?), last_updated = ? "
            "WHERE sku_id = ? AND location_id = ?",
            (qty_dispatched, datetime.utcnow().isoformat(),
             sku_id.upper().strip(), location_id),
        )
        conn.commit()

        # Return updated balance
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT on_hand, reserved FROM inventory_levels WHERE sku_id = ? AND location_id = ?",
            (sku_id.upper().strip(), location_id),
        ).fetchone()
        conn.close()

        new_on_hand  = dict(row)["on_hand"] if row else 0.0
        new_reserved = dict(row)["reserved"] if row else 0.0

        return {
            "success":        True,
            "dispatch_id":    dispatch_id,
            "sku_id":         sku_id.upper().strip(),
            "location_id":    location_id,
            "qty_dispatched": qty_dispatched,
            "channel":        channel,
            "dispatch_date":  str(date.today()),
            "new_on_hand":    new_on_hand,
            "new_available":  max(0.0, new_on_hand - new_reserved),
            "reference_doc":  ref_doc,
        }

    except Exception as exc:
        return {"success": False, "sku_id": sku_id, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════
# Tool 5 — manage_backorder
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=ManageBackorderArgs)
def manage_backorder(
    sku_id:        str,
    qty_held:      float,
    reason:        str           = "Insufficient stock",
    channel:       str           = "default",
    reference_doc: Optional[str] = None,
    eta_days:      Optional[int] = None,
) -> dict:
    """
    Record an order that cannot be fulfilled now — a backorder.

    Backorders are created when available stock is less than demand.
    This tool records the shortfall in the stock_movements table as an
    'adjustment' note (no inventory change) and returns a backorder record
    with estimated fulfillment date based on open POs.

    Use this for every held/partial order to maintain an audit trail.
    The Procurement Agent's expected_date is used for ETA if not provided.
    """
    try:
        backorder_id = f"BO-{uuid4().hex[:8].upper()}"
        ref_doc      = reference_doc or backorder_id

        # Estimate ETA from open POs if not provided
        eta_date = None
        if eta_days is not None:
            eta_date = str(date.today() + timedelta(days=eta_days))
        else:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            po_row = conn.execute(
                """SELECT pol.sku_id, po.expected_date
                   FROM purchase_orders po
                   JOIN purchase_order_lines pol ON po.po_number = pol.po_number
                   WHERE pol.sku_id = ? AND po.status IN ('submitted','confirmed','shipped')
                   ORDER BY po.expected_date ASC LIMIT 1""",
                (sku_id.upper().strip(),),
            ).fetchone()
            conn.close()
            if po_row and po_row["expected_date"]:
                eta_date = po_row["expected_date"]
            else:
                eta_date = str(date.today() + timedelta(days=settings.default_lead_time_days))

        # Log as stock_movement note (no qty change)
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """INSERT INTO stock_movements
               (movement_id, sku_id, location_id, movement_type, quantity,
                reference_doc, notes, recorded_by, occurred_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (backorder_id, sku_id.upper().strip(), "DC-01", "adjustment",
             qty_held, ref_doc,
             f"BACKORDER: {reason} | ETA: {eta_date} | channel: {channel}",
             "fulfillment_agent", datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()

        return {
            "success":      True,
            "backorder_id": backorder_id,
            "sku_id":       sku_id.upper().strip(),
            "qty_held":     qty_held,
            "reason":       reason,
            "channel":      channel,
            "eta_date":     eta_date,
            "days_to_eta":  (date.fromisoformat(eta_date) - date.today()).days if eta_date else None,
            "status":       "backordered",
        }

    except Exception as exc:
        return {"success": False, "sku_id": sku_id, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════
# Tool 6 — calculate_fill_rate
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=CalculateFillRateArgs)
def calculate_fill_rate(run_id: str, sku_ids: Optional[list] = None) -> dict:
    """
    Calculate the order fill rate for a fulfillment run.

    Fill rate = (units dispatched) / (units demanded) × 100

    Reads dispatches (stock movements of type 'sale' recorded by
    fulfillment_agent during this run) and compares to total demand.

    Thresholds:
      ≥ 95% → excellent
      ≥ 85% → good
      ≥ 70% → acceptable
      < 70% → poor — triggers an exception event

    Returns:
      - overall_fill_rate_pct
      - per_sku breakdown
      - grade: excellent / good / acceptable / poor
    """
    try:
        conn  = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cutoff = str(date.today() - timedelta(days=1))

        # Dispatches this run (sales recorded by fulfillment_agent today)
        dispatch_query = """
            SELECT sku_id, SUM(quantity) AS dispatched
            FROM stock_movements
            WHERE movement_type = 'sale'
              AND recorded_by   = 'fulfillment_agent'
              AND occurred_at  >= ?
            GROUP BY sku_id
        """
        dispatch_params: list = [cutoff]
        if sku_ids:
            ph = ",".join("?" * len(sku_ids))
            dispatch_query += f" AND sku_id IN ({ph})"
            dispatch_params.extend([s.upper().strip() for s in sku_ids])

        dispatches = {r["sku_id"]: r["dispatched"]
                      for r in conn.execute(dispatch_query, dispatch_params).fetchall()}

        # Demand last 7 days
        demand_query = """
            SELECT sku_id, SUM(quantity) AS demanded
            FROM orders
            WHERE order_date >= ?
            GROUP BY sku_id
        """
        demand_params: list = [str(date.today() - timedelta(days=7))]
        if sku_ids:
            ph = ",".join("?" * len(sku_ids))
            demand_query += f" AND sku_id IN ({ph})"
            demand_params.extend([s.upper().strip() for s in sku_ids])

        demands = {r["sku_id"]: r["demanded"]
                   for r in conn.execute(demand_query, demand_params).fetchall()}
        conn.close()

        per_sku: list[dict] = []
        total_demanded  = 0.0
        total_dispatched = 0.0

        all_sku_ids = set(list(dispatches.keys()) + list(demands.keys()))
        if sku_ids:
            all_sku_ids = {s.upper().strip() for s in sku_ids}

        for sku_id in sorted(all_sku_ids):
            demanded   = demands.get(sku_id, 0.0)
            dispatched = dispatches.get(sku_id, 0.0)
            fill_rate  = round(dispatched / demanded * 100, 1) if demanded > 0 else 100.0
            total_demanded   += demanded
            total_dispatched += dispatched
            per_sku.append({
                "sku_id":    sku_id,
                "demanded":  round(demanded, 1),
                "dispatched": round(dispatched, 1),
                "fill_rate_pct": fill_rate,
            })

        overall = round(total_dispatched / total_demanded * 100, 1) if total_demanded > 0 else 100.0
        grade   = ("excellent"  if overall >= 95 else
                   "good"       if overall >= 85 else
                   "acceptable" if overall >= 70 else "poor")

        return {
            "run_id":               run_id,
            "overall_fill_rate_pct": overall,
            "grade":                grade,
            "total_demanded":       round(total_demanded, 1),
            "total_dispatched":     round(total_dispatched, 1),
            "per_sku":              per_sku,
            "computed_at":          datetime.utcnow().isoformat(),
        }

    except Exception as exc:
        return {"run_id": run_id, "error": str(exc), "overall_fill_rate_pct": None}


# ── Tool registry ─────────────────────────────────────────────────────────────

FULFILLMENT_TOOLS = [
    get_open_orders,
    score_order_priority,
    route_to_best_dc,
    create_dispatch,
    manage_backorder,
    calculate_fill_rate,
]
