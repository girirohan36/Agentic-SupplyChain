"""
api/routers/dashboard.py
─────────────────────────
Dashboard router — read-only data endpoints for the Streamlit UI.

Endpoints:
  GET /dashboard/kpis                 → warehouse-wide KPI summary
  GET /dashboard/inventory            → all SKU health records
  GET /dashboard/inventory/{sku_id}   → single SKU detail
  GET /dashboard/exceptions           → recent unresolved exceptions
  GET /dashboard/purchase-orders      → open PO tracker
  GET /dashboard/agents/{run_id}      → per-agent result breakdown
  GET /dashboard/fill-rate            → recent fill rate trend
  GET /dashboard/demand/{sku_id}      → demand history for a SKU
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Query

from api.dependencies import PaginationDep, RawConnDep, SettingsDep

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


# ════════════════════════════════════════════════════════════════════
# GET /dashboard/kpis
# ════════════════════════════════════════════════════════════════════

@router.get(
    "/kpis",
    summary="Warehouse-wide KPI summary",
    description="Returns inventory health metrics, recent run stats, and PO summary in one call.",
)
async def get_kpis(conn: RawConnDep) -> dict:
    # ── Inventory KPIs ────────────────────────────────────────────────────────
    inv_rows = conn.execute(
        """SELECT il.sku_id, il.on_hand, il.reserved, il.in_transit,
                  il.reorder_point, il.safety_stock, s.unit_cost
           FROM inventory_levels il
           JOIN skus s ON il.sku_id = s.sku_id
           WHERE il.location_id = 'DC-01'"""
    ).fetchall()

    total_skus        = len(inv_rows)
    stockout_count    = 0
    reorder_count     = 0
    overstock_count   = 0
    total_inv_value   = 0.0
    on_hand_totals    = []

    for row in inv_rows:
        r       = dict(row)
        on_hand = r["on_hand"]
        rop     = r.get("reorder_point") or 50.0
        intrans = r.get("in_transit", 0.0)
        total_inv_value += on_hand * r.get("unit_cost", 10.0)
        on_hand_totals.append(on_hand)
        if on_hand <= 0:
            stockout_count += 1
        if (on_hand + intrans) <= rop:
            reorder_count += 1
        if on_hand > rop * 2:
            overstock_count += 1

    fill_rate = round((total_skus - stockout_count) / total_skus * 100, 1) if total_skus > 0 else 100.0

    # ── Recent runs ───────────────────────────────────────────────────────────
    runs_today = conn.execute(
        "SELECT COUNT(*) FROM workflow_runs WHERE created_at >= date('now')"
    ).fetchone()[0]

    last_run = conn.execute(
        "SELECT run_id, status, created_at, completed_at FROM workflow_runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()

    # ── Open POs ──────────────────────────────────────────────────────────────
    po_stats = conn.execute(
        """SELECT COUNT(*) AS cnt, SUM(total_value) AS val
           FROM purchase_orders WHERE status IN ('draft','submitted','confirmed')"""
    ).fetchone()

    # ── Unresolved exceptions ─────────────────────────────────────────────────
    unresolved_exceptions = conn.execute(
        "SELECT COUNT(*) FROM exception_events WHERE resolved = 0"
    ).fetchone()[0]

    critical_exceptions = conn.execute(
        "SELECT COUNT(*) FROM exception_events WHERE resolved = 0 AND severity = 'critical'"
    ).fetchone()[0]

    return {
        "inventory": {
            "total_skus":          total_skus,
            "stockout_count":      stockout_count,
            "reorder_trigger_count": reorder_count,
            "overstock_count":     overstock_count,
            "total_inventory_value": round(total_inv_value, 2),
            "fill_rate_pct":       fill_rate,
        },
        "runs": {
            "runs_today":  runs_today,
            "last_run_id":     dict(last_run)["run_id"]     if last_run else None,
            "last_run_status": dict(last_run)["status"]     if last_run else None,
            "last_run_at":     dict(last_run)["created_at"] if last_run else None,
        },
        "purchase_orders": {
            "open_count":       po_stats[0] or 0,
            "open_total_value": round(po_stats[1] or 0, 2),
        },
        "exceptions": {
            "unresolved_count": unresolved_exceptions,
            "critical_count":   critical_exceptions,
        },
        "generated_at": str(date.today()),
    }


# ════════════════════════════════════════════════════════════════════
# GET /dashboard/inventory
# ════════════════════════════════════════════════════════════════════

@router.get(
    "/inventory",
    summary="All SKU inventory health records",
)
async def get_inventory(
    conn:     RawConnDep,
    location: str = Query("DC-01", description="Warehouse location"),
    status_filter: Optional[str] = Query(None, description="Filter: healthy|at_risk|critical|stock_out"),
) -> dict:
    rows = conn.execute(
        """SELECT il.*, s.name, s.category, s.unit_cost, s.unit_price
           FROM inventory_levels il
           JOIN skus s ON il.sku_id = s.sku_id
           WHERE il.location_id = ?
           ORDER BY il.on_hand ASC""",
        (location,),
    ).fetchall()

    records = []
    for row in rows:
        r         = dict(row)
        on_hand   = r["on_hand"]
        reserved  = r.get("reserved", 0.0)
        in_transit = r.get("in_transit", 0.0)
        rop       = r.get("reorder_point") or 50.0
        ss        = r.get("safety_stock") or 20.0
        available = max(0.0, on_hand - reserved)

        # Quick health classification
        if on_hand <= 0:
            health_status = "stock_out"
        elif available < ss:
            health_status = "critical"
        elif (on_hand + in_transit) <= rop:
            health_status = "at_risk"
        else:
            health_status = "healthy"

        if status_filter and health_status != status_filter:
            continue

        records.append({
            **r,
            "available":    available,
            "total_supply": on_hand + in_transit,
            "health_status": health_status,
            "reorder_triggered": (on_hand + in_transit) <= rop,
            "overstock":    on_hand > rop * 2,
        })

    return {
        "location":    location,
        "total":       len(records),
        "records":     records,
    }


# ════════════════════════════════════════════════════════════════════
# GET /dashboard/inventory/{sku_id}
# ════════════════════════════════════════════════════════════════════

@router.get(
    "/inventory/{sku_id}",
    summary="Single SKU detail with recent movements",
)
async def get_sku_detail(sku_id: str, conn: RawConnDep) -> dict:
    sku_id = sku_id.upper().strip()

    sku = conn.execute("SELECT * FROM skus WHERE sku_id = ?", (sku_id,)).fetchone()
    if not sku:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' not found")

    inv = conn.execute(
        "SELECT * FROM inventory_levels WHERE sku_id = ? AND location_id = 'DC-01'",
        (sku_id,),
    ).fetchone()

    movements = conn.execute(
        """SELECT movement_type, quantity, occurred_at, reference_doc, notes
           FROM stock_movements WHERE sku_id = ?
           ORDER BY occurred_at DESC LIMIT 20""",
        (sku_id,),
    ).fetchall()

    demand_30d = conn.execute(
        """SELECT order_date, SUM(quantity) AS qty, channel
           FROM orders
           WHERE sku_id = ? AND order_date >= date('now','-30 days')
           GROUP BY order_date, channel
           ORDER BY order_date""",
        (sku_id,),
    ).fetchall()

    open_pos = conn.execute(
        """SELECT po.po_number, po.status, po.expected_date, pol.quantity, pol.unit_cost
           FROM purchase_orders po
           JOIN purchase_order_lines pol ON po.po_number = pol.po_number
           WHERE pol.sku_id = ? AND po.status IN ('draft','submitted','confirmed','shipped')
           ORDER BY po.issued_date DESC""",
        (sku_id,),
    ).fetchall()

    return {
        "sku":        dict(sku),
        "inventory":  dict(inv) if inv else None,
        "movements":  [dict(m) for m in movements],
        "demand_30d": [dict(d) for d in demand_30d],
        "open_pos":   [dict(p) for p in open_pos],
    }


# ════════════════════════════════════════════════════════════════════
# GET /dashboard/exceptions
# ════════════════════════════════════════════════════════════════════

@router.get(
    "/exceptions",
    summary="Recent exception events",
)
async def get_exceptions(
    conn:      RawConnDep,
    resolved:  Optional[bool] = Query(None, description="Filter by resolved status"),
    severity:  Optional[str]  = Query(None, description="Filter: info|warning|high|critical"),
    limit:     int             = Query(50, ge=1, le=200),
) -> dict:
    query  = "SELECT * FROM exception_events WHERE 1=1"
    params: list = []

    if resolved is not None:
        query  += " AND resolved = ?"
        params.append(1 if resolved else 0)
    if severity:
        query  += " AND severity = ?"
        params.append(severity)

    query += " ORDER BY raised_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    events = []
    for row in rows:
        r = dict(row)
        try:
            r["context"] = json.loads(r.get("context") or "{}")
        except Exception:
            pass
        events.append(r)

    severity_counts = {}
    for e in events:
        s = e.get("severity", "unknown")
        severity_counts[s] = severity_counts.get(s, 0) + 1

    return {
        "total":            len(events),
        "severity_counts":  severity_counts,
        "events":           events,
    }


# ════════════════════════════════════════════════════════════════════
# GET /dashboard/purchase-orders
# ════════════════════════════════════════════════════════════════════

@router.get(
    "/purchase-orders",
    summary="Open PO tracker with line items",
)
async def get_purchase_orders(
    conn:          RawConnDep,
    status_filter: Optional[str] = Query(None, description="Filter by PO status"),
    limit:         int            = Query(50),
) -> dict:
    query  = """
        SELECT po.*, s.name AS supplier_name
        FROM purchase_orders po
        LEFT JOIN suppliers s ON po.supplier_id = s.supplier_id
        WHERE 1=1
    """
    params: list = []

    if status_filter:
        query  += " AND po.status = ?"
        params.append(status_filter)
    else:
        query += " AND po.status IN ('draft','submitted','confirmed','shipped')"

    query += " ORDER BY po.issued_date DESC LIMIT ?"
    params.append(limit)

    po_rows = conn.execute(query, params).fetchall()

    pos = []
    for po_row in po_rows:
        po = dict(po_row)
        lines = conn.execute(
            "SELECT * FROM purchase_order_lines WHERE po_number = ?",
            (po["po_number"],),
        ).fetchall()
        po["lines"]       = [dict(l) for l in lines]
        po["sku_ids"]     = [l["sku_id"] for l in lines]
        pos.append(po)

    status_counts: dict[str, int] = {}
    total_value = 0.0
    for po in pos:
        s = po.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
        total_value += po.get("total_value") or 0.0

    return {
        "total":          len(pos),
        "total_value":    round(total_value, 2),
        "status_counts":  status_counts,
        "purchase_orders": pos,
    }


# ════════════════════════════════════════════════════════════════════
# GET /dashboard/agents/{run_id}
# ════════════════════════════════════════════════════════════════════

@router.get(
    "/agents/{run_id}",
    summary="Per-agent result breakdown for a run",
)
async def get_agent_results(run_id: str, conn: RawConnDep) -> dict:
    run = conn.execute(
        "SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,)
    ).fetchone()

    agents = conn.execute(
        "SELECT * FROM agent_results WHERE run_id = ? ORDER BY rowid",
        (run_id,),
    ).fetchall()

    agent_list = []
    for a in agents:
        ar = dict(a)
        try:
            ar["output"] = json.loads(ar.get("output") or "{}")
        except Exception:
            pass
        # Compute duration
        if ar.get("started_at") and ar.get("completed_at"):
            from datetime import datetime
            try:
                s = datetime.fromisoformat(ar["started_at"])
                c = datetime.fromisoformat(ar["completed_at"])
                ar["duration_seconds"] = round((c - s).total_seconds(), 2)
            except Exception:
                pass
        agent_list.append(ar)

    return {
        "run_id":        run_id,
        "run_status":    dict(run)["status"] if run else "unknown",
        "agent_results": agent_list,
        "total_agents":  len(agent_list),
        "success_count": sum(1 for a in agent_list if a.get("status") == "success"),
        "failed_count":  sum(1 for a in agent_list if a.get("status") == "failed"),
    }


# ════════════════════════════════════════════════════════════════════
# GET /dashboard/demand/{sku_id}
# ════════════════════════════════════════════════════════════════════

@router.get(
    "/demand/{sku_id}",
    summary="Demand history for a SKU (last N days)",
)
async def get_demand_history(
    sku_id:   str,
    conn:     RawConnDep,
    days:     int = Query(30, ge=1, le=180),
    channel:  Optional[str] = Query(None),
) -> dict:
    sku_id = sku_id.upper().strip()
    cutoff = str(date.today() - timedelta(days=days))

    query  = """
        SELECT order_date, SUM(quantity) AS total_qty, channel, region
        FROM orders
        WHERE sku_id = ? AND order_date >= ?
    """
    params: list = [sku_id, cutoff]

    if channel:
        query  += " AND channel = ?"
        params.append(channel)

    query += " GROUP BY order_date, channel ORDER BY order_date"

    rows = conn.execute(query, params).fetchall()

    # Daily totals across all channels
    daily: dict[str, float] = {}
    for row in rows:
        d   = row["order_date"]
        qty = row["total_qty"]
        daily[d] = daily.get(d, 0.0) + qty

    sorted_daily = [{"date": d, "qty": round(q, 1)} for d, q in sorted(daily.items())]
    total_qty    = round(sum(d["qty"] for d in sorted_daily), 1)
    avg_daily    = round(total_qty / len(sorted_daily), 2) if sorted_daily else 0.0

    return {
        "sku_id":      sku_id,
        "days":        days,
        "total_qty":   total_qty,
        "avg_daily":   avg_daily,
        "data_points": len(sorted_daily),
        "daily":       sorted_daily,
        "raw_records": [dict(r) for r in rows],
    }
