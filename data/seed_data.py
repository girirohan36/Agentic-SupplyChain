"""
data/seed_data.py
─────────────────
Faker-powered mock ERP data generator.

Generates:
  - 20 SKUs across 5 categories
  - 5 Suppliers with varied lead times & reliability
  - SKU ↔ Supplier mappings
  - 180 days of daily demand history (with trend + seasonality)
  - Current inventory levels per SKU
  - Stock movement ledger (last 30 days)
  - Sample CSV exports → data/sample_data/

Run directly:
    python data/seed_data.py

Safe to re-run — clears and re-seeds all tables.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

# ── Reproducible randomness ────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent
SAMPLE_DIR      = DATA_DIR / "sample_data"
DB_PATH         = DATA_DIR / "supply_demand.db"
SCHEMA_PATH     = DATA_DIR / "schema.sql"

SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
TODAY           = date.today()
HISTORY_DAYS    = 180
START_DATE      = TODAY - timedelta(days=HISTORY_DAYS)
LOCATION_ID     = "DC-01"


# ═══════════════════════════════════════════════════════════════════
# Master Data
# ═══════════════════════════════════════════════════════════════════

SKUS = [
    # (sku_id, name, category, uom, unit_cost, unit_price, weight_kg, storage_type, shelf_life_days)
    ("SKU-001", "Wireless Bluetooth Headphones",   "Electronics",  "EA", 45.00, 99.99,  0.35, "ambient",      None),
    ("SKU-002", "USB-C Charging Cable 2m",         "Electronics",  "EA",  3.50,  9.99,  0.08, "ambient",      None),
    ("SKU-003", "Laptop Stand Adjustable",         "Electronics",  "EA", 18.00, 49.99,  0.90, "ambient",      None),
    ("SKU-004", "Mechanical Keyboard TKL",         "Electronics",  "EA", 55.00,129.99,  0.75, "ambient",      None),
    ("SKU-005", "Ergonomic Mouse Wireless",        "Electronics",  "EA", 22.00, 59.99,  0.12, "ambient",      None),
    ("SKU-006", "Running Shoes Men Size 10",       "Apparel",      "PR", 38.00, 89.99,  0.65, "ambient",      None),
    ("SKU-007", "Yoga Pants Women M",              "Apparel",      "EA", 14.00, 39.99,  0.22, "ambient",      None),
    ("SKU-008", "Compression Socks Pack 3",        "Apparel",      "PK",  6.50, 18.99,  0.15, "ambient",      None),
    ("SKU-009", "Winter Jacket Men L",             "Apparel",      "EA", 65.00,159.99,  0.90, "ambient",      None),
    ("SKU-010", "Baseball Cap Adjustable",         "Apparel",      "EA",  8.00, 24.99,  0.12, "ambient",      None),
    ("SKU-011", "Whey Protein Powder 2kg",         "Nutrition",    "EA", 28.00, 59.99,  2.10, "ambient",      365),
    ("SKU-012", "Energy Bars Box of 12",           "Nutrition",    "BX",  9.00, 22.99,  0.72, "ambient",      180),
    ("SKU-013", "Multivitamin Tablets 90ct",       "Nutrition",    "EA",  7.50, 19.99,  0.18, "ambient",      730),
    ("SKU-014", "Pre-Workout Supplement 300g",     "Nutrition",    "EA", 22.00, 44.99,  0.32, "ambient",      365),
    ("SKU-015", "Organic Green Tea 100 bags",      "Nutrition",    "BX",  5.00, 14.99,  0.20, "ambient",      540),
    ("SKU-016", "Foam Roller High Density",        "Fitness",      "EA", 12.00, 34.99,  0.55, "ambient",      None),
    ("SKU-017", "Resistance Bands Set 5pc",        "Fitness",      "ST",  8.50, 24.99,  0.30, "ambient",      None),
    ("SKU-018", "Jump Rope Speed Cable",           "Fitness",      "EA",  6.00, 18.99,  0.18, "ambient",      None),
    ("SKU-019", "Yoga Mat Non-Slip 6mm",           "Fitness",      "EA", 14.00, 39.99,  1.10, "ambient",      None),
    ("SKU-020", "Pull-Up Bar Doorframe",           "Fitness",      "EA", 20.00, 54.99,  1.20, "ambient",      None),
]

SUPPLIERS = [
    # (supplier_id, name, country, lead_time_days, min_order_qty, unit_cost_multiplier, reliability, status, email)
    ("SUP-001", "TechSource Global Ltd",       "China",        14,  50, 0.90, 0.97, "preferred", "orders@techsource.example.com"),
    ("SUP-002", "FastShip Domestic LLC",       "USA",           3, 100, 1.10, 0.99, "preferred", "supply@fastship.example.com"),
    ("SUP-003", "EuroTrade Wholesale GmbH",    "Germany",      21,  25, 0.95, 0.92, "active",    "procurement@eurotrade.example.com"),
    ("SUP-004", "QuickFulfill Partners Inc",   "USA",           5, 200, 1.05, 0.95, "active",    "ops@quickfulfill.example.com"),
    ("SUP-005", "Pacific Rim Distributors",   "Vietnam",       28,  30, 0.82, 0.88, "active",    "sales@pacificrim.example.com"),
]

# SKU → primary supplier mapping
SKU_SUPPLIER_MAP = {
    "SKU-001": ("SUP-001", "SUP-003"),
    "SKU-002": ("SUP-001", "SUP-004"),
    "SKU-003": ("SUP-001", "SUP-005"),
    "SKU-004": ("SUP-001", "SUP-003"),
    "SKU-005": ("SUP-001", "SUP-004"),
    "SKU-006": ("SUP-005", "SUP-003"),
    "SKU-007": ("SUP-005", "SUP-004"),
    "SKU-008": ("SUP-005", "SUP-002"),
    "SKU-009": ("SUP-003", "SUP-005"),
    "SKU-010": ("SUP-005", "SUP-004"),
    "SKU-011": ("SUP-004", "SUP-002"),
    "SKU-012": ("SUP-002", "SUP-004"),
    "SKU-013": ("SUP-004", "SUP-002"),
    "SKU-014": ("SUP-004", "SUP-001"),
    "SKU-015": ("SUP-005", "SUP-003"),
    "SKU-016": ("SUP-004", "SUP-002"),
    "SKU-017": ("SUP-005", "SUP-004"),
    "SKU-018": ("SUP-005", "SUP-002"),
    "SKU-019": ("SUP-005", "SUP-004"),
    "SKU-020": ("SUP-001", "SUP-004"),
}

# Demand profile per SKU: (base_daily_demand, trend_pct_per_month, weekly_peak_day, seasonal_peak_month)
# seasonal_peak_month: month number where demand peaks (1-12), None = no seasonality
DEMAND_PROFILES = {
    "SKU-001": (25,  +0.05,  5,  11),   # headphones — trend up, Black Friday peak
    "SKU-002": (80,  +0.02,  3,  None),  # cables — steady high volume
    "SKU-003": (12,  +0.08,  2,  1),    # laptop stands — WFH trend, Jan peak
    "SKU-004": (8,   +0.03,  5,  11),   # keyboards — gaming, holiday peak
    "SKU-005": (15,  +0.04,  3,  None),  # mouse — steady
    "SKU-006": (18,  -0.01,  6,  3),    # running shoes — spring peak
    "SKU-007": (22,  +0.06,  6,  1),    # yoga pants — Jan resolution peak
    "SKU-008": (35,  +0.01,  3,  None),  # socks — steady
    "SKU-009": (10,  +0.02,  5,  10),   # winter jacket — Oct/Nov peak
    "SKU-010": (20,  -0.01,  6,  5),    # cap — summer peak
    "SKU-011": (30,  +0.04,  2,  1),    # protein — Jan resolution
    "SKU-012": (45,  +0.02,  2,  None),  # energy bars — steady
    "SKU-013": (40,  +0.01,  2,  1),    # vitamins — Jan peak
    "SKU-014": (20,  +0.05,  2,  1),    # pre-workout — Jan resolution
    "SKU-015": (25,  +0.01,  3,  None),  # green tea — steady
    "SKU-016": (14,  +0.03,  6,  1),    # foam roller — Jan peak
    "SKU-017": (18,  +0.04,  6,  1),    # resistance bands
    "SKU-018": (12,  +0.02,  6,  4),    # jump rope — spring
    "SKU-019": (16,  +0.05,  6,  1),    # yoga mat — Jan peak
    "SKU-020": (9,   +0.03,  6,  1),    # pull-up bar — Jan peak
}


# ═══════════════════════════════════════════════════════════════════
# Demand Generator
# ═══════════════════════════════════════════════════════════════════

def generate_daily_demand(
    sku_id: str,
    start: date,
    days: int,
) -> list[tuple]:
    """
    Generate realistic daily demand with:
      - base demand + random noise (±25%)
      - linear monthly trend
      - day-of-week seasonality (weekends spike for fitness/apparel)
      - annual seasonal peak (e.g. Jan for fitness, Nov for electronics)
    Returns list of (order_date, sku_id, quantity, channel, region) tuples.
    """
    profile = DEMAND_PROFILES.get(sku_id, (20, 0.0, 3, None))
    base, trend_pm, peak_dow, peak_month = profile

    channels = ["online", "retail", "wholesale"]
    regions  = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
    rows = []

    for i in range(days):
        d = start + timedelta(days=i)

        # 1. Trend component (compound monthly)
        months_elapsed = i / 30.0
        trend_factor   = (1 + trend_pm) ** months_elapsed

        # 2. Day-of-week component
        dow = d.weekday()   # 0=Mon … 6=Sun
        if dow == peak_dow:
            dow_factor = 1.35
        elif dow in (5, 6):  # weekend
            dow_factor = 1.15
        else:
            dow_factor = 0.95

        # 3. Annual seasonal component (sinusoidal peak at peak_month)
        if peak_month:
            day_of_year   = d.timetuple().tm_yday
            peak_doy      = (peak_month - 1) * 30 + 15
            seasonal_factor = 1 + 0.40 * math.cos(
                2 * math.pi * (day_of_year - peak_doy) / 365
            )
        else:
            seasonal_factor = 1.0

        # 4. Noise ±25%
        noise = random.uniform(0.75, 1.25)

        qty = max(0.0, round(base * trend_factor * dow_factor * seasonal_factor * noise, 1))

        channel = random.choices(channels, weights=[0.5, 0.35, 0.15])[0]
        region  = random.choice(regions)

        rows.append((str(d), sku_id, qty, channel, region))

    return rows


# ═══════════════════════════════════════════════════════════════════
# Inventory Level Generator
# ═══════════════════════════════════════════════════════════════════

def compute_inventory_levels(demand_rows: dict[str, list]) -> list[dict]:
    """
    Derive plausible current inventory levels from demand history.
    Uses the last 30 days of demand to compute:
      - average daily demand → reorder point & safety stock
      - random on-hand (some healthy, some low to show agent action)
    """
    records = []
    for sku in SKUS:
        sku_id = sku[0]
        profile  = DEMAND_PROFILES.get(sku_id, (20, 0.0, 3, None))
        base_demand = profile[0]

        # Get last 30 days avg
        recent = demand_rows.get(sku_id, [])[-30:]
        avg_daily = (
            sum(r[2] for r in recent) / len(recent) if recent else base_demand
        )

        lead_time = random.choice([3, 5, 7, 14, 21, 28])
        safety_stock = round(avg_daily * 14, 1)          # 14-day buffer
        reorder_pt   = round(avg_daily * lead_time + safety_stock, 1)

        # EOQ approximation: sqrt(2 * annual_demand * order_cost / holding_cost)
        annual_demand  = avg_daily * 365
        order_cost     = random.uniform(50, 150)
        holding_cost   = sku[4] * 0.25                  # 25% of unit cost/year
        eoq = round(math.sqrt(2 * annual_demand * order_cost / holding_cost), 1)

        # Vary on-hand: 20% chance critical, 20% at-risk, 60% healthy
        health_roll = random.random()
        if health_roll < 0.20:      # critical / near stock-out
            on_hand = round(random.uniform(0, safety_stock * 0.4), 1)
        elif health_roll < 0.40:    # at-risk
            on_hand = round(random.uniform(safety_stock * 0.4, reorder_pt * 0.85), 1)
        else:                       # healthy
            on_hand = round(random.uniform(reorder_pt, reorder_pt * 2.5), 1)

        reserved   = round(on_hand * random.uniform(0.05, 0.20), 1)
        in_transit = round(eoq * random.uniform(0, 0.5), 1) if health_roll < 0.40 else 0.0

        records.append({
            "sku_id":       sku_id,
            "location_id":  LOCATION_ID,
            "on_hand":      on_hand,
            "reserved":     reserved,
            "in_transit":   in_transit,
            "reorder_point": reorder_pt,
            "safety_stock":  safety_stock,
            "eoq":           eoq,
            "last_updated":  datetime.utcnow().isoformat(),
        })
    return records


# ═══════════════════════════════════════════════════════════════════
# Stock Movement Generator (last 30 days)
# ═══════════════════════════════════════════════════════════════════

def generate_movements(demand_rows: dict[str, list]) -> list[tuple]:
    """Generate a realistic 30-day stock movement ledger."""
    movement_types_inbound  = ["receipt", "return"]
    movement_types_outbound = ["sale", "write_off"]
    rows = []
    for sku_id, daily in demand_rows.items():
        last_30 = daily[-30:]
        for order_date, _, qty, channel, _ in last_30:
            # Sales movement
            if qty > 0:
                rows.append((
                    sku_id, LOCATION_ID, "sale",
                    round(qty, 1), f"SO-{sku_id}-{order_date}",
                    "inventory_agent", order_date,
                ))
            # Occasional receipt (every ~7 days)
            if random.random() < (1 / 7):
                receipt_qty = round(random.uniform(50, 300), 1)
                rows.append((
                    sku_id, LOCATION_ID, "receipt",
                    receipt_qty, f"PO-SEED-{sku_id}",
                    "inventory_agent", order_date,
                ))
    return rows


# ═══════════════════════════════════════════════════════════════════
# Database Seeder
# ═══════════════════════════════════════════════════════════════════

def seed(db_path: Path = DB_PATH) -> None:
    print("🌱 Starting seed …")

    # ── Init schema ──────────────────────────────────────────────────
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    schema_sql = SCHEMA_PATH.read_text()
    # Smart split: respect parenthesis depth so we never split
    # inside a CREATE TABLE ( ... ) body
    statements: list[str] = []
    depth, buf = 0, []
    for ch in schema_sql:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == ";" and depth == 0:
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(ch)
    if leftover := "".join(buf).strip():
        statements.append(leftover)
    for stmt in statements:
        conn.execute(stmt)
    conn.commit()
    print("  ✓ Schema applied")

    # ── Clear existing data (safe re-seed) ───────────────────────────
    for tbl in [
        "exception_events", "agent_results", "workflow_runs",
        "purchase_order_lines", "purchase_orders",
        "stock_movements", "orders", "inventory_levels",
        "sku_suppliers", "suppliers", "skus",
    ]:
        conn.execute(f"DELETE FROM {tbl}")
    conn.commit()
    print("  ✓ Existing data cleared")

    # ── Insert SKUs ──────────────────────────────────────────────────
    conn.executemany(
        """INSERT INTO skus
           (sku_id,name,category,unit_of_measure,unit_cost,unit_price,
            weight_kg,storage_type,shelf_life_days,status)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        [(s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8],"active") for s in SKUS],
    )
    conn.commit()
    print(f"  ✓ {len(SKUS)} SKUs inserted")

    # ── Insert Suppliers ─────────────────────────────────────────────
    supplier_rows = []
    for s in SUPPLIERS:
        # Assign a representative unit cost (avg of SKU costs for simplicity)
        avg_cost = round(
            sum(sku[4] for sku in SKUS) / len(SKUS) * s[5], 2
        )
        supplier_rows.append((
            s[0], s[1], s[2], s[3], s[4], avg_cost, s[6], s[7], s[8]
        ))
    conn.executemany(
        """INSERT INTO suppliers
           (supplier_id,name,country,lead_time_days,min_order_qty,
            unit_cost,reliability_score,status,contact_email)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        supplier_rows,
    )
    conn.commit()
    print(f"  ✓ {len(SUPPLIERS)} suppliers inserted")

    # ── Insert SKU ↔ Supplier mappings ───────────────────────────────
    mapping_rows = []
    for sku_id, (primary, secondary) in SKU_SUPPLIER_MAP.items():
        mapping_rows.append((sku_id, primary,   1, None))
        mapping_rows.append((sku_id, secondary, 0, None))
    conn.executemany(
        "INSERT INTO sku_suppliers (sku_id,supplier_id,is_primary,contracted_cost) VALUES (?,?,?,?)",
        mapping_rows,
    )
    conn.commit()
    print(f"  ✓ {len(mapping_rows)} SKU-Supplier mappings inserted")

    # ── Generate & insert demand history ────────────────────────────
    all_demand: dict[str, list] = {}
    all_order_rows = []
    for sku in SKUS:
        sku_id = sku[0]
        rows   = generate_daily_demand(sku_id, START_DATE, HISTORY_DAYS)
        all_demand[sku_id] = rows
        all_order_rows.extend(rows)

    conn.executemany(
        "INSERT OR IGNORE INTO orders (order_date,sku_id,quantity,channel,region) VALUES (?,?,?,?,?)",
        all_order_rows,
    )
    conn.commit()
    print(f"  ✓ {len(all_order_rows):,} demand history rows inserted ({HISTORY_DAYS} days × {len(SKUS)} SKUs)")

    # ── Generate & insert inventory levels ───────────────────────────
    inv_records = compute_inventory_levels(all_demand)
    conn.executemany(
        """INSERT INTO inventory_levels
           (sku_id,location_id,on_hand,reserved,in_transit,
            reorder_point,safety_stock,eoq,last_updated)
           VALUES (:sku_id,:location_id,:on_hand,:reserved,:in_transit,
                   :reorder_point,:safety_stock,:eoq,:last_updated)""",
        inv_records,
    )
    conn.commit()
    print(f"  ✓ {len(inv_records)} inventory level records inserted")

    # ── Generate & insert stock movements ────────────────────────────
    movement_rows = generate_movements(all_demand)
    conn.executemany(
        """INSERT INTO stock_movements
           (sku_id,location_id,movement_type,quantity,reference_doc,recorded_by,occurred_at)
           VALUES (?,?,?,?,?,?,?)""",
        movement_rows,
    )
    conn.commit()
    print(f"  ✓ {len(movement_rows):,} stock movement records inserted")

    conn.close()

    # ── Export CSVs ──────────────────────────────────────────────────
    _export_csvs(all_demand, inv_records, supplier_rows)
    print(f"\n✅ Seed complete — database: {db_path}")


# ═══════════════════════════════════════════════════════════════════
# CSV Exporters
# ═══════════════════════════════════════════════════════════════════

def _export_csvs(
    demand: dict[str, list],
    inventory: list[dict],
    suppliers: list[tuple],
) -> None:
    """Write the three sample CSV files used as demo data."""

    # ── orders.csv ───────────────────────────────────────────────────
    orders_path = SAMPLE_DIR / "orders.csv"
    with open(orders_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order_date", "sku_id", "quantity", "channel", "region"])
        for rows in demand.values():
            writer.writerows(rows)
    print(f"  ✓ Exported {orders_path.name}")

    # ── inventory.csv ─────────────────────────────────────────────────
    inv_path = SAMPLE_DIR / "inventory.csv"
    with open(inv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sku_id","location_id","on_hand","reserved","in_transit",
            "reorder_point","safety_stock","eoq","last_updated",
        ])
        writer.writeheader()
        writer.writerows(inventory)
    print(f"  ✓ Exported {inv_path.name}")

    # ── suppliers.csv ─────────────────────────────────────────────────
    sup_path = SAMPLE_DIR / "suppliers.csv"
    with open(sup_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "supplier_id","name","country","lead_time_days",
            "min_order_qty","unit_cost","reliability_score","status","contact_email",
        ])
        writer.writerows(suppliers)
    print(f"  ✓ Exported {sup_path.name}")


# ═══════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    seed()
