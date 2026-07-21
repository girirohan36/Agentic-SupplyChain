"""
tools/__init__.py  (Phase 5 — updated with fulfillment tools)
──────────────────────────────────────────────────────────────
Re-export all LangChain tools and registries.

    from tools import ALL_TOOLS
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
"""

# ── Forecast ───────────────────────────────────────────────────────────────
from tools.forecast_tools import (
    FORECAST_TOOLS,
    compute_mape,
    detect_demand_anomalies,
    get_demand_trend,
    load_demand_history,
    run_prophet_forecast,
)

# ── Inventory (Phase 3 + Phase 4) ─────────────────────────────────────────
from tools.inventory_tools import (
    INVENTORY_TOOLS,
    bulk_score_all_skus,
    calculate_reorder_point,
    check_expiry_risk,
    compute_eoq,
    get_days_of_supply,
    get_inventory_kpis,
    get_stock_level,
    list_critical_skus,
    record_stock_movement,
    score_inventory_health,
    update_stock_level,
)

# ── Procurement (Phase 3 + Phase 4) ───────────────────────────────────────
from tools.procurement_tools import (
    PROCUREMENT_TOOLS,
    calculate_split_order,
    cancel_po,
    compare_suppliers,
    find_best_supplier,
    generate_purchase_order,
    get_po_status,
    get_supplier_info,
    list_open_pos,
    submit_po,
)

# ── Fulfillment (Phase 5 NEW) ─────────────────────────────────────────────
from tools.fulfillment_tools import (
    FULFILLMENT_TOOLS,
    calculate_fill_rate,
    create_dispatch,
    get_open_orders,
    manage_backorder,
    route_to_best_dc,
    score_order_priority,
)

# ── Notification ───────────────────────────────────────────────────────────
from tools.notification_tools import (
    NOTIFICATION_TOOLS,
    build_run_summary,
    escalate_to_human,
    get_notification_log,
    get_unresolved_events,
    log_exception_event,
    send_alert,
    send_slack_notification,
)

# ── Master registry ────────────────────────────────────────────────────────
ALL_TOOLS = (
    FORECAST_TOOLS
    + INVENTORY_TOOLS
    + PROCUREMENT_TOOLS
    + FULFILLMENT_TOOLS
    + NOTIFICATION_TOOLS
)

__all__ = [
    "ALL_TOOLS", "FORECAST_TOOLS", "INVENTORY_TOOLS",
    "PROCUREMENT_TOOLS", "FULFILLMENT_TOOLS", "NOTIFICATION_TOOLS",
    # forecast
    "load_demand_history", "run_prophet_forecast", "compute_mape",
    "detect_demand_anomalies", "get_demand_trend",
    # inventory
    "get_stock_level", "compute_eoq", "calculate_reorder_point",
    "score_inventory_health", "list_critical_skus", "get_days_of_supply",
    "update_stock_level", "record_stock_movement", "bulk_score_all_skus",
    "get_inventory_kpis", "check_expiry_risk",
    # procurement
    "get_supplier_info", "find_best_supplier", "generate_purchase_order",
    "submit_po", "get_po_status", "list_open_pos", "cancel_po",
    "compare_suppliers", "calculate_split_order",
    # fulfillment
    "get_open_orders", "score_order_priority", "route_to_best_dc",
    "create_dispatch", "manage_backorder", "calculate_fill_rate",
    # notification
    "send_alert", "escalate_to_human", "log_exception_event",
    "build_run_summary", "send_slack_notification",
    "get_unresolved_events", "get_notification_log",
]
