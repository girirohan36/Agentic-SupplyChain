"""
tests/test_agents/test_workflow.py
───────────────────────────────────
End-to-end agent chain smoke tests.

These tests run every agent node in sequence — exactly as the
LangGraph workflow would — using the fallback (no-LLM) paths.

Test coverage:
  - demand_forecast_node   → produces ForecastResult in state
  - supply_planning_node   → produces SupplyPlanResult, sets replenishment_needed
  - inventory_node         → produces InventorySnapshot, raises exceptions
  - procurement_node       → generates POs, triggers HITL if high value
  - fulfillment_node       → routes orders against available stock
  - exception_handler_node → resolves all exceptions
  - Full chain integration → state flows correctly end-to-end
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import importlib
import pytest


def get_agent(name: str):
    return importlib.import_module(f"agents.{name}")


# ═══════════════════════════════════════════════════════════════════
# demand_forecast_node
# ═══════════════════════════════════════════════════════════════════

class TestDemandForecastNode:

    def test_returns_success_status(self, mock_settings, sample_state):
        df = get_agent("demand_forecast")
        result = df.demand_forecast_node(sample_state)
        assert result.get("demand_forecast_status") == "success"

    def test_produces_forecast_result(self, mock_settings, sample_state):
        df = get_agent("demand_forecast")
        result = df.demand_forecast_node(sample_state)
        fc = result.get("demand_forecast_result", {})

        assert fc.get("sku_id")          is not None
        assert fc.get("avg_daily_demand", 0) > 0
        assert fc.get("trend")           in ("rising", "falling", "stable", "volatile")
        assert fc.get("method_used")     in ("prophet", "moving_average")

    def test_supply_plan_input_forwarded(self, mock_settings, sample_state):
        df = get_agent("demand_forecast")
        result = df.demand_forecast_node(sample_state)

        spi = result.get("supply_plan_input", {})
        assert len(spi.get("forecasts", [])) > 0
        assert spi.get("avg_daily", 0) > 0

    def test_exception_raised_for_demand_spike(self, mock_settings, sample_state):
        """If peak > 2× avg, an exception_event should be appended."""
        df = get_agent("demand_forecast")

        # Patch fallback to produce artificial spike
        original_fallback = df._fallback_forecast
        def spiked_fallback(sku_ids, horizon):
            r = original_fallback(sku_ids, horizon)
            for fc in r["forecasts"]:
                fc["peak_demand_units"]   = fc.get("avg_daily_demand", 10) * 3.5
                fc["avg_daily_demand"]    = fc.get("avg_daily_demand", 10)
            return r
        df._fallback_forecast = spiked_fallback

        try:
            result = df.demand_forecast_node(sample_state)
            # Exception flag should be set if any spike was detected
            assert result.get("exception_detected", False) is True or \
                   len(result.get("exception_events", [])) > 0
        finally:
            df._fallback_forecast = original_fallback


# ═══════════════════════════════════════════════════════════════════
# supply_planning_node
# ═══════════════════════════════════════════════════════════════════

class TestSupplyPlanningNode:

    def test_returns_success_status(self, mock_settings, sample_state):
        sp = get_agent("supply_planning")
        result = sp.supply_planning_node(sample_state)
        assert result.get("supply_plan_status") == "success"

    def test_produces_sku_plans(self, mock_settings, sample_state):
        sp = get_agent("supply_planning")
        result = sp.supply_planning_node(sample_state)
        plans  = result.get("supply_plan_result", {}).get("sku_plans", [])

        assert len(plans) == len(sample_state["sku_ids"])
        for plan in plans:
            assert "sku_id"   in plan
            assert "urgency"  in plan
            assert "eoq"      in plan
            assert plan["eoq"] > 0

    def test_sets_replenishment_needed_flag(self, mock_settings, sample_state):
        sp = get_agent("supply_planning")
        result = sp.supply_planning_node(sample_state)
        # One or both of our test SKUs should need replenishment
        assert "replenishment_needed" in result

    def test_procurement_input_populated(self, mock_settings, sample_state):
        sp = get_agent("supply_planning")
        result = sp.supply_planning_node(sample_state)
        pi = result.get("procurement_input", {})

        assert "reorder_triggers" in pi
        assert "sku_plans"        in pi

    def test_urgency_values_valid(self, mock_settings, sample_state):
        sp = get_agent("supply_planning")
        result = sp.supply_planning_node(sample_state)
        plans  = result.get("supply_plan_result", {}).get("sku_plans", [])
        valid  = {"critical", "high", "medium", "low"}
        for plan in plans:
            assert plan["urgency"] in valid, \
                f"Invalid urgency: {plan['urgency']} for {plan['sku_id']}"


# ═══════════════════════════════════════════════════════════════════
# inventory_node
# ═══════════════════════════════════════════════════════════════════

class TestInventoryNode:

    def test_returns_success_status(self, mock_settings, sample_state):
        inv = get_agent("inventory")
        result = inv.inventory_node(sample_state)
        assert result.get("inventory_status") == "success"

    def test_produces_records_for_all_skus(self, mock_settings, sample_state):
        inv = get_agent("inventory")
        result = inv.inventory_node(sample_state)
        records = result.get("inventory_result", {}).get("records", [])

        sku_ids_in_records = {r["sku_id"] for r in records}
        for sku in sample_state["sku_ids"]:
            assert sku in sku_ids_in_records

    def test_health_scores_in_range(self, mock_settings, sample_state):
        inv = get_agent("inventory")
        result = inv.inventory_node(sample_state)
        for rec in result.get("inventory_result", {}).get("records", []):
            assert 0.0 <= rec["health_score"] <= 1.0

    def test_health_status_valid(self, mock_settings, sample_state):
        inv = get_agent("inventory")
        result  = inv.inventory_node(sample_state)
        valid   = {"healthy", "at_risk", "critical", "stock_out"}
        for rec in result.get("inventory_result", {}).get("records", []):
            assert rec["health_status"] in valid

    def test_aggregate_counts_consistent(self, mock_settings, sample_state):
        inv = get_agent("inventory")
        result = inv.inventory_node(sample_state)
        inv_res = result["inventory_result"]
        records = inv_res["records"]

        assert inv_res["total_skus"]    == len(records)
        assert (inv_res["critical_count"]
                + inv_res["at_risk_count"]
                + inv_res["healthy_count"]) == len(records)

    def test_fulfillment_input_populated(self, mock_settings, sample_state):
        inv = get_agent("inventory")
        result = inv.inventory_node(sample_state)
        fi = result.get("fulfillment_input", {})
        assert "open_orders"    in fi
        assert "health_records" in fi


# ═══════════════════════════════════════════════════════════════════
# procurement_node
# ═══════════════════════════════════════════════════════════════════

class TestProcurementNode:

    def test_returns_success_status(self, mock_settings, sample_state):
        proc = get_agent("procurement")
        result = proc.procurement_node(sample_state)
        assert result.get("procurement_status") == "success"

    def test_generates_pos_for_replenishment_skus(self, mock_settings, sample_state):
        proc   = get_agent("procurement")
        result = proc.procurement_node(sample_state)
        pr     = result.get("procurement_result", {})

        assert "issued_pos" in pr
        assert "total_value" in pr
        assert pr["total_value"] >= 0

    def test_hitl_triggered_for_high_value_pos(self, mock_settings, sample_state):
        """Both test SKUs have high on-hand=50 vs reorder=200 → POs generated."""
        proc   = get_agent("procurement")
        result = proc.procurement_node(sample_state)
        pr     = result.get("procurement_result", {})

        # If POs were generated, HITL should trigger when value > $5K
        if pr.get("total_value", 0) >= 5000:
            assert result.get("hitl_required") is True

    def test_no_replenishment_returns_empty_pos(self, mock_settings, sample_state):
        """If no SKU needs replenishment, PO list should be empty."""
        proc  = get_agent("procurement")
        # Override procurement_input to have no SKUs needing replenishment
        state = {
            **sample_state,
            "procurement_input": {
                "reorder_triggers": [],
                "recommended_pos":  [],
                "sku_plans": [
                    {**p, "needs_replenishment": False}
                    for p in sample_state["procurement_input"]["sku_plans"]
                ],
            },
        }
        result = proc.procurement_node(state)
        pr     = result.get("procurement_result", {})
        assert pr.get("total_pos", 0) == 0


# ═══════════════════════════════════════════════════════════════════
# fulfillment_node
# ═══════════════════════════════════════════════════════════════════

class TestFulfillmentNode:

    def _build_fulfillment_state(self, sample_state: dict) -> dict:
        """Build state with fulfillment_input pre-populated."""
        inv = get_agent("inventory")
        inv_result = inv.inventory_node(sample_state)
        state = {**sample_state, **inv_result}
        return state

    def test_returns_success_status(self, mock_settings, sample_state):
        ff    = get_agent("fulfillment")
        state = self._build_fulfillment_state(sample_state)
        result = ff.fulfillment_node(state)
        assert result.get("fulfillment_status") in ("success", "skipped")

    def test_produces_routed_and_held_orders(self, mock_settings, sample_state):
        ff    = get_agent("fulfillment")
        state = self._build_fulfillment_state(sample_state)
        result = ff.fulfillment_node(state)
        fr    = result.get("fulfillment_result", {})

        assert "routed_orders" in fr
        assert "held_orders"   in fr
        assert "dispatch_list" in fr

    def test_fulfilled_order_qty_leq_needed(self, mock_settings, sample_state):
        ff    = get_agent("fulfillment")
        state = self._build_fulfillment_state(sample_state)
        result = ff.fulfillment_node(state)
        for order in result.get("fulfillment_result", {}).get("routed_orders", []):
            assert order["qty_fulfilled"] <= order["qty_needed"]

    def test_skipped_when_hitl_rejected(self, mock_settings, sample_state):
        ff = get_agent("fulfillment")
        state = {
            **sample_state,
            "hitl_required": True,
            "hitl_approved": False,
        }
        result = ff.fulfillment_node(state)
        assert result.get("fulfillment_status") == "skipped"


# ═══════════════════════════════════════════════════════════════════
# exception_handler_node
# ═══════════════════════════════════════════════════════════════════

class TestExceptionHandlerNode:

    def test_resolves_all_exceptions(self, mock_settings, sample_state):
        exc = get_agent("exception_handler")
        state = {
            **sample_state,
            "exception_events": [
                {
                    "run_id":         "RUN-TEST-001",
                    "raised_by":      "inventory",
                    "exception_type": "stockout_imminent",
                    "severity":       "critical",
                    "sku_id":         "SKU-011",
                    "supplier_id":    None,
                    "description":    "SKU-011 is at zero stock.",
                    "context":        {},
                    "resolved":       False,
                },
                {
                    "run_id":         "RUN-TEST-001",
                    "raised_by":      "demand_forecast",
                    "exception_type": "demand_spike",
                    "severity":       "warning",
                    "sku_id":         "SKU-001",
                    "supplier_id":    None,
                    "description":    "Demand spike detected.",
                    "context":        {},
                    "resolved":       False,
                },
            ],
            "exception_detected": True,
        }

        result = exc.exception_handler_node(state)
        er     = result.get("exception_result", {})

        assert result.get("exception_status") == "success"
        assert er.get("total_events")   == 2
        assert er.get("resolved_count", 0) + er.get("escalated_count", 0) == 2

    def test_critical_events_escalated(self, mock_settings, sample_state):
        exc = get_agent("exception_handler")
        state = {
            **sample_state,
            "exception_events": [{
                "run_id": "RUN-TEST-001", "raised_by": "inventory",
                "exception_type": "stockout_imminent", "severity": "critical",
                "sku_id": "SKU-011", "supplier_id": None,
                "description": "Critical stock.", "context": {}, "resolved": False,
            }],
        }
        result = exc.exception_handler_node(state)
        er     = result.get("exception_result", {})
        assert er.get("escalated_count", 0) >= 1

    def test_notifications_generated(self, mock_settings, sample_state):
        exc = get_agent("exception_handler")
        state = {
            **sample_state,
            "exception_events": [{
                "run_id": "RUN-TEST-001", "raised_by": "inventory",
                "exception_type": "stockout_imminent", "severity": "high",
                "sku_id": "SKU-011", "supplier_id": None,
                "description": "High priority.", "context": {}, "resolved": False,
            }],
        }
        result = exc.exception_handler_node(state)
        notifs = result.get("exception_result", {}).get("notifications", [])
        assert len(notifs) > 0

    def test_exception_detected_reset_to_false(self, mock_settings, sample_state):
        exc   = get_agent("exception_handler")
        state = {**sample_state, "exception_events": [], "exception_detected": True}
        result = exc.exception_handler_node(state)
        assert result.get("exception_detected") is False


# ═══════════════════════════════════════════════════════════════════
# Full chain integration
# ═══════════════════════════════════════════════════════════════════

class TestFullChainIntegration:

    def test_full_agent_chain_runs_without_error(self, mock_settings, sample_state):
        """
        Run every agent in the correct order.
        State is accumulated between calls, exactly as LangGraph would do.
        Verifies the full data flow from forecast → procurement → exception.
        """
        df   = get_agent("demand_forecast")
        sp   = get_agent("supply_planning")
        inv  = get_agent("inventory")
        proc = get_agent("procurement")
        ff   = get_agent("fulfillment")
        exc  = get_agent("exception_handler")

        state = dict(sample_state)

        # 1. Demand Forecast
        state.update(df.demand_forecast_node(state))
        assert state.get("demand_forecast_status") == "success"

        # 2. Supply Planning
        state.update(sp.supply_planning_node(state))
        assert state.get("supply_plan_status") == "success"

        # 3. Inventory
        state.update(inv.inventory_node(state))
        assert state.get("inventory_status") == "success"

        # 4. Procurement (conditional on replenishment_needed)
        if state.get("replenishment_needed"):
            state.update(proc.procurement_node(state))
            assert state.get("procurement_status") == "success"

        # 5. Fulfillment
        state.update(ff.fulfillment_node(state))
        assert state.get("fulfillment_status") in ("success", "skipped")

        # 6. Exception Handler (conditional)
        if state.get("exception_detected"):
            state.update(exc.exception_handler_node(state))
            assert state.get("exception_status") == "success"
            assert state.get("exception_detected") is False

        # Final state assertions
        assert state.get("inventory_result") is not None
        assert state.get("demand_forecast_result") is not None

    def test_state_keys_populated_correctly(self, mock_settings, sample_state):
        """Verify specific state keys are correctly populated by each agent."""
        df  = get_agent("demand_forecast")
        sp  = get_agent("supply_planning")
        inv = get_agent("inventory")

        state = dict(sample_state)
        state.update(df.demand_forecast_node(state))
        state.update(sp.supply_planning_node(state))
        state.update(inv.inventory_node(state))

        # Demand forecast result has correct shape
        fc = state["demand_forecast_result"]
        assert "sku_id"          in fc
        assert "avg_daily_demand" in fc

        # Supply plan result has sku_plans
        sp_res = state["supply_plan_result"]
        assert len(sp_res["sku_plans"]) == len(sample_state["sku_ids"])

        # Inventory result has all SKUs
        inv_res = state["inventory_result"]
        assert inv_res["total_skus"] == len(sample_state["sku_ids"])

    def test_exception_events_accumulate_across_agents(self, mock_settings, sample_state):
        """Exceptions raised by inventory + fulfillment should both end up in state."""
        inv = get_agent("inventory")
        ff  = get_agent("fulfillment")

        state = dict(sample_state)
        state.update(inv.inventory_node(state))
        pre_count = len(state.get("exception_events", []))

        state.update(ff.fulfillment_node(state))
        post_count = len(state.get("exception_events", []))

        # Post count >= pre count (fulfillment may add more)
        assert post_count >= pre_count
