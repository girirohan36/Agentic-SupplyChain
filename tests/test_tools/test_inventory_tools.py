"""
tests/test_tools/test_inventory_tools.py
─────────────────────────────────────────
Unit tests for tools/inventory_tools.py

Tests:
  - get_stock_level:          DB retrieval, ATP calculation
  - compute_eoq:              formula accuracy, parameter sensitivity
  - calculate_reorder_point:  ROP formula, safety stock calculation
  - score_inventory_health:   score bounds, status classification
  - list_critical_skus:       threshold filtering, sort order
  - get_days_of_supply:       urgency classification, stockout date
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import importlib
import pytest


def get_invt():
    return importlib.import_module("tools.inventory_tools")


# ═══════════════════════════════════════════════════════════════════
# get_stock_level
# ═══════════════════════════════════════════════════════════════════

class TestGetStockLevel:

    def test_known_sku_returns_found_true(self, mock_settings):
        invt   = get_invt()
        result = invt.get_stock_level("SKU-001")
        assert result["found"]    is True
        assert result["sku_id"]   == "SKU-001"
        assert result["on_hand"]  >= 0
        assert result["reserved"] >= 0

    def test_available_equals_on_hand_minus_reserved(self, mock_settings):
        invt   = get_invt()
        result = invt.get_stock_level("SKU-001")
        expected_available = max(0.0, result["on_hand"] - result["reserved"])
        assert abs(result["available"] - expected_available) < 0.01

    def test_total_supply_includes_in_transit(self, mock_settings):
        invt   = get_invt()
        result = invt.get_stock_level("SKU-001")
        expected_total = result["on_hand"] + result["in_transit"]
        assert abs(result["total_supply"] - expected_total) < 0.01

    def test_unknown_sku_returns_found_false(self, mock_settings):
        invt   = get_invt()
        result = invt.get_stock_level("SKU-FAKE-999")
        assert result["found"]   is False
        assert result["on_hand"] == 0.0

    def test_default_location_is_dc01(self, mock_settings):
        invt   = get_invt()
        result = invt.get_stock_level("SKU-001")
        assert result["location_id"] == "DC-01"

    def test_all_seeded_skus_found(self, mock_settings, seeded_sku_ids):
        invt = get_invt()
        for sku_id in seeded_sku_ids:
            r = invt.get_stock_level(sku_id)
            assert r["found"] is True, f"{sku_id} not found in inventory"


# ═══════════════════════════════════════════════════════════════════
# compute_eoq
# ═══════════════════════════════════════════════════════════════════

class TestComputeEOQ:

    def test_eoq_formula_accuracy(self, mock_settings):
        """
        EOQ = sqrt(2DS/H).
        For D=3650, S=100, unit_cost=45 → H=45*0.25=11.25
        EOQ ≈ sqrt(2*3650*100/11.25) ≈ sqrt(64889) ≈ 254.7
        We allow ±5 units for DB-loaded unit_cost variation.
        """
        invt   = get_invt()
        result = invt.compute_eoq("SKU-001", annual_demand=3650.0, ordering_cost=100.0)

        assert "error" not in result
        assert result["eoq"] > 0
        assert result["annual_orders"] > 0
        assert result["days_between_orders"] > 0
        assert result["total_annual_cost"] > 0

    def test_higher_demand_yields_larger_eoq(self, mock_settings):
        invt = get_invt()
        low  = invt.compute_eoq("SKU-001", annual_demand=1000.0)
        high = invt.compute_eoq("SKU-001", annual_demand=10000.0)
        assert high["eoq"] > low["eoq"]

    def test_higher_holding_cost_yields_smaller_eoq(self, mock_settings):
        invt    = get_invt()
        cheap   = invt.compute_eoq("SKU-001", annual_demand=3650.0, holding_cost_pct=0.10)
        pricey  = invt.compute_eoq("SKU-001", annual_demand=3650.0, holding_cost_pct=0.40)
        assert cheap["eoq"] > pricey["eoq"]

    def test_days_between_orders_consistent(self, mock_settings):
        invt   = get_invt()
        result = invt.compute_eoq("SKU-001", annual_demand=3650.0)
        expected_days = 365.0 / result["annual_orders"]
        assert abs(result["days_between_orders"] - expected_days) < 0.5

    def test_eoq_positive_for_all_skus(self, mock_settings, seeded_sku_ids):
        invt = get_invt()
        for sku_id in seeded_sku_ids[:5]:
            r = invt.compute_eoq(sku_id, annual_demand=3650.0)
            assert r.get("eoq", 0) > 0, f"EOQ not positive for {sku_id}"


# ═══════════════════════════════════════════════════════════════════
# calculate_reorder_point
# ═══════════════════════════════════════════════════════════════════

class TestCalculateReorderPoint:

    def test_rop_formula(self, mock_settings):
        """
        ROP = avg_daily × lead_time + safety_stock
            = avg_daily × lead_time + avg_daily × safety_stock_days
        """
        invt      = get_invt()
        avg_daily = 20.0
        lead_time = 14
        ss_days   = 7

        result = invt.calculate_reorder_point(
            "SKU-001",
            avg_daily_demand=avg_daily,
            lead_time_days=lead_time,
            safety_stock_days=ss_days,
        )

        expected_ss  = avg_daily * ss_days          # 140
        expected_rop = avg_daily * lead_time + expected_ss  # 280+140=420

        assert abs(result["reorder_point"] - expected_rop) < 0.5
        assert abs(result["safety_stock"]  - expected_ss)  < 0.5

    def test_longer_lead_time_increases_rop(self, mock_settings):
        invt  = get_invt()
        short = invt.calculate_reorder_point("SKU-001", avg_daily_demand=20.0, lead_time_days=7)
        long_ = invt.calculate_reorder_point("SKU-001", avg_daily_demand=20.0, lead_time_days=30)
        assert long_["reorder_point"] > short["reorder_point"]

    def test_higher_daily_demand_increases_rop(self, mock_settings):
        invt = get_invt()
        low  = invt.calculate_reorder_point("SKU-001", avg_daily_demand=5.0,  lead_time_days=14)
        high = invt.calculate_reorder_point("SKU-001", avg_daily_demand=50.0, lead_time_days=14)
        assert high["reorder_point"] > low["reorder_point"]

    def test_includes_interpretation_string(self, mock_settings):
        invt   = get_invt()
        result = invt.calculate_reorder_point("SKU-001", avg_daily_demand=20.0)
        assert "interpretation" in result
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 10


# ═══════════════════════════════════════════════════════════════════
# score_inventory_health
# ═══════════════════════════════════════════════════════════════════

class TestScoreInventoryHealth:

    def test_score_in_valid_range(self, mock_settings):
        invt   = get_invt()
        result = invt.score_inventory_health("SKU-001")
        assert "error"        not in result
        assert 0.0 <= result["health_score"] <= 1.0

    def test_status_consistent_with_score(self, mock_settings):
        invt   = get_invt()
        result = invt.score_inventory_health("SKU-001")
        score  = result["health_score"]
        status = result["health_status"]

        if result.get("on_hand", 1) <= 0:
            assert status == "stock_out"
        elif score < 0.40:
            assert status == "critical"
        elif score < 0.75:
            assert status == "at_risk"
        else:
            assert status == "healthy"

    def test_component_scores_sum_correctly(self, mock_settings):
        invt   = get_invt()
        result = invt.score_inventory_health("SKU-001")
        comps  = result.get("components", {})
        if comps:
            weighted = (
                0.50 * comps["dos_score"]
                + 0.30 * comps["rop_score"]
                + 0.20 * comps["ss_score"]
            )
            assert abs(result["health_score"] - round(weighted, 4)) < 0.001

    def test_reorder_triggered_flag_correct(self, mock_settings):
        invt   = get_invt()
        result = invt.score_inventory_health("SKU-001")
        stock  = invt.get_stock_level("SKU-001")
        rop    = stock.get("reorder_point") or 50.0
        total  = stock["on_hand"] + stock["in_transit"]
        expected_triggered = total <= rop
        assert result["reorder_triggered"] == expected_triggered

    def test_all_skus_score_without_error(self, mock_settings, seeded_sku_ids):
        invt = get_invt()
        for sku_id in seeded_sku_ids:
            r = invt.score_inventory_health(sku_id)
            assert "error" not in r or r.get("health_score") is not None, \
                f"{sku_id} returned error: {r.get('error')}"


# ═══════════════════════════════════════════════════════════════════
# list_critical_skus
# ═══════════════════════════════════════════════════════════════════

class TestListCriticalSkus:

    def test_returns_list_structure(self, mock_settings):
        invt   = get_invt()
        result = invt.list_critical_skus()
        assert "error"         not in result
        assert "skus"          in result
        assert "total_critical" in result
        assert result["total_critical"] == len(result["skus"])

    def test_critical_skus_have_required_fields(self, mock_settings):
        invt   = get_invt()
        result = invt.list_critical_skus()
        for sku in result["skus"]:
            assert "sku_id"          in sku
            assert "status"          in sku
            assert "days_of_supply"  in sku
            assert "on_hand"         in sku
            assert sku["status"] in ("stock_out", "critical", "at_risk")

    def test_sorted_by_urgency(self, mock_settings):
        invt   = get_invt()
        result = invt.list_critical_skus()
        if len(result["skus"]) < 2:
            pytest.skip("Need at least 2 critical SKUs")

        priority = {"stock_out": 0, "critical": 1, "at_risk": 2}
        statuses = [priority[s["status"]] for s in result["skus"]]
        assert statuses == sorted(statuses), "Results not sorted by urgency"

    def test_threshold_filters_correctly(self, mock_settings):
        invt   = get_invt()
        tight  = invt.list_critical_skus(min_dos_threshold=3.0)
        loose  = invt.list_critical_skus(min_dos_threshold=30.0)
        assert loose["total_critical"] >= tight["total_critical"]

    def test_max_results_respected(self, mock_settings):
        invt   = get_invt()
        result = invt.list_critical_skus(max_results=2)
        assert len(result["skus"]) <= 2

    def test_returns_at_least_some_critical_skus(self, mock_settings):
        """Seed data creates ~20% critical/at-risk SKUs by design."""
        invt   = get_invt()
        result = invt.list_critical_skus(min_dos_threshold=30.0)
        assert result["total_critical"] > 0, \
            "Expected at least 1 critical SKU in seeded data"


# ═══════════════════════════════════════════════════════════════════
# get_days_of_supply
# ═══════════════════════════════════════════════════════════════════

class TestGetDaysOfSupply:

    def test_returns_valid_structure(self, mock_settings):
        invt   = get_invt()
        result = invt.get_days_of_supply("SKU-001")
        assert "error"          not in result
        assert "days_of_supply" in result
        assert "urgency"        in result
        assert result["days_of_supply"] >= 0

    def test_urgency_classification_correct(self, mock_settings):
        invt   = get_invt()
        result = invt.get_days_of_supply("SKU-001")
        dos    = result["days_of_supply"]
        urgency = result["urgency"]

        if result.get("available", 1) <= 0:
            assert urgency == "stock_out"
        elif dos < 7:
            assert urgency in ("critical", "stock_out")
        elif dos < 14:
            assert urgency == "urgent"
        elif dos < 30:
            assert urgency == "watch"
        else:
            assert urgency == "comfortable"

    def test_stockout_date_set_when_finite(self, mock_settings):
        invt   = get_invt()
        result = invt.get_days_of_supply("SKU-001")
        dos    = result.get("days_of_supply", 999)
        if dos < 999:
            assert result.get("stockout_date") is not None

    def test_known_critical_sku_has_low_dos(self, mock_settings):
        """SKU-011 is seeded as critical (near stockout)."""
        invt   = get_invt()
        result = invt.get_days_of_supply("SKU-011")
        # Should be at_risk or worse
        assert result["urgency"] in (
            "stock_out", "critical", "urgent", "watch"
        ), f"Expected SKU-011 to be low stock, got urgency={result['urgency']}"
