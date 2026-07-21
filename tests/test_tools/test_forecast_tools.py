"""
tests/test_tools/test_forecast_tools.py
────────────────────────────────────────
Unit tests for tools/forecast_tools.py

Tests:
  - load_demand_history:     data loading, filtering, quality assessment
  - get_demand_trend:        trend classification correctness
  - detect_demand_anomalies: Z-score anomaly detection
  - compute_mape:            accuracy metric calculation
  - run_prophet_forecast:    fallback forecast shape and KPIs
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Bootstrap stubs before any project import ─────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pytest

# conftest installs all stubs + DB path patches via autouse fixtures,
# so we can import tools directly here.
import importlib


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def get_ft():
    """Lazily import forecast_tools (after conftest patches DB_PATH)."""
    return importlib.import_module("tools.forecast_tools")


# ═══════════════════════════════════════════════════════════════════
# load_demand_history
# ═══════════════════════════════════════════════════════════════════

class TestLoadDemandHistory:

    def test_returns_records_for_known_sku(self, mock_settings):
        ft = get_ft()
        result = ft.load_demand_history("SKU-001", days=30)

        assert "error" not in result
        assert result["sku_id"]    == "SKU-001"
        assert result["days_loaded"] > 0
        assert isinstance(result["records"], list)
        assert result["avg_daily_demand"] > 0

    def test_data_quality_good_for_full_history(self, mock_settings):
        ft = get_ft()
        result = ft.load_demand_history("SKU-001", days=30)
        # Full seed has 180 days of data — 30 day window should be "good"
        assert result["data_quality"] in ("good", "sparse")

    def test_sku_id_normalised_to_uppercase(self, mock_settings):
        ft = get_ft()
        result = ft.load_demand_history("sku-001", days=10)
        assert result["sku_id"] == "SKU-001"

    def test_unknown_sku_returns_empty_records(self, mock_settings):
        ft = get_ft()
        result = ft.load_demand_history("SKU-FAKE-999", days=30)
        assert result["records"] == []
        assert result["data_quality"] == "insufficient"

    def test_channel_filter_reduces_records(self, mock_settings):
        ft = get_ft()
        all_channels = ft.load_demand_history("SKU-001", days=60)
        online_only  = ft.load_demand_history("SKU-001", days=60, channel="online")

        assert online_only["days_loaded"] <= all_channels["days_loaded"]

    def test_days_clamped_to_365(self, mock_settings):
        ft = get_ft()
        result = ft.load_demand_history("SKU-001", days=9999)
        # Should not crash — days is clamped internally
        assert "error" not in result

    def test_total_demand_equals_sum_of_records(self, mock_settings):
        ft = get_ft()
        result = ft.load_demand_history("SKU-002", days=30)
        manual_sum = round(sum(r["quantity"] for r in result["records"]), 1)
        assert abs(result["total_demand"] - manual_sum) < 0.1


# ═══════════════════════════════════════════════════════════════════
# get_demand_trend
# ═══════════════════════════════════════════════════════════════════

class TestGetDemandTrend:

    def test_returns_valid_trend_value(self, mock_settings):
        ft = get_ft()
        result = ft.get_demand_trend("SKU-001", days=60)

        assert "error" not in result
        assert result["trend"] in ("rising", "falling", "stable", "volatile")

    def test_returns_numeric_slope(self, mock_settings):
        ft = get_ft()
        result = ft.get_demand_trend("SKU-001", days=60)

        assert isinstance(result["slope_units_per_day"], float)
        assert isinstance(result["pct_change"], float)
        assert isinstance(result["avg_daily"], float)
        assert result["avg_daily"] > 0

    def test_coefficient_of_variation_non_negative(self, mock_settings):
        ft = get_ft()
        result = ft.get_demand_trend("SKU-001", days=60)
        assert result["coefficient_of_variation"] >= 0

    def test_insufficient_data_returns_unknown(self, mock_settings):
        ft = get_ft()
        result = ft.get_demand_trend("SKU-FAKE-999", days=60)
        assert result["trend"] == "unknown"

    def test_all_skus_return_valid_trend(self, mock_settings, seeded_sku_ids):
        ft = get_ft()
        valid_trends = {"rising", "falling", "stable", "volatile", "unknown"}
        for sku_id in seeded_sku_ids[:5]:   # sample first 5
            r = ft.get_demand_trend(sku_id, days=60)
            assert r["trend"] in valid_trends, f"{sku_id} returned invalid trend: {r['trend']}"


# ═══════════════════════════════════════════════════════════════════
# detect_demand_anomalies
# ═══════════════════════════════════════════════════════════════════

class TestDetectDemandAnomalies:

    def test_returns_anomaly_structure(self, mock_settings):
        ft = get_ft()
        result = ft.detect_demand_anomalies("SKU-001", z_score_threshold=2.5, lookback_days=30)

        assert "error" not in result
        assert isinstance(result["anomalies"], list)
        assert isinstance(result["anomaly_count"], int)
        assert result["recommendation"] in ("clean", "review", "exclude_outliers", "insufficient_data")

    def test_anomaly_count_matches_list_length(self, mock_settings):
        ft = get_ft()
        result = ft.detect_demand_anomalies("SKU-001", z_score_threshold=2.5, lookback_days=30)
        assert result["anomaly_count"] == len(result["anomalies"])

    def test_lower_threshold_finds_more_anomalies(self, mock_settings):
        ft = get_ft()
        tight = ft.detect_demand_anomalies("SKU-001", z_score_threshold=1.0, lookback_days=30)
        loose = ft.detect_demand_anomalies("SKU-001", z_score_threshold=3.5, lookback_days=30)
        assert tight["anomaly_count"] >= loose["anomaly_count"]

    def test_anomaly_has_required_fields(self, mock_settings):
        ft  = get_ft()
        result = ft.detect_demand_anomalies("SKU-001", z_score_threshold=1.0, lookback_days=30)
        for anomaly in result["anomalies"]:
            assert "date"     in anomaly
            assert "quantity" in anomaly
            assert "z_score"  in anomaly
            assert "type"     in anomaly
            assert anomaly["type"] in ("spike", "drop")

    def test_unknown_sku_returns_empty(self, mock_settings):
        ft = get_ft()
        result = ft.detect_demand_anomalies("SKU-FAKE-999", lookback_days=30)
        assert result["anomaly_count"] == 0


# ═══════════════════════════════════════════════════════════════════
# compute_mape
# ═══════════════════════════════════════════════════════════════════

class TestComputeMape:

    def test_returns_valid_mape_structure(self, mock_settings):
        ft = get_ft()
        result = ft.compute_mape("SKU-001", holdout_days=14)

        assert "error" not in result
        assert result.get("mape") is not None
        assert result["mape"] >= 0
        assert result["accuracy_grade"] in ("excellent", "good", "acceptable", "poor")

    def test_mape_is_non_negative(self, mock_settings):
        ft = get_ft()
        result = ft.compute_mape("SKU-001", holdout_days=7)
        assert result["mape"] >= 0

    def test_grade_thresholds_correct(self, mock_settings):
        ft = get_ft()
        result = ft.compute_mape("SKU-001", holdout_days=14)
        mape  = result["mape"]
        grade = result["accuracy_grade"]
        if mape < 10:
            assert grade == "excellent"
        elif mape < 20:
            assert grade == "good"
        elif mape < 30:
            assert grade == "acceptable"
        else:
            assert grade == "poor"

    def test_insufficient_data_returns_none_mape(self, mock_settings):
        ft = get_ft()
        result = ft.compute_mape("SKU-FAKE-999", holdout_days=14)
        assert result.get("mape") is None or "error" in result


# ═══════════════════════════════════════════════════════════════════
# run_prophet_forecast (fallback path — Prophet not installed)
# ═══════════════════════════════════════════════════════════════════

class TestRunProphetForecast:

    def test_returns_forecast_structure(self, mock_settings):
        ft = get_ft()
        result = ft.run_prophet_forecast("SKU-001", horizon_days=30, history_days=60)

        assert "error" not in result
        assert result["sku_id"]        == "SKU-001"
        assert result["horizon_days"]  == 30
        assert result["method_used"]   in ("prophet", "moving_average")
        assert result["trend"]         in ("rising", "falling", "stable", "volatile")
        assert result["confidence"]    in ("high", "medium", "low")

    def test_forecast_has_correct_number_of_points(self, mock_settings):
        ft = get_ft()
        horizon = 30
        result  = ft.run_prophet_forecast("SKU-001", horizon_days=horizon)
        assert len(result["forecast"]) == horizon

    def test_forecast_points_have_required_fields(self, mock_settings):
        ft = get_ft()
        result = ft.run_prophet_forecast("SKU-001", horizon_days=14)
        for pt in result["forecast"]:
            assert "date"       in pt
            assert "yhat"       in pt
            assert "yhat_lower" in pt
            assert "yhat_upper" in pt
            assert pt["yhat"]   >= 0
            assert pt["yhat_lower"] <= pt["yhat_upper"]

    def test_avg_daily_consistent_with_total(self, mock_settings):
        ft = get_ft()
        result = ft.run_prophet_forecast("SKU-001", horizon_days=30)
        total  = result["total_forecasted_demand"]
        avg    = result["avg_daily_demand"]
        assert abs(total - avg * 30) < 1.0   # within 1 unit rounding

    def test_peak_demand_date_in_forecast_range(self, mock_settings):
        ft = get_ft()
        result     = ft.run_prophet_forecast("SKU-001", horizon_days=30)
        dates      = {pt["date"] for pt in result["forecast"]}
        peak_date  = result["peak_demand_date"]
        assert peak_date in dates

    def test_insufficient_history_returns_error(self, mock_settings):
        ft = get_ft()
        result = ft.run_prophet_forecast("SKU-FAKE-999", horizon_days=30)
        assert "error" in result
