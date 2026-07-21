"""
tools/forecast_tools.py
────────────────────────
LangChain @tool definitions for the Demand Forecast Agent.

Every function here is:
  1. Decorated with @tool so the LLM can call it via bind_tools()
  2. Typed with a Pydantic args_schema for structured input validation
  3. Documented so the LLM knows when and how to use it
  4. Self-contained — can be called directly in tests

Tools:
  - load_demand_history      → pull raw demand rows from SQLite
  - run_prophet_forecast     → run Prophet and return a serialised ForecastResult
  - compute_mape             → measure forecast accuracy on a holdout set
  - detect_demand_anomalies  → flag unusual demand spikes / drops
  - get_demand_trend         → classify the overall demand direction
"""

from __future__ import annotations

import math
import sqlite3
import statistics
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config.settings import get_settings

settings = get_settings()
DB_PATH  = Path("data/supply_demand.db")


# ═══════════════════════════════════════════════════════════════════
# Arg schemas (Pydantic v2) — give the LLM structured input contracts
# ═══════════════════════════════════════════════════════════════════

class LoadDemandHistoryArgs(BaseModel):
    sku_id: str  = Field(..., description="The SKU identifier to load demand for (e.g. 'SKU-001')")
    days:   int  = Field(90,  description="Number of historical days to load (default 90, max 365)")
    channel: Optional[str] = Field(None, description="Filter by sales channel: 'online', 'retail', 'wholesale', or None for all")


class RunProphetForecastArgs(BaseModel):
    sku_id:      str = Field(..., description="SKU to forecast")
    horizon_days: int = Field(90,  description="Number of days ahead to forecast (1–365)")
    history_days: int = Field(180, description="Days of history to train on (min 30)")


class ComputeMapeArgs(BaseModel):
    sku_id:       str = Field(..., description="SKU to evaluate")
    holdout_days: int = Field(14,  description="Number of recent days to hold out for accuracy evaluation")


class DetectAnomaliesArgs(BaseModel):
    sku_id:           str   = Field(..., description="SKU to check for demand anomalies")
    z_score_threshold: float = Field(2.5,  description="Z-score threshold above which a data point is flagged as an anomaly")
    lookback_days:    int   = Field(30,   description="Days of recent history to scan")


class GetDemandTrendArgs(BaseModel):
    sku_id: str = Field(..., description="SKU to analyse for trend direction")
    days:   int = Field(60,  description="Rolling window in days for trend calculation")


# ═══════════════════════════════════════════════════════════════════
# Tool 1 — load_demand_history
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=LoadDemandHistoryArgs)
def load_demand_history(
    sku_id:  str,
    days:    int = 90,
    channel: Optional[str] = None,
) -> dict:
    """
    Load historical daily demand records for a specific SKU from the database.

    Use this tool when you need raw demand data to:
    - Pass to run_prophet_forecast as training data
    - Analyse consumption patterns
    - Spot unusual demand behaviour before forecasting

    Returns a dict with keys:
      - sku_id: str
      - days_loaded: int
      - records: list of {date, quantity, channel, region}
      - avg_daily_demand: float
      - total_demand: float
      - data_quality: 'good' | 'sparse' | 'insufficient'
    """
    days    = min(max(days, 1), 365)
    cutoff  = str(date.today() - timedelta(days=days))

    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        query  = "SELECT order_date, sku_id, quantity, channel, region FROM orders WHERE sku_id = ? AND order_date >= ?"
        params: list = [sku_id.upper().strip(), cutoff]

        if channel:
            query  += " AND channel = ?"
            params.append(channel)

        query += " ORDER BY order_date"
        rows   = conn.execute(query, params).fetchall()
        conn.close()

        records = [dict(r) for r in rows]
        quantities = [r["quantity"] for r in records]

        total   = round(sum(quantities), 1)
        avg     = round(total / len(quantities), 2) if quantities else 0.0

        # Data quality assessment
        if len(records) >= days * 0.8:
            quality = "good"
        elif len(records) >= 7:
            quality = "sparse"
        else:
            quality = "insufficient"

        return {
            "sku_id":           sku_id.upper().strip(),
            "days_requested":   days,
            "days_loaded":      len(records),
            "records":          records,
            "avg_daily_demand": avg,
            "total_demand":     total,
            "data_quality":     quality,
        }

    except Exception as exc:
        return {"error": str(exc), "sku_id": sku_id, "records": []}


# ═══════════════════════════════════════════════════════════════════
# Tool 2 — run_prophet_forecast
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=RunProphetForecastArgs)
def run_prophet_forecast(
    sku_id:       str,
    horizon_days: int = 90,
    history_days: int = 180,
) -> dict:
    """
    Run a Facebook Prophet time-series forecast for a SKU.

    Automatically loads demand history, trains the model,
    and returns a full forecast with confidence intervals.

    Use this tool when you need future demand estimates to:
    - Calculate how much stock to order
    - Set reorder points and safety stock levels
    - Identify upcoming demand peaks

    Returns a ForecastResult dict with:
      - forecast: list of {date, yhat, yhat_lower, yhat_upper}
      - avg_daily_demand, total_forecasted_demand
      - peak_demand_date, peak_demand_units
      - trend: 'rising' | 'falling' | 'stable' | 'volatile'
      - confidence: 'high' | 'medium' | 'low'
      - method_used: 'prophet' | 'moving_average' (fallback)
    """
    # Load history first
    history_data = load_demand_history(sku_id=sku_id, days=history_days)

    records = history_data.get("records", [])

    if len(records) < 7:
        return {
            "error": f"Insufficient history for {sku_id}: {len(records)} records (need ≥7)",
            "sku_id": sku_id,
        }

    # ── Attempt Prophet ────────────────────────────────────────────
    try:
        import pandas as pd
        from prophet import Prophet

        df          = pd.DataFrame(records)[["order_date", "quantity"]].copy()
        df.columns  = ["ds", "y"]
        df["ds"]    = pd.to_datetime(df["ds"])
        df          = df.groupby("ds", as_index=False)["y"].sum()  # aggregate multi-channel

        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=len(records) >= 365,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.80,
        )
        model.fit(df)

        future   = model.make_future_dataframe(periods=horizon_days)
        forecast = model.predict(future)
        fcast    = forecast.tail(horizon_days)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

        points = [
            {
                "date":       str(row.ds.date()),
                "yhat":       round(max(0.0, row.yhat), 1),
                "yhat_lower": round(max(0.0, row.yhat_lower), 1),
                "yhat_upper": round(max(0.0, row.yhat_upper), 1),
                "is_anomaly": False,
            }
            for row in fcast.itertuples()
        ]
        method = "prophet"

    except ImportError:
        # ── Moving-average fallback ───────────────────────────────
        recent = [r["quantity"] for r in records[-30:]]
        avg    = statistics.mean(recent) if recent else 10.0
        std    = statistics.stdev(recent) if len(recent) > 1 else avg * 0.15

        points = []
        for i in range(1, horizon_days + 1):
            d    = date.today() + timedelta(days=i)
            yhat = round(max(0.0, avg + (avg * 0.02 * math.sin(2 * math.pi * i / 7))), 1)
            points.append({
                "date":       str(d),
                "yhat":       yhat,
                "yhat_lower": round(max(0.0, yhat - std), 1),
                "yhat_upper": round(yhat + std, 1),
                "is_anomaly": False,
            })
        method = "moving_average"

    # ── Compute KPIs ──────────────────────────────────────────────
    yhats   = [p["yhat"] for p in points]
    total   = round(sum(yhats), 1)
    avg_d   = round(total / len(yhats), 2) if yhats else 0.0
    peak_i  = yhats.index(max(yhats))

    # Trend: compare first vs second half averages
    half    = len(yhats) // 2
    a1, a2  = (sum(yhats[:half]) / half), (sum(yhats[half:]) / half)
    ratio   = a2 / a1 if a1 > 0 else 1.0
    trend   = "rising" if ratio > 1.10 else "falling" if ratio < 0.90 else "stable"

    # Confidence: based on data quality and method
    quality = history_data.get("data_quality", "sparse")
    confidence = "high" if (method == "prophet" and quality == "good") \
                 else "medium" if quality == "sparse" \
                 else "low"

    return {
        "sku_id":                  sku_id.upper().strip(),
        "generated_at":            datetime.utcnow().isoformat(),
        "method_used":             method,
        "horizon_days":            horizon_days,
        "forecast":                points,
        "total_forecasted_demand": total,
        "avg_daily_demand":        avg_d,
        "peak_demand_date":        points[peak_i]["date"],
        "peak_demand_units":       points[peak_i]["yhat"],
        "trend":                   trend,
        "confidence":              confidence,
        "mape":                    None,
        "llm_commentary":          None,
        "warnings":                [] if quality == "good" else [f"Data quality: {quality}"],
    }


# ═══════════════════════════════════════════════════════════════════
# Tool 3 — compute_mape
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=ComputeMapeArgs)
def compute_mape(sku_id: str, holdout_days: int = 14) -> dict:
    """
    Compute Mean Absolute Percentage Error (MAPE) for a SKU's forecast.

    Holds out the last `holdout_days` of actual demand, trains a forecast
    on the prior history, and compares predictions to actuals.

    Use this tool to:
    - Validate forecast quality before acting on it
    - Flag low-confidence SKUs for manual review

    Returns:
      - mape: float (lower is better; <10% = excellent, <20% = good, >30% = poor)
      - mae: float (mean absolute error in units)
      - accuracy_grade: 'excellent' | 'good' | 'acceptable' | 'poor'
    """
    try:
        # Load all history
        all_data = load_demand_history(sku_id=sku_id, days=365)
        records  = all_data.get("records", [])

        if len(records) < holdout_days + 7:
            return {
                "sku_id": sku_id,
                "error": f"Insufficient data: {len(records)} rows",
                "mape": None,
            }

        train   = records[:-holdout_days]
        actuals = records[-holdout_days:]

        if not train:
            return {"sku_id": sku_id, "error": "No training data after holdout split", "mape": None}

        train_qty = [r["quantity"] for r in train]
        avg_train = statistics.mean(train_qty)

        # Simple moving-average forecast for holdout
        pct_errors, abs_errors = [], []
        for actual_rec in actuals:
            actual = actual_rec["quantity"]
            pred   = avg_train
            if actual > 0:
                pct_errors.append(abs(actual - pred) / actual * 100)
            abs_errors.append(abs(actual - pred))

        mape = round(statistics.mean(pct_errors), 2) if pct_errors else None
        mae  = round(statistics.mean(abs_errors), 2) if abs_errors else None

        grade = (
            "excellent"  if mape is not None and mape < 10  else
            "good"       if mape is not None and mape < 20  else
            "acceptable" if mape is not None and mape < 30  else
            "poor"
        )

        return {
            "sku_id":         sku_id,
            "holdout_days":   holdout_days,
            "mape":           mape,
            "mae":            mae,
            "accuracy_grade": grade,
        }

    except Exception as exc:
        return {"sku_id": sku_id, "error": str(exc), "mape": None}


# ═══════════════════════════════════════════════════════════════════
# Tool 4 — detect_demand_anomalies
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=DetectAnomaliesArgs)
def detect_demand_anomalies(
    sku_id:            str,
    z_score_threshold: float = 2.5,
    lookback_days:     int   = 30,
) -> dict:
    """
    Detect unusual demand spikes or drops for a SKU using Z-score analysis.

    A data point is flagged as an anomaly when its Z-score exceeds
    the threshold (default 2.5 standard deviations from the mean).

    Use this tool when:
    - You suspect a demand signal is corrupted or represents a one-off event
    - You want to avoid letting a spike distort the forecast
    - An ExceptionEvent of type demand_spike has been raised

    Returns:
      - anomalies: list of {date, quantity, z_score, type}
      - anomaly_count: int
      - recommendation: 'clean' | 'review' | 'exclude_outliers'
    """
    try:
        data    = load_demand_history(sku_id=sku_id, days=lookback_days + 30)
        records = data.get("records", [])[-lookback_days:]

        if len(records) < 7:
            return {"sku_id": sku_id, "anomalies": [], "anomaly_count": 0, "recommendation": "insufficient_data"}

        quantities = [r["quantity"] for r in records]
        mean_q     = statistics.mean(quantities)
        std_q      = statistics.stdev(quantities) if len(quantities) > 1 else 1.0

        anomalies = []
        for rec in records:
            qty = rec["quantity"]
            z   = (qty - mean_q) / std_q if std_q > 0 else 0.0
            if abs(z) >= z_score_threshold:
                anomalies.append({
                    "date":     rec["order_date"],
                    "quantity": qty,
                    "z_score":  round(z, 2),
                    "type":     "spike" if z > 0 else "drop",
                })

        rec_map = {
            0: "clean",
            1: "review",
        }
        recommendation = rec_map.get(
            len(anomalies),
            "exclude_outliers" if len(anomalies) > 1 else "review"
        )

        return {
            "sku_id":          sku_id,
            "lookback_days":   lookback_days,
            "mean_daily":      round(mean_q, 2),
            "std_daily":       round(std_q, 2),
            "anomalies":       anomalies,
            "anomaly_count":   len(anomalies),
            "recommendation":  recommendation,
        }

    except Exception as exc:
        return {"sku_id": sku_id, "error": str(exc), "anomalies": []}


# ═══════════════════════════════════════════════════════════════════
# Tool 5 — get_demand_trend
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=GetDemandTrendArgs)
def get_demand_trend(sku_id: str, days: int = 60) -> dict:
    """
    Analyse the demand trend for a SKU over a rolling window.

    Uses linear regression on the daily demand series to determine
    the slope and classify the overall trend direction.

    Use this tool when:
    - The orchestrator needs a quick signal before running a full forecast
    - You want to set appropriate safety stock levels (rising trend = more stock)
    - Deciding whether to run a full Prophet forecast or use a simpler model

    Returns:
      - trend: 'rising' | 'falling' | 'stable' | 'volatile'
      - slope_units_per_day: float (positive = rising)
      - pct_change: float (percentage change first→last week)
      - avg_daily: float
      - coefficient_of_variation: float (CV — higher = more volatile)
    """
    try:
        data    = load_demand_history(sku_id=sku_id, days=days)
        records = data.get("records", [])

        if len(records) < 7:
            return {"sku_id": sku_id, "trend": "unknown", "error": "Insufficient data"}

        quantities = [r["quantity"] for r in records]
        n          = len(quantities)
        mean_q     = statistics.mean(quantities)
        std_q      = statistics.stdev(quantities) if n > 1 else 0.0
        cv         = round(std_q / mean_q, 4) if mean_q > 0 else 0.0

        # Simple linear regression: slope via least squares
        x_mean = (n - 1) / 2
        num    = sum((i - x_mean) * (q - mean_q) for i, q in enumerate(quantities))
        den    = sum((i - x_mean) ** 2 for i in range(n))
        slope  = round(num / den, 4) if den > 0 else 0.0

        # Week-over-week pct change
        first_week = statistics.mean(quantities[:7])
        last_week  = statistics.mean(quantities[-7:])
        pct_change = round((last_week - first_week) / first_week * 100, 2) if first_week > 0 else 0.0

        # Classify
        if cv > 0.50:
            trend = "volatile"
        elif pct_change > 10:
            trend = "rising"
        elif pct_change < -10:
            trend = "falling"
        else:
            trend = "stable"

        return {
            "sku_id":                   sku_id,
            "days_analysed":            n,
            "trend":                    trend,
            "slope_units_per_day":      slope,
            "pct_change":               pct_change,
            "avg_daily":                round(mean_q, 2),
            "coefficient_of_variation": cv,
        }

    except Exception as exc:
        return {"sku_id": sku_id, "trend": "unknown", "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════
# Tool registry — import this in agent files
# ═══════════════════════════════════════════════════════════════════

FORECAST_TOOLS = [
    load_demand_history,
    run_prophet_forecast,
    compute_mape,
    detect_demand_anomalies,
    get_demand_trend,
]
