"""
models/demand.py
────────────────
Pydantic v2 schemas for the Demand Forecast Agent.

Covers:
  - ForecastRequest   → what the agent receives as input
  - DemandSignal      → a single historical / real-time demand data point
  - SeasonalityConfig → optional tuning for Prophet
  - ForecastDataPoint → one point in the output forecast series
  - ForecastResult    → full agent output (forecast + metadata + reasoning)
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ──────────────────────────────────────────────────────────────────────

class ForecastMethod(str, Enum):
    PROPHET = "prophet"
    ARIMA   = "arima"
    LLM     = "llm"            # pure LLM reasoning (fallback / short horizon)
    HYBRID  = "hybrid"         # ML forecast + LLM adjustment layer


class DemandTrend(str, Enum):
    RISING   = "rising"
    FALLING  = "falling"
    STABLE   = "stable"
    VOLATILE = "volatile"


class ConfidenceLevel(str, Enum):
    HIGH   = "high"    # ±10%
    MEDIUM = "medium"  # ±20%
    LOW    = "low"     # ±35%+


# ── Input Schemas ──────────────────────────────────────────────────────────────

class DemandSignal(BaseModel):
    """
    A single historical demand observation.
    Maps directly to a row in the orders/demand table.
    """
    date:     date  = Field(..., description="Observation date")
    sku_id:   str   = Field(..., description="Stock-keeping unit identifier")
    quantity: float = Field(..., ge=0, description="Units demanded on this date")
    channel:  str   = Field("default", description="Sales channel (e.g. online, retail, wholesale)")
    region:   Optional[str] = Field(None, description="Geographic region if available")

    @field_validator("sku_id")
    @classmethod
    def sku_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("sku_id cannot be blank")
        return v.upper().strip()


class SeasonalityConfig(BaseModel):
    """
    Optional overrides for Prophet seasonality model.
    If omitted, Prophet auto-detects.
    """
    weekly_seasonality:  bool  = Field(True,  description="Enable weekly pattern")
    yearly_seasonality:  bool  = Field(True,  description="Enable yearly pattern")
    daily_seasonality:   bool  = Field(False, description="Enable daily pattern (high-freq data only)")
    changepoint_scale:   float = Field(0.05,  ge=0.001, le=1.0,
                                        description="Flexibility of trend changepoints (0=rigid, 1=very flexible)")
    seasonality_scale:   float = Field(10.0,  ge=0.1,
                                        description="Strength of seasonality components")


class ForecastRequest(BaseModel):
    """
    Input contract for the Demand Forecast Agent.
    The Orchestrator builds and passes this to the agent.
    """
    sku_id:           str              = Field(..., description="Target SKU to forecast")
    horizon_days:     int              = Field(90, ge=1, le=365,
                                               description="How many days ahead to forecast")
    history:          list[DemandSignal] = Field(..., min_length=7,
                                                  description="Historical demand signals (≥7 days required)")
    method:           ForecastMethod   = Field(ForecastMethod.HYBRID,
                                               description="Forecasting method to use")
    seasonality:      Optional[SeasonalityConfig] = Field(None,
                                                           description="Prophet seasonality overrides")
    external_factors: Optional[dict[str, float]]  = Field(
        None,
        description="Optional external regressors e.g. {'promo_flag': 1.0, 'price_index': 0.92}"
    )

    @model_validator(mode="after")
    def validate_history_length(self) -> "ForecastRequest":
        if len(self.history) < 7:
            raise ValueError(
                f"At least 7 historical data points required, got {len(self.history)}"
            )
        return self


# ── Output Schemas ─────────────────────────────────────────────────────────────

class ForecastDataPoint(BaseModel):
    """One day in the output forecast series."""
    date:           date  = Field(..., description="Forecasted date")
    yhat:           float = Field(..., description="Point forecast (units)")
    yhat_lower:     float = Field(..., description="Lower confidence bound")
    yhat_upper:     float = Field(..., description="Upper confidence bound")
    is_anomaly:     bool  = Field(False, description="True if this point is flagged as unusual")


class ForecastResult(BaseModel):
    """
    Full output from the Demand Forecast Agent.
    Stored in WorkflowState and passed downstream to Supply Planning Agent.
    """
    sku_id:              str                    = Field(..., description="SKU that was forecasted")
    generated_at:        datetime               = Field(default_factory=datetime.utcnow)
    method_used:         ForecastMethod         = Field(..., description="Method that produced this forecast")
    horizon_days:        int                    = Field(..., description="Forecast horizon in days")
    forecast:            list[ForecastDataPoint] = Field(..., description="Day-by-day forecast series")

    # Aggregate KPIs derived from the series
    total_forecasted_demand: float = Field(..., description="Sum of yhat across horizon")
    avg_daily_demand:        float = Field(..., description="Mean daily yhat")
    peak_demand_date:        date  = Field(..., description="Date with highest yhat")
    peak_demand_units:       float = Field(..., description="Units on peak date")

    # Intelligence layer
    trend:           DemandTrend    = Field(..., description="Overall demand trend signal")
    confidence:      ConfidenceLevel = Field(..., description="Forecast confidence classification")
    mape:            Optional[float] = Field(None, ge=0, description="MAPE on holdout set if computed")
    llm_commentary:  Optional[str]  = Field(None, description="LLM-generated plain-English demand insight")
    warnings:        list[str]      = Field(default_factory=list,
                                            description="Any data quality or model warnings")

    @property
    def daily_demand_by_date(self) -> dict[date, float]:
        """Convenience lookup: date → yhat."""
        return {pt.date: pt.yhat for pt in self.forecast}
