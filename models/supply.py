"""
models/supply.py
────────────────
Pydantic v2 schemas for the Supply Planning Agent.

Covers:
  - SupplierInfo      → vendor master data
  - EOQInput          → inputs to the Economic Order Quantity formula
  - EOQResult         → computed EOQ + reorder metrics
  - ReorderPoint      → reorder trigger threshold per SKU
  - PurchaseOrderLine → single line item on a PO
  - PurchaseOrder     → full PO document sent to a supplier
  - SupplyPlanResult  → full agent output
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, field_validator


# ── Enums ──────────────────────────────────────────────────────────────────────

class SupplierStatus(str, Enum):
    ACTIVE    = "active"
    ON_HOLD   = "on_hold"
    PREFERRED = "preferred"
    BLACKLIST = "blacklisted"


class POStatus(str, Enum):
    DRAFT     = "draft"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    SHIPPED   = "shipped"
    RECEIVED  = "received"
    CANCELLED = "cancelled"


class ReplenishmentUrgency(str, Enum):
    CRITICAL  = "critical"   # stock-out imminent (< 3 days)
    HIGH      = "high"       # below safety stock
    MEDIUM    = "medium"     # approaching reorder point
    LOW       = "low"        # healthy — plan ahead


# ── Supplier ───────────────────────────────────────────────────────────────────

class SupplierInfo(BaseModel):
    """Vendor master record."""
    supplier_id:       str            = Field(..., description="Unique supplier ID")
    name:              str            = Field(..., description="Supplier legal name")
    country:           str            = Field(..., description="Country of origin")
    lead_time_days:    int            = Field(..., ge=0, description="Average lead time in days")
    min_order_qty:     float          = Field(0.0, ge=0, description="Minimum order quantity (units)")
    unit_cost:         float          = Field(..., gt=0, description="Cost per unit (USD)")
    reliability_score: float          = Field(1.0, ge=0.0, le=1.0,
                                              description="On-time delivery rate 0–1")
    status:            SupplierStatus = Field(SupplierStatus.ACTIVE)
    contact_email:     Optional[str]  = Field(None)

    @field_validator("supplier_id")
    @classmethod
    def normalise_id(cls, v: str) -> str:
        return v.upper().strip()


# ── EOQ ────────────────────────────────────────────────────────────────────────

class EOQInput(BaseModel):
    """
    Inputs to the Economic Order Quantity (EOQ) formula.

        EOQ = sqrt( (2 × D × S) / H )

    Where:
        D = annual demand (units)
        S = ordering cost per order (USD)
        H = holding cost per unit per year (USD)
    """
    sku_id:             str   = Field(..., description="SKU being optimised")
    annual_demand:      float = Field(..., gt=0, description="D — forecasted annual demand (units)")
    ordering_cost:      float = Field(..., gt=0, description="S — cost to place one order (USD)")
    holding_cost:       float = Field(..., gt=0, description="H — annual holding cost per unit (USD)")
    unit_cost:          float = Field(..., gt=0, description="Purchase cost per unit (USD)")
    lead_time_days:     int   = Field(..., ge=0, description="Supplier lead time (days)")
    safety_stock_days:  int   = Field(14, ge=0,
                                      description="Buffer stock expressed in days of supply")


class EOQResult(BaseModel):
    """Computed EOQ and derived replenishment metrics."""
    sku_id:             str   = Field(...)
    eoq:                float = Field(..., description="Optimal order quantity (units)")
    reorder_point:      float = Field(..., description="Units on-hand that trigger a new order")
    safety_stock:       float = Field(..., description="Buffer stock (units)")
    annual_order_count: float = Field(..., description="How many POs per year at EOQ")
    total_annual_cost:  float = Field(..., description="Total inventory cost at EOQ (USD/year)")
    days_between_orders: float = Field(..., description="Average days between replenishment orders")
    computed_at:        datetime = Field(default_factory=datetime.utcnow)


# ── Reorder Point ──────────────────────────────────────────────────────────────

class ReorderPoint(BaseModel):
    """
    Per-SKU reorder trigger maintained by the Supply Planning Agent.
    Updated every time a new forecast is received.
    """
    sku_id:            str                  = Field(...)
    reorder_at_units:  float                = Field(..., ge=0,
                                                    description="Place a PO when on-hand stock hits this level")
    safety_stock:      float                = Field(..., ge=0)
    eoq:               float                = Field(..., gt=0)
    supplier_id:       str                  = Field(..., description="Preferred supplier for this SKU")
    urgency:           ReplenishmentUrgency = Field(ReplenishmentUrgency.LOW)
    last_updated:      datetime             = Field(default_factory=datetime.utcnow)
    next_review_date:  Optional[date]       = Field(None)


# ── Purchase Order ─────────────────────────────────────────────────────────────

class PurchaseOrderLine(BaseModel):
    """One line item on a Purchase Order."""
    line_number: int   = Field(..., ge=1)
    sku_id:      str   = Field(...)
    description: str   = Field("")
    quantity:    float = Field(..., gt=0)
    unit_cost:   float = Field(..., gt=0)

    @computed_field
    @property
    def line_total(self) -> float:
        return round(self.quantity * self.unit_cost, 2)


class PurchaseOrder(BaseModel):
    """
    Full Purchase Order document generated by the Procurement Agent
    based on Supply Planning Agent recommendations.
    """
    po_number:        str                  = Field(
                                               default_factory=lambda: f"PO-{uuid4().hex[:8].upper()}",
                                               description="Unique PO identifier")
    supplier_id:      str                  = Field(...)
    supplier_name:    str                  = Field("")
    status:           POStatus             = Field(POStatus.DRAFT)
    lines:            list[PurchaseOrderLine] = Field(..., min_length=1)
    issued_date:      date                 = Field(default_factory=date.today)
    expected_date:    Optional[date]       = Field(None, description="Expected delivery date")
    notes:            Optional[str]        = Field(None)
    created_by_agent: str                  = Field("procurement_agent")
    approved_by:      Optional[str]        = Field(None, description="Human approver (HITL)")
    approved_at:      Optional[datetime]   = Field(None)

    @computed_field
    @property
    def total_value(self) -> float:
        return round(sum(line.line_total for line in self.lines), 2)

    @computed_field
    @property
    def total_units(self) -> float:
        return sum(line.quantity for line in self.lines)


# ── Agent Output ───────────────────────────────────────────────────────────────

class SupplyPlanResult(BaseModel):
    """
    Full output from the Supply Planning Agent.
    Passed to the Procurement Agent and stored in WorkflowState.
    """
    sku_id:          str                    = Field(...)
    generated_at:    datetime               = Field(default_factory=datetime.utcnow)
    eoq_result:      EOQResult              = Field(...)
    reorder_point:   ReorderPoint           = Field(...)
    recommended_pos: list[PurchaseOrder]    = Field(default_factory=list,
                                                    description="Suggested POs if replenishment is needed")
    urgency:         ReplenishmentUrgency   = Field(...)
    llm_commentary:  Optional[str]          = Field(None,
                                                    description="LLM plain-English supply plan rationale")
    warnings:        list[str]              = Field(default_factory=list)
