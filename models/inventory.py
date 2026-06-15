"""
models/inventory.py
────────────────────
Pydantic v2 schemas for the Inventory Agent.

Covers:
  - SKU               → product master record
  - StockLevel        → current on-hand snapshot per SKU per location
  - StockMovement     → inbound/outbound inventory transaction
  - InventoryHealth   → computed health score + risk flags per SKU
  - InventorySnapshot → full agent output (collection of all SKU health records)
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, field_validator


# ── Enums ──────────────────────────────────────────────────────────────────────

class SKUStatus(str, Enum):
    ACTIVE       = "active"
    DISCONTINUED = "discontinued"
    SEASONAL     = "seasonal"
    NEW          = "new"


class MovementType(str, Enum):
    RECEIPT      = "receipt"       # goods received from supplier
    SALE         = "sale"          # goods sold / fulfilled
    ADJUSTMENT   = "adjustment"    # manual stock adjustment
    TRANSFER_IN  = "transfer_in"   # moved in from another location
    TRANSFER_OUT = "transfer_out"  # moved out to another location
    RETURN       = "return"        # customer return
    WRITE_OFF    = "write_off"     # damaged / expired


class HealthStatus(str, Enum):
    HEALTHY      = "healthy"       # score ≥ 0.75
    AT_RISK      = "at_risk"       # score 0.40–0.74
    CRITICAL     = "critical"      # score < 0.40
    STOCK_OUT    = "stock_out"     # on-hand = 0


class StorageType(str, Enum):
    AMBIENT      = "ambient"
    REFRIGERATED = "refrigerated"
    FROZEN       = "frozen"
    HAZMAT       = "hazmat"


# ── Product Master ─────────────────────────────────────────────────────────────

class SKU(BaseModel):
    """
    Product master record.
    One row per sellable item in the catalogue.
    """
    sku_id:          str          = Field(..., description="Unique stock-keeping unit ID")
    name:            str          = Field(..., description="Product display name")
    category:        str          = Field(..., description="Product category (e.g. Electronics, Apparel)")
    unit_of_measure: str          = Field("EA", description="EA=each, KG, L, BOX, etc.")
    unit_cost:       float        = Field(..., gt=0, description="Standard cost per unit (USD)")
    unit_price:      float        = Field(..., gt=0, description="Selling price per unit (USD)")
    weight_kg:       Optional[float] = Field(None, ge=0)
    storage_type:    StorageType  = Field(StorageType.AMBIENT)
    shelf_life_days: Optional[int]   = Field(None, ge=1,
                                             description="Shelf life in days (None = non-perishable)")
    status:          SKUStatus    = Field(SKUStatus.ACTIVE)
    created_at:      datetime     = Field(default_factory=datetime.utcnow)

    @field_validator("sku_id")
    @classmethod
    def normalise_sku(cls, v: str) -> str:
        return v.upper().strip()

    @computed_field
    @property
    def gross_margin_pct(self) -> float:
        """Gross margin percentage."""
        return round((self.unit_price - self.unit_cost) / self.unit_price * 100, 2)


# ── Stock Level ────────────────────────────────────────────────────────────────

class StockLevel(BaseModel):
    """
    Current on-hand inventory snapshot for a SKU at a given location.
    Refreshed by the Inventory Agent on each workflow run.
    """
    sku_id:           str            = Field(...)
    location_id:      str            = Field("DC-01", description="Warehouse / DC code")
    on_hand:          float          = Field(..., ge=0, description="Units physically on hand")
    reserved:         float          = Field(0.0, ge=0,
                                             description="Units allocated to open orders (not available)")
    in_transit:       float          = Field(0.0, ge=0,
                                             description="Units on order but not yet received")
    last_updated:     datetime       = Field(default_factory=datetime.utcnow)
    expiry_date:      Optional[date] = Field(None,
                                             description="Earliest expiry date for perishable batches")

    @computed_field
    @property
    def available(self) -> float:
        """Units available to promise (ATP)."""
        return max(0.0, self.on_hand - self.reserved)

    @computed_field
    @property
    def total_supply(self) -> float:
        """On-hand + in-transit (projected stock)."""
        return self.on_hand + self.in_transit


# ── Stock Movement ─────────────────────────────────────────────────────────────

class StockMovement(BaseModel):
    """
    A single inventory transaction event.
    Positive delta = stock increases; negative delta = stock decreases.
    """
    movement_id:   str          = Field(default_factory=lambda: str(uuid4()))
    sku_id:        str          = Field(...)
    location_id:   str          = Field("DC-01")
    movement_type: MovementType = Field(...)
    quantity:      float        = Field(..., description="Absolute quantity moved (always positive)")
    reference_doc: Optional[str] = Field(None, description="PO number, SO number, or adjustment ref")
    notes:         Optional[str] = Field(None)
    occurred_at:   datetime     = Field(default_factory=datetime.utcnow)
    recorded_by:   str          = Field("inventory_agent")

    @computed_field
    @property
    def delta(self) -> float:
        """
        Net impact on on-hand stock.
        Outbound movements return a negative value.
        """
        outbound = {
            MovementType.SALE,
            MovementType.TRANSFER_OUT,
            MovementType.WRITE_OFF,
        }
        return -self.quantity if self.movement_type in outbound else self.quantity


# ── Inventory Health ───────────────────────────────────────────────────────────

class InventoryHealth(BaseModel):
    """
    Computed health record for a single SKU.
    The Inventory Agent produces one of these per SKU per run.
    """
    sku_id:            str           = Field(...)
    location_id:       str           = Field("DC-01")
    stock_level:       StockLevel    = Field(...)
    health_status:     HealthStatus  = Field(...)
    health_score:      float         = Field(..., ge=0.0, le=1.0,
                                            description="Composite 0–1 score (1 = perfectly healthy)")
    days_of_supply:    float         = Field(..., ge=0,
                                            description="Days until stock-out at current daily demand rate")
    reorder_triggered: bool          = Field(False,
                                            description="True if stock has crossed the reorder point")
    stockout_risk_date: Optional[date] = Field(None,
                                               description="Projected stock-out date if no replenishment")
    overstock_flag:    bool          = Field(False,
                                            description="True if on-hand > 2× reorder point (capital tied up)")
    expiry_risk_flag:  bool          = Field(False,
                                            description="True if perishable stock expires within lead time")
    llm_commentary:    Optional[str] = Field(None,
                                            description="LLM narrative on inventory risk")
    computed_at:       datetime      = Field(default_factory=datetime.utcnow)

    @field_validator("health_score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 4)


# ── Agent Output ───────────────────────────────────────────────────────────────

class InventorySnapshot(BaseModel):
    """
    Full output from the Inventory Agent for a single workflow run.
    Contains health records for every SKU that was evaluated.
    """
    run_id:          str                    = Field(default_factory=lambda: str(uuid4()))
    generated_at:    datetime               = Field(default_factory=datetime.utcnow)
    location_id:     str                    = Field("DC-01")
    records:         list[InventoryHealth]  = Field(..., description="One record per evaluated SKU")
    total_skus:      int                    = Field(0)
    critical_count:  int                    = Field(0, description="SKUs in CRITICAL or STOCK_OUT status")
    at_risk_count:   int                    = Field(0, description="SKUs in AT_RISK status")
    healthy_count:   int                    = Field(0, description="SKUs in HEALTHY status")
    reorder_triggers: list[str]             = Field(default_factory=list,
                                                    description="SKU IDs that triggered reorder this run")
    warnings:        list[str]              = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        """Auto-compute aggregate counts from records."""
        self.total_skus     = len(self.records)
        self.critical_count = sum(
            1 for r in self.records
            if r.health_status in (HealthStatus.CRITICAL, HealthStatus.STOCK_OUT)
        )
        self.at_risk_count  = sum(1 for r in self.records if r.health_status == HealthStatus.AT_RISK)
        self.healthy_count  = sum(1 for r in self.records if r.health_status == HealthStatus.HEALTHY)
        self.reorder_triggers = [r.sku_id for r in self.records if r.reorder_triggered]
