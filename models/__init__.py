"""
models/__init__.py
──────────────────
Re-export all Pydantic schemas for convenient one-line imports:

    from models import ForecastResult, PurchaseOrder, WorkflowRun
"""

# ── Demand ──────────────────────────────────────────────────────────────────
from models.demand import (
    ConfidenceLevel,
    DemandSignal,
    DemandTrend,
    ForecastDataPoint,
    ForecastMethod,
    ForecastRequest,
    ForecastResult,
    SeasonalityConfig,
)

# ── Supply ──────────────────────────────────────────────────────────────────
from models.supply import (
    EOQInput,
    EOQResult,
    POStatus,
    PurchaseOrder,
    PurchaseOrderLine,
    ReplenishmentUrgency,
    ReorderPoint,
    SupplierInfo,
    SupplierStatus,
    SupplyPlanResult,
)

# ── Inventory ───────────────────────────────────────────────────────────────
from models.inventory import (
    HealthStatus,
    InventoryHealth,
    InventorySnapshot,
    MovementType,
    SKU,
    SKUStatus,
    StockLevel,
    StockMovement,
    StorageType,
)

# ── Workflow ─────────────────────────────────────────────────────────────────
from models.workflow import (
    AgentName,
    AgentResult,
    AgentStatus,
    ExceptionEvent,
    ExceptionSeverity,
    ExceptionType,
    HITLAction,
    HITLCheckpoint,
    WorkflowRun,
    WorkflowStatus,
)

__all__ = [
    # demand
    "ConfidenceLevel", "DemandSignal", "DemandTrend", "ForecastDataPoint",
    "ForecastMethod", "ForecastRequest", "ForecastResult", "SeasonalityConfig",
    # supply
    "EOQInput", "EOQResult", "POStatus", "PurchaseOrder", "PurchaseOrderLine",
    "ReplenishmentUrgency", "ReorderPoint", "SupplierInfo", "SupplierStatus",
    "SupplyPlanResult",
    # inventory
    "HealthStatus", "InventoryHealth", "InventorySnapshot", "MovementType",
    "SKU", "SKUStatus", "StockLevel", "StockMovement", "StorageType",
    # workflow
    "AgentName", "AgentResult", "AgentStatus", "ExceptionEvent",
    "ExceptionSeverity", "ExceptionType", "HITLAction", "HITLCheckpoint",
    "WorkflowRun", "WorkflowStatus",
]
