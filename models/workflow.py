"""
models/workflow.py
──────────────────
Pydantic v2 schemas for the overall workflow orchestration layer.

Covers:
  - AgentStatus       → lifecycle state of a single agent execution
  - AgentResult       → output wrapper for any agent (typed union)
  - ExceptionEvent    → supply chain disruption / alert record
  - HITLCheckpoint    → human-in-the-loop review request
  - WorkflowRun       → top-level run record (stored + returned by API)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field


# ── Enums ──────────────────────────────────────────────────────────────────────

class AgentName(str, Enum):
    ORCHESTRATOR    = "orchestrator"
    DEMAND_FORECAST = "demand_forecast"
    SUPPLY_PLANNING = "supply_planning"
    INVENTORY       = "inventory"
    PROCUREMENT     = "procurement"
    FULFILLMENT     = "fulfillment"
    EXCEPTION       = "exception_handler"


class AgentStatus(str, Enum):
    PENDING   = "pending"    # not yet started
    RUNNING   = "running"    # currently executing
    SUCCESS   = "success"    # completed without error
    FAILED    = "failed"     # terminated with error
    SKIPPED   = "skipped"    # skipped by orchestrator (condition not met)
    WAITING   = "waiting"    # paused at HITL checkpoint


class WorkflowStatus(str, Enum):
    INITIALISED = "initialised"
    RUNNING     = "running"
    PAUSED      = "paused"          # at HITL checkpoint
    COMPLETED   = "completed"
    FAILED      = "failed"
    CANCELLED   = "cancelled"


class ExceptionSeverity(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    HIGH     = "high"
    CRITICAL = "critical"


class ExceptionType(str, Enum):
    DEMAND_SPIKE       = "demand_spike"
    DEMAND_DROP        = "demand_drop"
    SUPPLY_DISRUPTION  = "supply_disruption"
    STOCKOUT_IMMINENT  = "stockout_imminent"
    SUPPLIER_DELAY     = "supplier_delay"
    OVERSTOCK          = "overstock"
    DATA_QUALITY       = "data_quality"
    AGENT_FAILURE      = "agent_failure"


class HITLAction(str, Enum):
    APPROVE  = "approve"
    REJECT   = "reject"
    MODIFY   = "modify"
    ESCALATE = "escalate"


# ── Agent Result ───────────────────────────────────────────────────────────────

class AgentResult(BaseModel):
    """
    Standardised wrapper around any agent's output.
    Stored in the LangGraph state under `agent_results`.

    The `output` field holds the raw Pydantic model (ForecastResult,
    SupplyPlanResult, InventorySnapshot, etc.) serialised as a dict.
    We keep it as `Any` so the graph state stays serialisable.
    """
    agent:        AgentName  = Field(...)
    status:       AgentStatus = Field(...)
    output:       Optional[Any]  = Field(None, description="Agent-specific result payload (dict)")
    error:        Optional[str]  = Field(None, description="Error message if status=FAILED")
    started_at:   Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    iteration:    int            = Field(1, ge=1, description="Which iteration (for retry tracking)")
    tokens_used:  Optional[int]  = Field(None, description="LLM tokens consumed by this agent")

    @computed_field
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return round((self.completed_at - self.started_at).total_seconds(), 2)
        return None

    @computed_field
    @property
    def succeeded(self) -> bool:
        return self.status == AgentStatus.SUCCESS


# ── Exception Event ────────────────────────────────────────────────────────────

class ExceptionEvent(BaseModel):
    """
    A supply chain disruption or anomaly detected during a workflow run.
    Raised by any agent; handled by the Exception Agent.
    """
    event_id:     str               = Field(default_factory=lambda: str(uuid4()))
    run_id:       str               = Field(...)
    raised_by:    AgentName         = Field(..., description="Which agent raised this exception")
    exception_type: ExceptionType   = Field(...)
    severity:     ExceptionSeverity = Field(...)
    sku_id:       Optional[str]     = Field(None)
    supplier_id:  Optional[str]     = Field(None)
    description:  str               = Field(..., description="Human-readable description of the issue")
    context:      dict[str, Any]    = Field(default_factory=dict,
                                           description="Structured data relevant to this event")
    resolved:     bool              = Field(False)
    resolution:   Optional[str]     = Field(None)
    raised_at:    datetime          = Field(default_factory=datetime.utcnow)
    resolved_at:  Optional[datetime] = Field(None)

    def resolve(self, resolution: str) -> None:
        self.resolved    = True
        self.resolution  = resolution
        self.resolved_at = datetime.utcnow()


# ── HITL Checkpoint ────────────────────────────────────────────────────────────

class HITLCheckpoint(BaseModel):
    """
    A human-in-the-loop pause point within the workflow.
    The Orchestrator writes this to state; the API surfaces it
    to the user; the user's response resumes the graph.
    """
    checkpoint_id:  str           = Field(default_factory=lambda: str(uuid4()))
    run_id:         str           = Field(...)
    agent:          AgentName     = Field(..., description="Agent waiting for human decision")
    prompt:         str           = Field(..., description="Question or action requiring human input")
    context:        dict[str, Any] = Field(default_factory=dict,
                                           description="Data the human needs to make a decision")
    required_action: HITLAction   = Field(HITLAction.APPROVE)
    response:       Optional[str]  = Field(None, description="Human's free-text response")
    action_taken:   Optional[HITLAction] = Field(None)
    created_at:     datetime       = Field(default_factory=datetime.utcnow)
    responded_at:   Optional[datetime] = Field(None)
    timeout_minutes: int           = Field(60, description="Auto-escalate after this many minutes")

    @computed_field
    @property
    def is_pending(self) -> bool:
        return self.action_taken is None


# ── Workflow Run ───────────────────────────────────────────────────────────────

class WorkflowRun(BaseModel):
    """
    Top-level record for a single end-to-end workflow execution.
    Created when the API receives a trigger request and stored
    throughout the run. Returned in full by GET /workflow/{run_id}.
    """
    run_id:          str            = Field(default_factory=lambda: f"RUN-{uuid4().hex[:8].upper()}")
    status:          WorkflowStatus = Field(WorkflowStatus.INITIALISED)
    sku_ids:         list[str]      = Field(..., min_length=1,
                                           description="SKUs included in this run")
    triggered_by:    str            = Field("api", description="api | scheduler | manual | test")
    agent_results:   list[AgentResult]   = Field(default_factory=list)
    exceptions:      list[ExceptionEvent] = Field(default_factory=list)
    hitl_checkpoints: list[HITLCheckpoint] = Field(default_factory=list)
    started_at:      Optional[datetime] = Field(None)
    completed_at:    Optional[datetime] = Field(None)
    created_at:      datetime       = Field(default_factory=datetime.utcnow)
    metadata:        dict[str, Any] = Field(default_factory=dict,
                                           description="Arbitrary run-level metadata")

    # ── Convenience helpers ────────────────────────────────────────────────────

    @computed_field
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return round((self.completed_at - self.started_at).total_seconds(), 2)
        return None

    @computed_field
    @property
    def total_tokens_used(self) -> int:
        return sum(r.tokens_used or 0 for r in self.agent_results)

    @computed_field
    @property
    def has_pending_hitl(self) -> bool:
        return any(cp.is_pending for cp in self.hitl_checkpoints)

    @computed_field
    @property
    def critical_exceptions(self) -> list[ExceptionEvent]:
        return [e for e in self.exceptions if e.severity == ExceptionSeverity.CRITICAL]

    def get_agent_result(self, agent: AgentName) -> Optional[AgentResult]:
        """Retrieve the latest result for a given agent."""
        results = [r for r in self.agent_results if r.agent == agent]
        return results[-1] if results else None

    def mark_started(self) -> None:
        self.status     = WorkflowStatus.RUNNING
        self.started_at = datetime.utcnow()

    def mark_completed(self) -> None:
        self.status       = WorkflowStatus.COMPLETED
        self.completed_at = datetime.utcnow()

    def mark_failed(self) -> None:
        self.status       = WorkflowStatus.FAILED
        self.completed_at = datetime.utcnow()
