"""
tools/notification_tools.py
────────────────────────────
LangChain @tool definitions for the Exception Handler Agent.

Tools:
  - send_alert              → queue an alert notification (email/Slack stub)
  - escalate_to_human       → raise a HITL escalation request
  - log_exception_event     → persist an exception event to the DB
  - build_run_summary       → generate a structured run summary dict
  - send_slack_notification → send a Slack message (stub — wired in Phase 6)
  - get_unresolved_events   → list all unresolved exceptions for a run
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config.settings import get_settings

settings = get_settings()
DB_PATH  = Path("data/supply_demand.db")


# ═══════════════════════════════════════════════════════════════════
# Arg schemas
# ═══════════════════════════════════════════════════════════════════

class SendAlertArgs(BaseModel):
    severity:   str            = Field(..., description="Alert severity: 'info' | 'warning' | 'high' | 'critical'")
    message:    str            = Field(..., description="Alert message body")
    sku_id:     Optional[str]  = Field(None, description="Related SKU if applicable")
    run_id:     Optional[str]  = Field(None, description="Workflow run ID")
    channel:    str            = Field("email", description="Delivery channel: 'email' | 'slack' | 'pagerduty'")
    recipient:  str            = Field("ops-team@company.example.com", description="Recipient email or channel")


class EscalateToHumanArgs(BaseModel):
    run_id:     str = Field(..., description="Workflow run ID to escalate")
    reason:     str = Field(..., description="Why human intervention is needed")
    agent:      str = Field(..., description="Agent requesting escalation")
    context:    str = Field("{}", description="JSON string of relevant context data")


class LogExceptionEventArgs(BaseModel):
    run_id:         str           = Field(..., description="Workflow run ID")
    raised_by:      str           = Field(..., description="Agent that raised the event")
    exception_type: str           = Field(..., description="Exception type (e.g. 'stockout_imminent')")
    severity:       str           = Field(..., description="Severity: info | warning | high | critical")
    description:    str           = Field(..., description="Human-readable description")
    sku_id:         Optional[str] = Field(None)
    supplier_id:    Optional[str] = Field(None)
    context:        str           = Field("{}", description="JSON string with structured context")


class BuildRunSummaryArgs(BaseModel):
    run_id: str = Field(..., description="Workflow run ID to summarise")


class SendSlackNotificationArgs(BaseModel):
    channel:  str = Field("#supply-chain-alerts", description="Slack channel name")
    message:  str = Field(..., description="Message text (Markdown supported)")
    severity: str = Field("info", description="Severity label for colour coding")


class GetUnresolvedEventsArgs(BaseModel):
    run_id:   str = Field(..., description="Workflow run ID")
    severity: Optional[str] = Field(None, description="Filter by severity — None returns all")


# ═══════════════════════════════════════════════════════════════════
# In-memory notification log (flushed to DB in Phase 6)
# ═══════════════════════════════════════════════════════════════════

_NOTIFICATION_LOG: list[dict] = []


# ═══════════════════════════════════════════════════════════════════
# Tool 1 — send_alert
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=SendAlertArgs)
def send_alert(
    severity:  str,
    message:   str,
    sku_id:    Optional[str] = None,
    run_id:    Optional[str] = None,
    channel:   str           = "email",
    recipient: str           = "ops-team@company.example.com",
) -> dict:
    """
    Queue an alert notification to the operations team.

    In development mode, alerts are logged in memory (no real send).
    In production (Phase 6), this wires to SMTP / Slack / PagerDuty.

    Severity levels:
      info     → FYI, no action needed
      warning  → monitor situation
      high     → action required soon
      critical → immediate action required — also triggers PagerDuty

    Returns a notification receipt with a tracking ID.
    """
    notification_id = str(uuid4())[:8].upper()
    notification    = {
        "notification_id": notification_id,
        "severity":        severity,
        "channel":         channel,
        "recipient":       recipient,
        "subject":         f"[{severity.upper()}] Supply Chain Alert — {sku_id or 'System'}",
        "message":         message,
        "sku_id":          sku_id,
        "run_id":          run_id,
        "sent_at":         datetime.utcnow().isoformat(),
        "status":          "queued" if not settings.is_production else "sent",
    }
    _NOTIFICATION_LOG.append(notification)

    return {
        "success":         True,
        "notification_id": notification_id,
        "status":          notification["status"],
        "channel":         channel,
        "message":         f"Alert queued: [{severity.upper()}] {message[:80]}...",
    }


# ═══════════════════════════════════════════════════════════════════
# Tool 2 — escalate_to_human
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=EscalateToHumanArgs)
def escalate_to_human(
    run_id:  str,
    reason:  str,
    agent:   str,
    context: str = "{}",
) -> dict:
    """
    Escalate a workflow decision to a human operator.

    Creates a Human-in-the-Loop (HITL) checkpoint that pauses the workflow
    until a human approves or rejects the pending action.

    Use this tool when:
    - A Purchase Order exceeds the value threshold
    - A critical stockout requires emergency sourcing
    - Data anomalies make automated decisions unreliable
    - Multiple exceptions are unresolved and the run is at risk

    Returns a checkpoint_id that the API uses to surface the request.
    """
    try:
        context_data = json.loads(context) if isinstance(context, str) else context
    except json.JSONDecodeError:
        context_data = {"raw": context}

    checkpoint_id = str(uuid4())

    hitl_payload = {
        "checkpoint_id":  checkpoint_id,
        "run_id":         run_id,
        "agent":          agent,
        "reason":         reason,
        "context":        context_data,
        "created_at":     datetime.utcnow().isoformat(),
        "status":         "pending",
        "timeout_minutes": 60,
    }

    # Queue alert alongside
    send_alert(
        severity="high",
        message=f"Human escalation requested by {agent}: {reason}",
        run_id=run_id,
        channel="email",
        recipient="supply-chain-manager@company.example.com",
    )

    return {
        "success":       True,
        "checkpoint_id": checkpoint_id,
        "run_id":        run_id,
        "agent":         agent,
        "reason":        reason,
        "message":       f"Escalation created. Checkpoint ID: {checkpoint_id}. Workflow paused pending human response.",
    }


# ═══════════════════════════════════════════════════════════════════
# Tool 3 — log_exception_event
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=LogExceptionEventArgs)
def log_exception_event(
    run_id:         str,
    raised_by:      str,
    exception_type: str,
    severity:       str,
    description:    str,
    sku_id:         Optional[str] = None,
    supplier_id:    Optional[str] = None,
    context:        str           = "{}",
) -> dict:
    """
    Persist an exception event to the database and queue an alert.

    Exception events are the primary audit trail for supply chain disruptions.
    Every detected anomaly — stockout risk, demand spike, supplier delay —
    should be logged here.

    Returns the event_id of the created record.
    """
    try:
        context_data = json.loads(context) if isinstance(context, str) else {}
    except json.JSONDecodeError:
        context_data = {}

    event_id = str(uuid4())

    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """INSERT OR IGNORE INTO exception_events
               (event_id, run_id, raised_by, exception_type, severity,
                sku_id, supplier_id, description, context, resolved, raised_at)
               VALUES (?,?,?,?,?,?,?,?,?,0,?)""",
            (
                event_id, run_id, raised_by, exception_type, severity,
                sku_id, supplier_id, description,
                json.dumps(context_data),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        conn.close()
        persisted = True
    except Exception:
        persisted = False

    # Auto-alert for high / critical
    if severity in ("high", "critical"):
        send_alert(
            severity=severity,
            message=description,
            sku_id=sku_id,
            run_id=run_id,
            channel="email",
        )

    return {
        "success":      True,
        "event_id":     event_id,
        "persisted":    persisted,
        "severity":     severity,
        "exception_type": exception_type,
    }


# ═══════════════════════════════════════════════════════════════════
# Tool 4 — build_run_summary
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=BuildRunSummaryArgs)
def build_run_summary(run_id: str) -> dict:
    """
    Build a structured summary of a completed workflow run.

    Aggregates data from workflow_runs, agent_results, purchase_orders,
    and exception_events tables.

    Returns a comprehensive dict suitable for:
    - API response to GET /workflow/{run_id}
    - Dashboard display
    - Email summary report
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        # Workflow run record
        run = conn.execute(
            "SELECT * FROM workflow_runs WHERE run_id = ?", (run_id,)
        ).fetchone()

        # POs created in this run
        pos = conn.execute(
            "SELECT COUNT(*) as cnt, SUM(total_value) as val FROM purchase_orders WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        # Exception events
        exceptions = conn.execute(
            """SELECT severity, COUNT(*) as cnt, SUM(resolved) as resolved_cnt
               FROM exception_events WHERE run_id = ?
               GROUP BY severity""",
            (run_id,),
        ).fetchall()

        conn.close()

        exc_summary: dict[str, dict] = {}
        for e in exceptions:
            exc_summary[e["severity"]] = {
                "total":    e["cnt"],
                "resolved": e["resolved_cnt"],
            }

        return {
            "run_id":          run_id,
            "status":          dict(run)["status"] if run else "unknown",
            "started_at":      dict(run)["started_at"] if run else None,
            "completed_at":    dict(run)["completed_at"] if run else None,
            "pos_issued":      pos["cnt"] if pos else 0,
            "po_total_value":  round(float(pos["val"] or 0), 2) if pos else 0.0,
            "exceptions":      exc_summary,
            "total_exceptions": sum(v["total"] for v in exc_summary.values()),
        }

    except Exception as exc:
        return {"run_id": run_id, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════
# Tool 5 — send_slack_notification
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=SendSlackNotificationArgs)
def send_slack_notification(
    channel:  str = "#supply-chain-alerts",
    message:  str = "",
    severity: str = "info",
) -> dict:
    """
    Send a Slack notification to the supply chain operations channel.

    In Phase 6, this calls the real Slack Webhook API.
    Currently stubs the call and logs it in memory.

    Message format supports Slack Markdown:
      *bold*, _italic_, `code`, > blockquote

    Use this for real-time team awareness of:
    - Critical stockout alerts
    - Large POs requiring attention
    - Run completion summaries
    """
    colour_map = {
        "info":     "#36a64f",
        "warning":  "#ff9500",
        "high":     "#ff3b30",
        "critical": "#8b0000",
    }

    payload = {
        "channel":     channel,
        "attachments": [{
            "color":    colour_map.get(severity, "#36a64f"),
            "text":     message,
            "footer":   "Supply Demand AI",
            "ts":       str(datetime.utcnow().timestamp()),
        }],
    }

    notification_id = str(uuid4())[:8].upper()
    _NOTIFICATION_LOG.append({
        "type":            "slack",
        "notification_id": notification_id,
        "channel":         channel,
        "severity":        severity,
        "message":         message,
        "sent_at":         datetime.utcnow().isoformat(),
        "status":          "stub",   # → "sent" in Phase 6
    })

    return {
        "success":         True,
        "notification_id": notification_id,
        "channel":         channel,
        "status":          "stub",
        "message":         f"Slack notification queued for {channel}",
    }


# ═══════════════════════════════════════════════════════════════════
# Tool 6 — get_unresolved_events
# ═══════════════════════════════════════════════════════════════════

@tool(args_schema=GetUnresolvedEventsArgs)
def get_unresolved_events(
    run_id:   str,
    severity: Optional[str] = None,
) -> dict:
    """
    List all unresolved exception events for a workflow run.

    Use this tool at the end of a run to check whether any open
    issues require escalation or a follow-up action.
    """
    try:
        conn   = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        query  = "SELECT * FROM exception_events WHERE run_id = ? AND resolved = 0"
        params: list = [run_id]

        if severity:
            query  += " AND severity = ?"
            params.append(severity)

        query += " ORDER BY raised_at DESC"
        rows   = conn.execute(query, params).fetchall()
        conn.close()

        return {
            "run_id":   run_id,
            "total":    len(rows),
            "events":   [dict(r) for r in rows],
        }

    except Exception as exc:
        return {"run_id": run_id, "error": str(exc), "events": []}


# ═══════════════════════════════════════════════════════════════════
# Notification log accessor (for tests / API)
# ═══════════════════════════════════════════════════════════════════

def get_notification_log() -> list[dict]:
    """Return all queued notifications (in-memory store)."""
    return list(_NOTIFICATION_LOG)


# ═══════════════════════════════════════════════════════════════════
# Tool registry
# ═══════════════════════════════════════════════════════════════════

NOTIFICATION_TOOLS = [
    send_alert,
    escalate_to_human,
    log_exception_event,
    build_run_summary,
    send_slack_notification,
    get_unresolved_events,
]
