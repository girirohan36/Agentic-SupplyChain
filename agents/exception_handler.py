"""
agents/exception_handler.py  (Phase 5 — enhanced)
────────────────────────────────────────────────────
Exception Handler Agent — LangGraph node upgraded with:
  1. Cascading exception detection — one event can spawn related events
  2. Root cause analysis — LLM identifies primary vs contributing causes
  3. Auto-escalation rule — >3 criticals in one run → forced HITL
  4. Notification wiring — send_alert + send_slack per severity tier
  5. Resolution playbooks — structured per exception_type
  6. Deduplication — avoids duplicate alerts for the same SKU+type combo

Node name in graph: "exception_handler"
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from config.settings import get_settings
from graph.state import WorkflowState
from tools.notification_tools import (
    send_alert,
    send_slack_notification,
    log_exception_event,
)

settings = get_settings()

# ── Severity ordering ─────────────────────────────────────────────────────────
_SEVERITY_ORDER = {"critical": 0, "high": 1, "warning": 2, "info": 3}

# ── Auto-escalation threshold ─────────────────────────────────────────────────
AUTO_ESCALATE_CRITICAL_COUNT = 3   # force HITL if ≥ this many criticals in one run

# ── Resolution playbooks ──────────────────────────────────────────────────────
RESOLUTION_PLAYBOOKS: dict[str, str] = {
    "stockout_imminent":   "Expedite PO with primary supplier. Activate secondary supplier as backup. Consider emergency spot-buy if lead time > 5 days.",
    "demand_spike":        "Increase safety stock multiplier by 1.5×. Alert procurement to pre-order. Check if spike is seasonal or one-off.",
    "supply_disruption":   "Switch to secondary supplier immediately. Update lead time estimates. Notify affected downstream orders.",
    "overstock":           "Pause all replenishment for this SKU. Run promotions or redistribute to high-demand DC. Review demand forecast accuracy.",
    "supplier_delay":      "Request revised ETA from supplier. Evaluate alternative suppliers. Notify fulfillment of potential shortfall.",
    "data_quality":        "Flag SKU for manual data review. Do not auto-generate POs until resolved. Check order history for anomalies.",
    "agent_failure":       "Retry failed agent with exponential backoff. Check API key and connectivity. Escalate to engineering if retry fails.",
    "demand_drop":         "Review demand signals — may indicate churn or seasonal decline. Reduce reorder point. Consider discontinuation review.",
}

# ── Cascading rules ────────────────────────────────────────────────────────────
# Maps an exception_type → list of related events that should be generated
CASCADE_RULES: dict[str, list[dict]] = {
    "stockout_imminent": [
        {
            "exception_type": "supply_disruption",
            "severity":       "high",
            "description_suffix": "Downstream fulfillment risk due to stockout.",
        }
    ],
    "supplier_delay": [
        {
            "exception_type": "stockout_imminent",
            "severity":       "warning",
            "description_suffix": "Stock may run out before delayed shipment arrives.",
        }
    ],
    "demand_spike": [
        {
            "exception_type": "stockout_imminent",
            "severity":       "warning",
            "description_suffix": "Spike in demand may exhaust current stock sooner than forecast.",
        }
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Resolution recommendation (LLM + playbook fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_recommendation(event: dict) -> str:
    """LLM-generated resolution with playbook fallback."""
    exc_type = event.get("exception_type", "")
    sku_id   = event.get("sku_id", "N/A")
    desc     = event.get("description", "")

    try:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.1,
            api_key=settings.openai_api_key,
        )
        playbook = RESOLUTION_PLAYBOOKS.get(exc_type, "Review and resolve manually.")
        prompt = (
            f"Supply chain exception — provide a 2-sentence actionable resolution.\n"
            f"Exception type: {exc_type}\n"
            f"SKU: {sku_id}\n"
            f"Description: {desc}\n"
            f"Standard playbook: {playbook}\n"
            f"Context: {event.get('context', {})}\n"
            "Be specific, concise, and ops-team-friendly."
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content.strip()

    except Exception:
        return RESOLUTION_PLAYBOOKS.get(exc_type,
               f"Review and resolve exception for {sku_id}. Manual intervention may be required.")


# ─────────────────────────────────────────────────────────────────────────────
# Root cause analysis
# ─────────────────────────────────────────────────────────────────────────────

def _root_cause_analysis(events: list[dict]) -> str:
    """
    Identify the primary root cause across all exceptions in a run.
    Uses LLM if available, otherwise a rule-based heuristic.
    """
    if not events:
        return "No exceptions detected."

    # Count by type
    type_counts: dict[str, int] = {}
    for e in events:
        t = e.get("exception_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    primary_type = max(type_counts, key=lambda k: type_counts[k])

    try:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.1,
            api_key=settings.openai_api_key,
        )
        summary = "\n".join([
            f"- {e.get('exception_type')} [{e.get('severity')}]: {e.get('description', '')[:80]}"
            for e in events[:6]
        ])
        prompt = (
            f"In 2 sentences, identify the primary root cause of these supply chain exceptions "
            f"and recommend the single most important corrective action.\n\n{summary}"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content.strip()

    except Exception:
        return (
            f"Primary issue: {primary_type} ({type_counts[primary_type]} occurrences). "
            f"Total exceptions: {len(events)}. "
            f"Recommend addressing {primary_type} first to prevent cascading failures."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cascade generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_cascading_events(
    event:   dict,
    run_id:  str,
    existing_types: set,
) -> list[dict]:
    """
    Generate secondary exception events based on cascade rules.
    De-duplicates: won't generate an event of a type already present for the same SKU.
    """
    exc_type  = event.get("exception_type", "")
    sku_id    = event.get("sku_id")
    cascades  = CASCADE_RULES.get(exc_type, [])
    new_events: list[dict] = []

    for cascade in cascades:
        dedup_key = f"{sku_id}:{cascade['exception_type']}"
        if dedup_key in existing_types:
            continue  # already have this type for this SKU

        new_events.append({
            "run_id":         run_id,
            "raised_by":      "exception_handler",
            "exception_type": cascade["exception_type"],
            "severity":       cascade["severity"],
            "sku_id":         sku_id,
            "supplier_id":    event.get("supplier_id"),
            "description": (
                f"{sku_id}: [Cascaded from {exc_type}] "
                f"{cascade['description_suffix']}"
            ),
            "context": {"cascaded_from": exc_type, "parent_event_sku": sku_id},
            "resolved": False,
        })
        existing_types.add(dedup_key)

    return new_events


# ─────────────────────────────────────────────────────────────────────────────
# Notification dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch_notification(event: dict, resolution: str, run_id: str) -> dict:
    """Send appropriate notification based on severity."""
    severity   = event.get("severity", "info")
    sku_id     = event.get("sku_id", "System")
    exc_type   = event.get("exception_type", "unknown")
    description = event.get("description", "")

    # Email alert for all high+ severity
    if severity in ("critical", "high"):
        send_alert(
            severity=severity,
            message=f"[{exc_type.upper()}] {description[:200]}\n\nResolution: {resolution[:200]}",
            sku_id=sku_id,
            run_id=run_id,
            channel="email",
        )

    # Slack for critical only
    if severity == "critical":
        send_slack_notification(
            channel="#supply-chain-alerts",
            message=(
                f"🚨 *CRITICAL EXCEPTION* — {exc_type}\n"
                f"*SKU:* {sku_id}\n"
                f"*Issue:* {description[:150]}\n"
                f"*Action:* {resolution[:150]}"
            ),
            severity="critical",
        )

    return {
        "channel":  "email" if severity in ("critical","high") else "log",
        "severity": severity,
        "sent":     True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph Node
# ─────────────────────────────────────────────────────────────────────────────

def exception_handler_node(state: WorkflowState) -> dict[str, Any]:
    """
    LangGraph node: exception_handler (Phase 5 enhanced)
    ──────────────────────────────────────────────────────
    Triages all exception events with:
      - Cascading event generation
      - Root cause analysis
      - Per-severity notification routing (email + Slack)
      - Auto-escalation when critical count ≥ threshold
      - Structured resolution per playbook + LLM refinement
    """
    raw_events = list(state.get("exception_events", []))
    run_id     = state.get("run_id", "UNKNOWN")

    if not raw_events:
        return {
            "exception_events":  [],
            "exception_result": {
                "run_id":           run_id,
                "total_events":     0,
                "resolved_count":   0,
                "escalated_count":  0,
                "resolved":         [],
                "escalated":        [],
                "notifications":    [],
                "root_cause":       "No exceptions detected.",
                "processed_at":     datetime.utcnow().isoformat(),
            },
            "exception_status":   "success",
            "exception_detected": False,
        }

    # ── Step 1: Build dedup set and generate cascading events ─────────────────
    existing_types: set[str] = {
        f"{e.get('sku_id')}:{e.get('exception_type')}" for e in raw_events
    }

    cascaded: list[dict] = []
    for event in list(raw_events):   # iterate over original set
        cascaded.extend(_generate_cascading_events(event, run_id, existing_types))

    all_events = raw_events + cascaded

    # ── Step 2: Sort by severity ───────────────────────────────────────────────
    all_events.sort(key=lambda e: _SEVERITY_ORDER.get(e.get("severity", "info"), 3))

    # ── Step 3: Root cause analysis ───────────────────────────────────────────
    root_cause = _root_cause_analysis(all_events)

    # ── Step 4: Triage each event ──────────────────────────────────────────────
    resolved_events:  list[dict] = []
    escalated_events: list[dict] = []
    notifications:    list[dict] = []

    critical_count = sum(1 for e in all_events if e.get("severity") == "critical")

    for event in all_events:
        severity       = event.get("severity", "info")
        recommendation = _resolve_recommendation(event)
        notification   = _dispatch_notification(event, recommendation, run_id)
        notifications.append(notification)

        resolved_event = {
            **event,
            "resolved":    True,
            "resolution":  recommendation,
            "resolved_at": datetime.utcnow().isoformat(),
        }

        # Escalate criticals OR if auto-escalation threshold is met
        if severity == "critical" or critical_count >= AUTO_ESCALATE_CRITICAL_COUNT:
            escalated_events.append(resolved_event)
        else:
            resolved_events.append(resolved_event)

    # ── Step 5: Auto-escalation guard ─────────────────────────────────────────
    auto_escalated = critical_count >= AUTO_ESCALATE_CRITICAL_COUNT
    hitl_required  = state.get("hitl_required", False) or auto_escalated

    if auto_escalated:
        send_alert(
            severity="critical",
            message=(
                f"AUTO-ESCALATION: {critical_count} critical exceptions in run {run_id}. "
                f"Human review required. Root cause: {root_cause[:200]}"
            ),
            run_id=run_id,
            channel="email",
        )
        send_slack_notification(
            channel="#supply-chain-escalations",
            message=(
                f"🚨 *AUTO-ESCALATION* — {critical_count} critical exceptions\n"
                f"*Run:* {run_id}\n"
                f"*Root cause:* {root_cause[:200]}"
            ),
            severity="critical",
        )

    all_resolved = resolved_events + escalated_events

    exception_result = {
        "run_id":           run_id,
        "total_events":     len(all_events),
        "original_events":  len(raw_events),
        "cascaded_events":  len(cascaded),
        "resolved_count":   len(resolved_events),
        "escalated_count":  len(escalated_events),
        "critical_count":   critical_count,
        "auto_escalated":   auto_escalated,
        "resolved":         resolved_events,
        "escalated":        escalated_events,
        "notifications":    notifications,
        "root_cause":       root_cause,
        "processed_at":     datetime.utcnow().isoformat(),
    }

    return {
        "exception_events":  all_resolved,
        "exception_result":  exception_result,
        "exception_status":  "success",
        "exception_detected": False,     # reset — handled
        "hitl_required":     hitl_required,
    }
