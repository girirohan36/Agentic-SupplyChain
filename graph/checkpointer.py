"""
graph/checkpointer.py
─────────────────────
LangGraph checkpointer configuration and HITL (Human-in-the-Loop) helpers.

LangGraph uses a "checkpointer" to persist the full graph state between
node executions. This enables:
  1. State persistence across async turns
  2. Human-in-the-loop interrupts (graph pauses, waits for human input)
  3. Resume from any checkpoint (fault tolerance)
  4. Full audit trail of every state transition

Two checkpointers are configured:
  - MemorySaver   → used in tests and local dev (in-process, no DB)
  - SqliteSaver   → used in production (persists to supply_demand.db)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import get_settings

settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Checkpointer factories
# ─────────────────────────────────────────────────────────────────────────────

def get_memory_checkpointer() -> MemorySaver:
    """
    In-memory checkpointer — use in unit tests and local dev.
    State is lost when the process exits.

    Usage:
        checkpointer = get_memory_checkpointer()
        graph = build_graph(checkpointer=checkpointer)
    """
    return MemorySaver()


def get_sqlite_checkpointer() -> SqliteSaver:
    """
    SQLite-backed checkpointer — use in staging and production.
    Persists graph state to the same DB as the application data.

    The checkpointer creates its own tables (checkpoints, checkpoint_blobs,
    checkpoint_writes) in the database automatically on first use.

    Usage:
        with get_sqlite_checkpointer() as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            result = graph.invoke(state, config=thread_config("RUN-001"))
    """
    db_path = settings.database_url.replace("sqlite:///", "")
    return SqliteSaver.from_conn_string(db_path)


def get_checkpointer():
    """
    Return the appropriate checkpointer based on APP_ENV.
      - development / test → MemorySaver (fast, no setup)
      - staging / production → SqliteSaver (persistent)
    """
    if settings.is_development or os.getenv("TESTING"):
        return get_memory_checkpointer()
    return get_sqlite_checkpointer()


# ─────────────────────────────────────────────────────────────────────────────
# Thread config factory
# ─────────────────────────────────────────────────────────────────────────────

def thread_config(run_id: str, **extra: Any) -> dict[str, Any]:
    """
    Build the LangGraph `config` dict for a workflow run.

    Every graph.invoke() / graph.astream() call must pass this config
    so the checkpointer can associate state with the correct thread.

    Args:
        run_id: unique workflow run identifier (e.g. "RUN-A1B2C3D4")
        **extra: optional extra keys merged into configurable

    Returns:
        {"configurable": {"thread_id": run_id, ...}}

    Usage:
        config = thread_config("RUN-A1B2C3D4")
        result = await graph.ainvoke(initial_state, config=config)
    """
    return {
        "configurable": {
            "thread_id": run_id,
            **extra,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# HITL helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_interrupt_payload(
    run_id:   str,
    agent:    str,
    prompt:   str,
    context:  dict[str, Any],
    timeout_minutes: int = 60,
) -> dict[str, Any]:
    """
    Build a structured interrupt payload that the graph raises via
    NodeInterrupt. The API surfaces this to the user; the user's
    response is passed back via graph.invoke(None, config=...).

    This payload is stored in state["hitl_checkpoint"].

    Args:
        run_id:           current workflow run ID
        agent:            name of the agent requesting human input
        prompt:           human-readable question / action description
        context:          structured data the human needs to decide
        timeout_minutes:  auto-escalate if no response within this window

    Returns:
        dict matching HITLCheckpoint schema (without checkpoint_id — set by caller)
    """
    from datetime import datetime
    from uuid import uuid4

    return {
        "checkpoint_id":   str(uuid4()),
        "run_id":          run_id,
        "agent":           agent,
        "prompt":          prompt,
        "context":         context,
        "required_action": "approve",
        "response":        None,
        "action_taken":    None,
        "created_at":      datetime.utcnow().isoformat(),
        "responded_at":    None,
        "timeout_minutes": timeout_minutes,
    }


def resume_with_approval(approved: bool, response: str = "") -> dict[str, Any]:
    """
    Build the state patch to resume a HITL-paused graph.

    Pass the return value as the first argument to graph.invoke():
        graph.invoke(resume_with_approval(True), config=thread_config(run_id))

    Args:
        approved: True = human approved, False = human rejected
        response: optional free-text response from the human

    Returns:
        Partial WorkflowState dict — merged into current state on resume
    """
    from datetime import datetime

    return {
        "hitl_required":  False,
        "hitl_approved":  approved,
        "hitl_response":  {
            "approved":     approved,
            "response":     response,
            "responded_at": datetime.utcnow().isoformat(),
        },
    }
