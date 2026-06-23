"""
agents/demand_forecast.py  (Phase 3 · Part 2 — ReAct upgrade)
───────────────────────────────────────────────────────────────
Demand Forecast Agent — LangGraph node with full ReAct tool-calling loop.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │  demand_forecast_node (LangGraph node)                      │
  │                                                             │
  │  1. Build system prompt with SKU context                    │
  │  2. LLM.bind_tools(FORECAST_TOOLS) → reason + call tools   │
  │     ┌─────────────────────────────────────────────────┐     │
  │     │  ReAct mini-loop (max MAX_ITERATIONS steps)     │     │
  │     │  LLM → tool_calls? → ToolNode → ToolMessages   │     │
  │     │  → LLM → tool_calls? → ... → AIMessage (done) │     │
  │     └─────────────────────────────────────────────────┘     │
  │  3. Parse structured ForecastResult from final AIMessage    │
  │  4. Write result + status to WorkflowState                  │
  └─────────────────────────────────────────────────────────────┘

Node name in graph: "demand_forecast"
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from graph.state import WorkflowState, add_exception
from tools.forecast_tools import (
    FORECAST_TOOLS,
    compute_mape,
    detect_demand_anomalies,
    get_demand_trend,
    load_demand_history,
    run_prophet_forecast,
)

settings      = get_settings()
MAX_ITERATIONS = settings.max_agent_iterations


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory (lazy singleton per agent)
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm_with_tools() -> Any:
    """Return a ChatOpenAI instance with FORECAST_TOOLS bound."""
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.0,
        api_key=settings.openai_api_key,
    )
    return llm.bind_tools(FORECAST_TOOLS)


# ─────────────────────────────────────────────────────────────────────────────
# ReAct loop executor
# ─────────────────────────────────────────────────────────────────────────────

def _run_react_loop(
    llm_with_tools: Any,
    messages:       list,
    tool_map:       dict[str, Any],
    max_iters:      int = MAX_ITERATIONS,
) -> tuple[list, str]:
    """
    Execute the ReAct (Reason + Act) loop.

    At each step:
      1. Call the LLM with the current message history
      2. If the response contains tool_calls → execute each tool,
         append ToolMessages, loop back
      3. If the response is a plain AIMessage → exit, return final content

    Args:
        llm_with_tools: LLM with tools bound via bind_tools()
        messages:       starting message list (System + Human)
        tool_map:       {tool_name: callable} for local execution
        max_iters:      safety ceiling on loop iterations

    Returns:
        (updated_messages, final_text_content)
    """
    for _ in range(max_iters):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # No tool calls → LLM is done reasoning
        if not getattr(response, "tool_calls", None):
            return messages, response.content

        # Execute each requested tool call
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id   = tc.get("id", tool_name)

            fn = tool_map.get(tool_name)
            if fn is None:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    result = fn(**tool_args)
                except Exception as exc:
                    result = {"error": str(exc)}

            messages.append(
                ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tool_id,
                )
            )

    # Max iterations reached — return whatever content we have
    last = messages[-1]
    return messages, getattr(last, "content", "")


# ─────────────────────────────────────────────────────────────────────────────
# Tool map for local execution (no LangGraph ToolNode needed here)
# ─────────────────────────────────────────────────────────────────────────────

FORECAST_TOOL_MAP: dict[str, Any] = {
    "load_demand_history":     load_demand_history,
    "run_prophet_forecast":    run_prophet_forecast,
    "compute_mape":            compute_mape,
    "detect_demand_anomalies": detect_demand_anomalies,
    "get_demand_trend":        get_demand_trend,
}


# ─────────────────────────────────────────────────────────────────────────────
# System prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(sku_ids: list[str], horizon: int) -> str:
    return f"""You are the Demand Forecast Agent in a Supply Demand Execution AI system.

Your job is to produce accurate demand forecasts for the following SKUs: {sku_ids}

You have access to these tools:
- load_demand_history:     Load historical demand data for a SKU
- run_prophet_forecast:    Run a Prophet time-series forecast
- compute_mape:            Measure forecast accuracy
- detect_demand_anomalies: Detect demand spikes or drops
- get_demand_trend:        Get overall demand direction

Instructions:
1. For each SKU, call load_demand_history first to assess data quality
2. Call get_demand_trend to understand the demand direction
3. Call detect_demand_anomalies to check for data issues
4. Call run_prophet_forecast with horizon_days={horizon}
5. Optionally call compute_mape if you want an accuracy estimate
6. After analysing all SKUs, return a JSON summary with this exact structure:

{{
  "forecasts": [
    {{
      "sku_id": "SKU-XXX",
      "avg_daily_demand": <float>,
      "total_forecasted_demand": <float>,
      "peak_demand_date": "YYYY-MM-DD",
      "peak_demand_units": <float>,
      "trend": "rising|falling|stable|volatile",
      "confidence": "high|medium|low",
      "method_used": "prophet|moving_average",
      "warnings": []
    }}
  ],
  "overall_commentary": "<2-3 sentence summary for the operations team>"
}}

Be concise with tool calls — one call per tool per SKU. Return only the JSON at the end."""


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph Node
# ─────────────────────────────────────────────────────────────────────────────

def demand_forecast_node(state: WorkflowState) -> dict[str, Any]:
    """
    LangGraph node: demand_forecast
    ─────────────────────────────────
    ReAct-upgraded demand forecast agent.

    The LLM autonomously decides which tools to call and in what order,
    reasons about data quality, and produces a structured ForecastResult.
    """
    sku_ids  = state.get("sku_ids", [])
    run_id   = state.get("run_id", "UNKNOWN")
    horizon  = settings.forecast_horizon_days
    exception_patch: dict[str, Any] = {}

    try:
        llm_with_tools = _get_llm_with_tools()
        messages = [
            SystemMessage(content=_build_system_prompt(sku_ids, horizon)),
            HumanMessage(content=(
                f"Analyse demand for SKUs: {sku_ids}. "
                f"Forecast horizon: {horizon} days. "
                f"Today's date: {datetime.utcnow().date()}. "
                f"Run ID: {run_id}"
            )),
        ]

        messages, final_content = _run_react_loop(
            llm_with_tools, messages, FORECAST_TOOL_MAP
        )

        # Parse the JSON result from the LLM's final message
        raw = final_content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result_data = json.loads(raw.strip())

    except Exception as exc:
        # Graceful fallback — run tools directly without LLM
        result_data = _fallback_forecast(sku_ids, horizon)
        result_data["_fallback_reason"] = str(exc)

    # ── Normalise into standard shape ─────────────────────────────────────────
    forecasts    = result_data.get("forecasts", [])
    commentary   = result_data.get("overall_commentary", "")
    all_forecasts: list[dict] = []

    for fc in forecasts:
        sku_id = fc.get("sku_id", "")

        # Flag demand spikes as exception events
        avg_d = fc.get("avg_daily_demand", 0)
        peak  = fc.get("peak_demand_units", 0)
        if avg_d > 0 and peak > avg_d * 2.0:
            exc_event = {
                "run_id":         run_id,
                "raised_by":      "demand_forecast",
                "exception_type": "demand_spike",
                "severity":       "warning",
                "sku_id":         sku_id,
                "supplier_id":    None,
                "description": (
                    f"{sku_id}: peak demand {peak:.0f} units on "
                    f"{fc.get('peak_demand_date')} is "
                    f"{peak/avg_d:.1f}× above average ({avg_d:.0f}/day)."
                ),
                "context": {
                    "peak_date":  fc.get("peak_demand_date"),
                    "peak_units": peak,
                    "avg_daily":  avg_d,
                },
                "resolved": False,
            }
            exception_patch = add_exception(state, exc_event)

        all_forecasts.append({
            **fc,
            "generated_at":   datetime.utcnow().isoformat(),
            "horizon_days":   horizon,
            "llm_commentary": commentary if len(forecasts) == 1 else None,
        })

    # Primary result = use last SKU's forecast (Phase 3 Part 3 upgrades to per-SKU map)
    primary = all_forecasts[-1] if all_forecasts else {}
    primary["llm_commentary"] = commentary

    return {
        **exception_patch,
        "demand_forecast_result": primary,
        "demand_forecast_status": "success",
        "supply_plan_input": {
            "sku_id":       primary.get("sku_id", sku_ids[0] if sku_ids else ""),
            "avg_daily":    primary.get("avg_daily_demand", 10.0),
            "horizon_days": horizon,
            "forecasts":    all_forecasts,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fallback — runs tools directly when LLM is unavailable
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_forecast(sku_ids: list[str], horizon: int) -> dict:
    """Direct tool calls when LLM is unavailable (no API key / offline)."""
    forecasts = []
    for sku_id in sku_ids:
        trend_data    = get_demand_trend(sku_id=sku_id, days=60)
        forecast_data = run_prophet_forecast(sku_id=sku_id, horizon_days=horizon)

        forecasts.append({
            "sku_id":                  sku_id,
            "avg_daily_demand":        forecast_data.get("avg_daily_demand", 10.0),
            "total_forecasted_demand": forecast_data.get("total_forecasted_demand", 0.0),
            "peak_demand_date":        forecast_data.get("peak_demand_date"),
            "peak_demand_units":       forecast_data.get("peak_demand_units", 0.0),
            "trend":                   trend_data.get("trend", "stable"),
            "confidence":              forecast_data.get("confidence", "medium"),
            "method_used":             forecast_data.get("method_used", "moving_average"),
            "warnings":                forecast_data.get("warnings", []),
        })

    return {
        "forecasts": forecasts,
        "overall_commentary": (
            f"Fallback forecast (LLM unavailable) for {len(sku_ids)} SKUs. "
            "Results based on moving-average model."
        ),
    }
