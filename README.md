<div align="center">

<img src="https://img.shields.io/badge/🏭Agentic%20AI-007aff?style=for-the-badge" alt="Supply Demand AI"/>

# Supply Demand Fulfillment  — Agentic AI

**A production-grade multi-agent AI system that autonomously executes the full supply-demand cycle — from demand forecasting to procurement, inventory management, fulfillment, and exception handling.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/GPT--4o-Powered-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-54%2B%20passing-34c759?style=flat-square&logo=pytest&logoColor=white)](#-testing)
[![License](https://img.shields.io/badge/License-MIT-34c759?style=flat-square)](LICENSE)

<br/>

[**Architecture**](#-agentic-architecture) · [**How It Works**](#-how-the-agents-work) · [**Quick Start**](#-quick-start) · [**API**](#-api-reference) · [**Dashboard**](#-live-dashboard) · [**Design Decisions**](#-key-design-decisions)

</div>

---

## 🎯 What This Project Does

Traditional supply chain systems are **reactive, siloed, and manually intensive**. This system replaces that with a coordinated team of AI agents that reason, act, and adapt in real time:

```
Human asks: "What do we need to order this week?"

The system:
  1. Forecasts demand for all 20 SKUs using Prophet + LLM reasoning
  2. Computes Economic Order Quantities and reorder points
  3. Scores inventory health across the entire warehouse in parallel
  4. Generates Purchase Orders — comparing suppliers, splitting critical orders
  5. Routes open orders to available stock, creates backorders with ETAs
  6. Detects, cascades, and resolves exceptions autonomously
  7. Pauses for human approval on high-value decisions (HITL)
  8. Returns a structured run summary with every decision explained

All without a single human touching a spreadsheet.
```

---

## 🏗 Agentic Architecture

### The Big Picture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   USER / API REQUEST                                                    │
│   POST /workflow/run {"sku_ids": ["SKU-001", ..., "SKU-020"]}           │
│                            │                                            │
│                            ▼                                            │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                  LangGraph StateGraph                            │  │
│   │                                                                  │  │
│   │  WorkflowState (shared TypedDict — flows through every node)    │  │
│   │  {                                                               │  │
│   │    run_id, sku_ids, messages,                                    │  │
│   │    demand_forecast_result,  supply_plan_result,                  │  │
│   │    inventory_result,         procurement_result,                 │  │
│   │    fulfillment_result,       exception_events,                   │  │
│   │    hitl_required,            hitl_checkpoint,                    │  │
│   │    per_sku_results[],        exception_detected,                 │  │
│   │    replenishment_needed,     run_complete, summary               │  │
│   │  }                                                               │  │
│   └──────────────────────────┬──────────────────────────────────────┘  │
│                               │                                         │
└───────────────────────────────┼─────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │           orchestrator_plan node              │
        │  • LLM analyses SKUs and creates exec plan    │
        │  • Sets agent execution order + priorities    │
        └───────────────────────┬───────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │           demand_forecast node                │
        │  • Loads 180-day demand history per SKU       │
        │  • Runs Prophet time-series forecast          │
        │  • Detects demand spikes → ExceptionEvent     │
        │  • LLM generates commentary                   │
        └───────────────────────┬───────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │           supply_planning node                │
        │  • Reads ForecastResult from state            │
        │  • Computes EOQ per SKU: √(2DS/H)            │
        │  • Sets reorder points + safety stock         │
        │  • Flags SKUs needing replenishment           │
        └───────────────────────┬───────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │             sku_dispatcher node               │  ← FAN-OUT
        │  • Reads sku_ids from state                   │
        │  • Issues Send(sku_worker, per_sku_state) ×N  │
        │  • All 20 SKUs processed IN PARALLEL          │
        └──┬──────┬──────┬──────┬──────┬──────┬────────┘
           │      │      │      │      │      │
           ▼      ▼      ▼      ▼      ▼      ▼
         [W1]   [W2]   [W3]  [W4]  [W5]  ... [W20]   ← PARALLEL WORKERS
           │      │      │      │      │      │
           └──────┴──────┴──────┴──────┴──────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │            sku_aggregator node                │  ← FAN-IN
        │  • Waits for ALL 20 sku_workers to finish     │
        │  • Merges per_sku_results[] into snapshot     │
        │  • Sets reorder_triggers list                 │
        └───────────────────────┬───────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │               inventory node                  │
        │  • bulk_score_all_skus() — 1 DB call          │
        │  • Parallel LLM commentary (ThreadPoolExecutor│
        │  • Overstock + expiry risk detection          │
        │  • Raises ExceptionEvents for critical SKUs   │
        └───────────────────────┬───────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │ replenishment_needed? │  ← CONDITIONAL EDGE
                    │    True    │  False   │
                    ▼            ▼
        ┌────────────────┐  ┌─────────────────────────┐
        │ procurement    │  │  skip to fulfillment     │
        │ node           │  └────────────┬────────────┘
        │                │               │
        │ • compare_     │               │
        │   suppliers()  │               │
        │ • split orders │               │
        │   for critical │               │
        │ • HITL if>$5K  │               │
        └───────┬────────┘               │
                │                        │
        ┌───────┴────────┐               │
        │  hitl_required?│  ← CONDITIONAL│
        │  True  │ False │               │
        │    ↓   │   ↓   │               │
        │ pause  │ cont. │               │
        │  END   │       │               │
        └────────┘       │               │
                         └───────┬───────┘
                                 │
                                 ▼
        ┌───────────────────────────────────────────────┐
        │               fulfillment node                │
        │  • Priority-sorts orders (critical-first)     │
        │  • route_to_best_dc() for each SKU            │
        │  • create_dispatch() — writes to DB           │
        │  • manage_backorder() — ETA from open POs     │
        │  • Raises exception if fill_rate < 80%        │
        └───────────────────────┬───────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │  exception_detected?  │  ← CONDITIONAL EDGE
                    │    True    │  False   │
                    ▼            ▼
        ┌────────────────┐  ┌────────────────────────┐
        │ exception_     │  │ orchestrator_finalise  │
        │ handler node   │  │ • Builds run summary   │
        │                │  │ • Marks completed      │
        │ • Cascade rules│  │ • Returns to API       │
        │ • Root cause   │  └────────────┬───────────┘
        │ • Auto-escalate│               │
        │   ≥3 criticals │               │
        │ • Email+Slack  │               │
        └───────┬────────┘               │
                └───────────┬────────────┘
                            │
                            ▼
                           END
```

---

## 🔬 How the Agents Work

### The ReAct Pattern (Reason → Act → Observe)

Every agent uses the same core loop: the LLM decides **which tools to call** and **in what order** based on what it observes. No hardcoded logic — the agent reasons its way to a decision.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent ReAct Loop                            │
│                                                                 │
│  ┌─────────────────────────────────────────────┐               │
│  │  System Prompt                              │               │
│  │  "You are the Inventory Agent. You have     │               │
│  │   access to these tools: [list]. Your job   │               │
│  │   is to assess health for SKUs: [sku_ids]"  │               │
│  └──────────────────┬──────────────────────────┘               │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────┐                  │
│  │  LLM (GPT-4o with bind_tools())          │◄──────────────┐  │
│  │                                          │               │  │
│  │  "I should call get_inventory_kpis       │               │  │
│  │   first for the warehouse overview,      │               │  │
│  │   then bulk_score_all_skus..."           │               │  │
│  └──────────────────┬───────────────────────┘               │  │
│                     │                                        │  │
│         tool_calls in response?                              │  │
│                YES ─┤                                        │  │
│                     ▼                                        │  │
│  ┌──────────────────────────────────────────┐               │  │
│  │  Tool Execution (ToolNode)               │               │  │
│  │                                          │               │  │
│  │  get_inventory_kpis()                    │               │  │
│  │    → {fill_rate: 94%, total_value: $188K │               │  │
│  │       stockout_count: 2, ...}            │               │  │
│  │                                          │               │  │
│  │  bulk_score_all_skus(include_healthy=True│               │  │
│  │    → [{sku_id: SKU-011, status: critical │               │  │
│  │        dos: 0.0d, health_score: 0.002},  │               │  │
│  │       {sku_id: SKU-008, status: at_risk, │               │  │
│  │        dos: 11.0d, ...}, ...]            │               │  │
│  └──────────────────┬───────────────────────┘               │  │
│                     │                                        │  │
│             ToolMessage results ───────────────────────────►─┘  │
│             appended to messages[]                              │
│                                                                 │
│                     more tool_calls? ──► loop again            │
│                     NO tool_calls? ───► extract JSON result    │
│                                                                 │
│  ┌──────────────────────────────────────────┐                  │
│  │  Final AIMessage (structured JSON)       │                  │
│  │  {                                       │                  │
│  │    "records": [                          │                  │
│  │      {"sku_id": "SKU-011",               │                  │
│  │       "health_status": "critical",       │                  │
│  │       "days_of_supply": 0.0,             │                  │
│  │       "llm_commentary": "SKU-011 is      │                  │
│  │        at zero stock — expedite PO..."}  │                  │
│  │    ],                                    │                  │
│  │    "kpis": {...},                        │                  │
│  │    "inventory_commentary": "..."         │                  │
│  │  }                                       │                  │
│  └──────────────────────────────────────────┘                  │
│                     │                                           │
│                     ▼                                           │
│         Write result to WorkflowState                          │
│         state["inventory_result"] = result                     │
│         state["inventory_status"] = "success"                  │
└─────────────────────────────────────────────────────────────────┘
```

### 🔧 The Tool Ecosystem (36 Tools)

```
╔══════════════════════════════════════════════════════════════════╗
║                    @tool Function Registry                       ║
╠══════════════╦═══════════════════════════════════════════════════╣
║ FORECAST     ║ load_demand_history   · run_prophet_forecast      ║
║ (5 tools)    ║ compute_mape          · detect_demand_anomalies   ║
║              ║ get_demand_trend                                   ║
╠══════════════╬═══════════════════════════════════════════════════╣
║ INVENTORY    ║ get_stock_level       · compute_eoq               ║
║ (11 tools)   ║ calculate_reorder_pt  · score_inventory_health    ║
║              ║ list_critical_skus    · get_days_of_supply        ║
║              ║ update_stock_level    · record_stock_movement     ║
║              ║ bulk_score_all_skus   · get_inventory_kpis        ║
║              ║ check_expiry_risk                                  ║
╠══════════════╬═══════════════════════════════════════════════════╣
║ PROCUREMENT  ║ get_supplier_info     · find_best_supplier        ║
║ (9 tools)    ║ generate_purchase_order · submit_po · cancel_po   ║
║              ║ get_po_status         · list_open_pos             ║
║              ║ compare_suppliers     · calculate_split_order     ║
╠══════════════╬═══════════════════════════════════════════════════╣
║ FULFILLMENT  ║ get_open_orders       · score_order_priority      ║
║ (6 tools)    ║ route_to_best_dc      · create_dispatch           ║
║              ║ manage_backorder      · calculate_fill_rate       ║
╠══════════════╬═══════════════════════════════════════════════════╣
║ NOTIFICATION ║ send_alert            · escalate_to_human         ║
║ (6 tools)    ║ log_exception_event   · build_run_summary         ║
║              ║ send_slack_notification · get_unresolved_events   ║
╚══════════════╩═══════════════════════════════════════════════════╝
```

### 📦 The 7 Agents in Detail

#### 1️⃣ Orchestrator Agent
```
Role: Master controller — plans, routes, and finalises every run.

Two nodes in the graph:
  orchestrator_plan      → called at START
  orchestrator_finalise  → called before END

How it plans:
  Input:  run_id, sku_ids
  Action: LLM generates JSON execution plan
          {"execution_order": [...], "priority_skus": [...], "notes": "..."}
  Output: Sets all agent statuses to "pending", marks workflow "running"

How it finalises:
  Input:  all agent results from state
  Action: Aggregates counts (critical SKUs, POs issued, fill rate, exceptions)
  Output: Structured summary dict written to state["summary"]
```

#### 2️⃣ Demand Forecast Agent
```
Role: Time-series demand forecasting with LLM augmentation.

ReAct tool chain:
  1. load_demand_history(sku_id, days=180) — loads 180 rows per SKU
  2. get_demand_trend(sku_id)              — slope + pct_change analysis
  3. detect_demand_anomalies(sku_id)       — Z-score spike/drop detection
  4. run_prophet_forecast(sku_id, horizon=90) — Facebook Prophet model
  5. compute_mape(sku_id)                  — forecast accuracy on holdout

Output to state:
  demand_forecast_result: {
    sku_id, avg_daily_demand, total_forecasted_demand,
    peak_demand_date, trend, confidence, method_used,
    llm_commentary   ← LLM-generated plain English insight
  }

Side effects:
  • demand_spike detected → ExceptionEvent{severity: "warning"}
  • Writes supply_plan_input for the next agent
```

#### 3️⃣ Supply Planning Agent
```
Role: Compute optimal replenishment quantities using inventory theory.

EOQ Formula:
        EOQ = √( 2 × D × S / H )
  Where: D = annual demand (from forecast)
         S = ordering cost ($100/order default)
         H = holding cost = unit_cost × 25%

Reorder Point:
        ROP = avg_daily × lead_time_days + safety_stock
        safety_stock = avg_daily × 14 (days)

Urgency classification:
  critical  → < 3 days of supply remaining
  high      → available < safety_stock
  medium    → (on_hand + in_transit) ≤ reorder_point
  low       → healthy

Output to state:
  supply_plan_result: {sku_plans: [...], reorder_triggers: [...]}
  replenishment_needed: True/False    ← drives conditional edge
```

#### 4️⃣ Inventory Agent (with Parallel Fan-Out)
```
Role: Comprehensive warehouse health assessment.

Phase 4 upgrade — why it's fast:

  BEFORE (sequential):             AFTER (parallel):
  score_health(SKU-001)            bulk_score_all_skus()    ← 1 DB call
  score_health(SKU-002)            get_inventory_kpis()     ← 1 DB call
  score_health(SKU-003)            check_expiry_risk()      ← 1 DB call
  ...                              ThreadPoolExecutor(
  score_health(SKU-020)              LLM commentary ×N     ) ← parallel
  20 DB calls                      3 DB calls total

Health score formula (0.0 → 1.0):
  score = (0.50 × dos_score) + (0.30 × rop_score) + (0.20 × ss_score)
  Where:
    dos_score = min(1.0, days_of_supply / 30)
    rop_score = min(1.0, available / reorder_point)
    ss_score  = min(1.0, available / safety_stock)

  ≥ 0.75 → healthy   |  0.40–0.74 → at_risk
  < 0.40 → critical  |  0.00 → stock_out

Exception events raised:
  critical/stock_out  → {severity: "critical"}  → email + Slack
  overstock           → {severity: "info"}       → email
  expiry_risk         → {severity: "warning"}    → email
```

#### 5️⃣ Procurement Agent
```
Role: Autonomous Purchase Order generation with supplier intelligence.

Supplier selection logic:
  composite_score = reliability × 0.40
                  + (1 - norm_cost) × 0.35
                  + (1 - norm_lead) × 0.25

Split order logic (for CRITICAL urgency):
  calculate_split_order(sku_id, total_qty, primary_split=0.70)
  → {
      primary:   {supplier: SUP-001, qty: 84, cost: $X, eta: +14d}
      secondary: {supplier: SUP-003, qty: 36, cost: $Y, eta: +21d}
      combined_reliability: 1-(1-0.97)*(1-0.88) = 99.6%
    }
  → Two POs generated in parallel

HITL (Human-in-the-Loop) trigger:
  if po.total_value >= $5,000 AND ENABLE_HUMAN_IN_THE_LOOP:
    state["hitl_required"]  = True
    state["hitl_checkpoint"] = {prompt, context, po_numbers}
    graph routes to __interrupt__ → workflow PAUSES
    API surfaces checkpoint → human approves/rejects
    POST /workflow/{run_id}/resume → graph RESUMES
```

#### 6️⃣ Fulfillment Agent
```
Role: Priority-sorted order routing and dispatch.

Priority scoring (0–100):
  score = inventory_urgency + dos_urgency + channel_premium
  Where:
    inventory_urgency: stock_out=50, critical=40, at_risk=25, healthy=10
    dos_urgency:       <3d=30, <7d=24, <14d=18, <30d=9, ≥30d=0
    channel_premium:   online=10, retail=7, wholesale=4

Routing:
  route_to_best_dc(sku_id, qty_needed)
  → Selects DC with most available stock that can fully satisfy order
  → Falls back to partial fill if no single DC has enough

DB write-back:
  create_dispatch()   → records "sale" movement in stock_movements
                      → decrements inventory_levels.on_hand
  manage_backorder()  → records held order with ETA from open POs

Fill rate KPI:
  fill_rate = dispatched / demanded × 100
  < 80% → ExceptionEvent{type: "supply_disruption", severity: "high"}
```

#### 7️⃣ Exception Handler Agent
```
Role: Intelligent triage, cascading detection, and resolution.

Cascade rules (Phase 5):
  stockout_imminent → spawns supply_disruption{severity: "high"}
  supplier_delay    → spawns stockout_imminent{severity: "warning"}
  demand_spike      → spawns stockout_imminent{severity: "warning"}

Auto-escalation rule:
  if critical_count >= 3:
    auto_escalated = True
    hitl_required  = True
    send_alert("critical", ...)
    send_slack("#supply-chain-escalations", ...)

Resolution playbooks:
  stockout_imminent → "Expedite PO with primary supplier. Activate secondary..."
  demand_spike      → "Increase safety stock 1.5×. Alert procurement to pre-order..."
  overstock         → "Pause replenishment. Run promotions or redistribute..."
  supplier_delay    → "Switch to secondary supplier. Update lead time estimates..."

Root cause analysis:
  LLM analyses all events → identifies primary cause → 2-sentence recommendation
  Fallback: rule-based heuristic if LLM unavailable
```

---

## 🔗 The State Machine

```
WorkflowState TypedDict — flows through every node unchanged
  (each node returns a PARTIAL dict update; LangGraph merges it)

Key state fields:
┌─────────────────────────────────────────────────────────────────┐
│  CONTROL FLAGS (drive conditional edges)                        │
│  replenishment_needed: bool  → routes inventory → procurement  │
│  exception_detected:   bool  → routes fulfillment → handler    │
│  hitl_required:        bool  → routes procurement → __pause__  │
│  run_complete:         bool  → routes finalise → END           │
├─────────────────────────────────────────────────────────────────┤
│  PER-AGENT RESULTS (immutable once written)                     │
│  demand_forecast_result:  {sku_id, avg_daily, trend, ...}      │
│  supply_plan_result:      {sku_plans, reorder_triggers, ...}   │
│  inventory_result:        {records, kpis, reorder_triggers}    │
│  procurement_result:      {issued_pos, total_value, ...}       │
│  fulfillment_result:      {routed_orders, fill_rate_pct, ...}  │
├─────────────────────────────────────────────────────────────────┤
│  PHASE 4 PARALLEL ACCUMULATOR                                   │
│  per_sku_results:    list[dict]  ← appended by each sku_worker │
│  sku_processing_complete: bool                                  │
├─────────────────────────────────────────────────────────────────┤
│  HITL STATE                                                     │
│  hitl_checkpoint: {prompt, context, po_numbers, timeout_mins}  │
│  hitl_response:   {approved, response, responded_at}           │
│  hitl_approved:   Optional[bool]                               │
├─────────────────────────────────────────────────────────────────┤
│  MESSAGE THREAD (LangChain add_messages reducer)                │
│  messages: Annotated[list[BaseMessage], add_messages]          │
│            ← auto-appended, never overwritten                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔐 Human-in-the-Loop (HITL) Flow

```
Procurement Agent generates PO for $12,000
              │
              ▼
    $12,000 >= $5,000 threshold?
              │ YES
              ▼
    state["hitl_required"]  = True
    state["hitl_checkpoint"] = {
      prompt:  "PO-A1B2C3D4 for TechSource ($12,000) needs approval",
      context: {po_number, supplier, value, expected_date, lines},
      timeout_minutes: 60
    }
              │
              ▼
    Graph routes to __interrupt__ ──► WORKFLOW PAUSES
              │
              ▼
    GET /workflow/{run_id} returns:
      {"status": "paused", "hitl_checkpoint": {...}}
              │
              ▼
    Dashboard shows HITL Widget (approve / reject)
              │
         ┌────┴────┐
      APPROVE   REJECT
         │         │
         ▼         ▼
    POST /workflow/{run_id}/resume
    {"approved": true/false, "response": "..."}
         │         │
         ▼         ▼
    Graph RESUMES with hitl_approved = True/False
         │         │
         ▼         ▼
    PO submitted   Fulfillment skipped
    → Fulfillment  → ExceptionEvent raised
```

---

## 🗂 Project Structure

```
supply-demand-agentic-ai/
│
├── 🤖 agents/                   One file per LangGraph node
│   ├── orchestrator.py          Plans + finalises (2 nodes)
│   ├── demand_forecast.py       Prophet + ReAct loop
│   ├── supply_planning.py       EOQ + reorder reasoning
│   ├── inventory.py             Parallel scoring + KPIs
│   ├── procurement.py           Multi-supplier + split orders
│   ├── fulfillment.py           Priority routing + dispatch
│   └── exception_handler.py     Cascading + auto-escalation
│
├── 🔀 graph/                    LangGraph wiring
│   ├── state.py                 WorkflowState TypedDict
│   ├── workflow.py              StateGraph (fan-out/fan-in)
│   └── checkpointer.py          MemorySaver + SqliteSaver
│
├── 🔧 tools/                    36 LangChain @tool functions
│   ├── forecast_tools.py        5 forecasting tools
│   ├── inventory_tools.py       11 inventory tools
│   ├── procurement_tools.py     9 procurement tools
│   ├── fulfillment_tools.py     6 fulfillment tools
│   └── notification_tools.py   6 notification/alert tools
│
├── 📐 models/                   Pydantic v2 schemas
│   ├── demand.py                ForecastRequest, ForecastResult
│   ├── supply.py                EOQInput, PurchaseOrder
│   ├── inventory.py             SKU, StockLevel, InventoryHealth
│   └── workflow.py              WorkflowRun, AgentResult, HITL
│
├── 🌐 api/                      FastAPI backend
│   ├── main.py                  App factory + lifespan + CORS
│   ├── dependencies.py          DB session, validators, pagination
│   └── routers/
│       ├── workflow.py          POST /run, SSE stream, HITL resume
│       └── dashboard.py         KPIs, inventory, exceptions, POs
│
├── 📊 dashboard/
│   └── app.py                   Streamlit — 9 sections, 5 charts
│
├── 🗄️ data/
│   ├── schema.sql               11-table SQLite schema
│   ├── database.py              SQLAlchemy engine + session
│   ├── seed_data.py             Faker ERP mock (20 SKUs, 180 days)
│   └── sample_data/             orders.csv, inventory.csv, suppliers.csv
│
├── ⚙️ config/
│   └── settings.py              Pydantic-Settings singleton
│
├── 🧪 tests/
│   ├── conftest.py              Fixtures: test DB, mock LLM, state factory
│   ├── test_tools/              Unit tests — EOQ, ROP, health scoring
│   └── test_agents/             Integration + end-to-end chain tests
│
├── 🐳 Dockerfile.api            Multi-stage build, non-root, healthcheck
├── 🐳 Dockerfile.dashboard      Streamlit container
├── 🐳 docker-compose.yml        One-command full stack
└── 📋 requirements.txt          All dependencies
```

---

## 🚀 Quick Start

### Option A — Docker (zero setup, recommended)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/supply-demand-agentic-ai.git
cd supply-demand-agentic-ai

# 2. Configure
cp .env.example .env
# ✏️  Edit .env — add your OPENAI_API_KEY

# 3. Launch everything
docker compose up --build

# ✅ API + docs:  http://localhost:8000/docs
# ✅ Dashboard:   http://localhost:8501
```

### Option B — Local development

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env   # add OPENAI_API_KEY

# Seed the database (20 SKUs, 180 days of demand history)
python data/seed_data.py
# → 3,600 demand rows · 5 suppliers · 40 SKU-supplier links

# Terminal 1 — API
uvicorn api.main:app --reload
# → http://localhost:8000/docs

# Terminal 2 — Dashboard
streamlit run dashboard/app.py
# → http://localhost:8501
```

### Option C — Trigger a run via curl

```bash
# Start a workflow for specific SKUs
curl -X POST http://localhost:8000/workflow/run \
  -H "Content-Type: application/json" \
  -d '{"sku_ids": ["SKU-001", "SKU-011", "SKU-008"]}'

# Response: {"run_id": "RUN-A1B2C3D4", "status": "initialised", ...}

# Stream live agent events (Server-Sent Events)
curl -N http://localhost:8000/workflow/RUN-A1B2C3D4/stream
# → data: {"agent": "demand_forecast", "status": "running", ...}
# → data: {"agent": "demand_forecast", "status": "success", ...}
# → data: {"agent": "supply_planning",  "status": "running", ...}
# ...

# Get warehouse KPIs
curl http://localhost:8000/dashboard/kpis
# → {"inventory": {"fill_rate_pct": 94.5, "total_inventory_value": 188791, ...}}

# Approve a HITL checkpoint
curl -X POST http://localhost:8000/workflow/RUN-A1B2C3D4/resume \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "response": "Approved by ops manager"}'
```

---

## 📊 Live Dashboard

The Streamlit dashboard has **9 sections** that update in real time:

| Section | What It Shows |
|---------|--------------|
| **KPI Cards** | Inventory value · Fill rate · Reorder triggers · Stockouts · Open POs · Critical exceptions |
| **Inventory Health Grid** | 4×5 colour-coded heatmap (🔴 stock_out · 🟠 critical · 🟡 at_risk · 🟢 healthy) with 🔔 reorder and ⬆ overstock badges |
| **Demand Chart** | 60-day history + 7-day MA + 14-day MA per SKU (Plotly) |
| **Days of Supply Bar** | Top 10 worst DoS SKUs — worst case immediately visible |
| **Agent Timeline** | Status badges + duration per agent for the selected run |
| **PO Tracker** | Open purchase orders with supplier, SKUs, value, expected delivery |
| **Exception Feed** | Unresolved exceptions with colour-coded severity + donut chart |
| **HITL Widget** | Inline approve/reject for high-value PO checkpoints |
| **SKU Deep-Dive** | Area chart + stock movements table for any selected SKU |

---

## 🌐 API Reference

```
Core
  GET  /                          API info + version + uptime
  GET  /health                    DB connectivity + environment status

Workflow
  POST /workflow/run              Trigger a new run (returns run_id immediately)
  GET  /workflow/runs             List recent runs with status
  GET  /workflow/{run_id}         Full run status + agent results + POs + exceptions
  GET  /workflow/{run_id}/stream  SSE stream of live agent events
  POST /workflow/{run_id}/resume  Submit HITL decision (approve/reject)
  DEL  /workflow/{run_id}         Cancel a running workflow

Dashboard
  GET  /dashboard/kpis                  Warehouse KPIs (inventory, runs, POs, exceptions)
  GET  /dashboard/inventory             All SKU health records with status classification
  GET  /dashboard/inventory/{sku_id}    Single SKU detail + movements + demand + open POs
  GET  /dashboard/exceptions            Exception feed (filterable by severity/resolved)
  GET  /dashboard/purchase-orders       Open PO tracker with line items
  GET  /dashboard/agents/{run_id}       Per-agent status + duration breakdown
  GET  /dashboard/demand/{sku_id}       Demand history (configurable window up to 180 days)
```

Full interactive docs at **http://localhost:8000/docs** (Swagger UI).

---

## 🛠 Tech Stack

| Layer | Technology | Why chosen over alternatives |
|-------|-----------|-------------------------------|
| **Agent Framework** | **LangGraph 0.2+** | Explicit state machine with typed nodes and conditional edges. Superior to CrewAI (too opinionated) and AutoGen (less control over routing) for production workflows |
| **LLM** | **GPT-4o** | Best-in-class tool calling with structured JSON output. The `bind_tools()` + ReAct pattern works reliably across all 7 agents |
| **ML Forecasting** | **Prophet** | Facebook's production-grade model handles weekly/annual seasonality and trend changepoints. Moving-average fallback for sparse data |
| **Concurrency** | **asyncio + ThreadPoolExecutor** | Async FastAPI for the API layer; thread pool for parallel LLM commentary in the inventory agent |
| **Data Validation** | **Pydantic v2** | Type-safe agent I/O with `@computed_field`, validators, and auto-serialisation. All 40+ WorkflowState fields are validated |
| **API** | **FastAPI** | Async, auto-generated OpenAPI, SSE streaming for live events, dependency injection |
| **Dashboard** | **Streamlit** | Fast interactive UI with Plotly charts; HITL approval widget; real-time polling |
| **Database** | **SQLite + SQLAlchemy** | Zero-infrastructure with WAL mode for concurrent reads. 11-table schema handles full audit trail. Easily swap to Postgres |
| **Containerisation** | **Docker Compose** | Multi-stage builds (2× smaller images); non-root users; healthchecks with dependency ordering |
| **Testing** | **Pytest** | 54+ tests across 3 layers: unit (tools), integration (agents), end-to-end (full chain) |

---

## 🔑 Key Design Decisions

### Why LangGraph over CrewAI or AutoGen?

LangGraph gives **explicit control over the state machine topology**. Every node is a pure Python function, conditional edges are just functions returning strings, and the checkpointer enables HITL interrupts without framework magic. In CrewAI, the execution order is abstracted; in LangGraph, you own it. This matters for production reliability and debugging.

### Why the fan-out/fan-in pattern for inventory?

With 20 SKUs, sequential scoring takes 20× longer than parallel. The `sku_dispatcher → sku_worker×20 → sku_aggregator` pattern using LangGraph's `Send` API processes all SKUs simultaneously. The aggregator merges results into a single `InventorySnapshot` — the same interface for downstream agents regardless of SKU count. Easily scales to 2,000 SKUs.

### Why split orders for critical SKUs?

Single-supplier dependency for a critical SKU is a business risk. Splitting 70/30 across two suppliers raises on-time delivery probability from e.g. `0.88` to `1-(1-0.97)*(1-0.88) = 99.6%`. The rationale is written to every PO's `notes` field for full auditability.

### Why HITL at $5,000 and not fully automated?

Real procurement requires human oversight for large financial commitments. The $5K threshold is configurable (`HITL_VALUE_THRESHOLD` in procurement.py). The graph uses `__interrupt__` to pause deterministically — the human's response is re-injected via `POST /workflow/{run_id}/resume`, and execution resumes from the exact checkpoint.

### Why SQLite and not Postgres?

SQLite in WAL mode handles this workload (20 SKUs, batch runs, dashboard reads) without any infrastructure setup, making the project instantly runnable. The SQLAlchemy layer and the schema design (`DATABASE_URL` env var) make a Postgres swap a one-line change.

---

## 🧪 Testing

```bash
pytest tests/ -v --tb=short
```

**54+ tests** covering:

| Layer | Tests | What's tested |
|-------|-------|---------------|
| Tool functions | 27 | EOQ formula accuracy, ROP math proof, health score bounds, sort order invariants, Z-score thresholds |
| Agent nodes | 8 | Status fields, result shape, state propagation, HITL guard, qty constraints |
| Full chain | 3 | All 6 agents in sequence, state accumulation, exception flag lifecycle |
| API endpoints | 10 | DB queries, run creation, status transitions, health check |
| Dashboard data | 10 | KPI shapes, inventory classification, demand chart, exception feed |

---

## 📈 Data Model

```
skus ─────────────────── sku_suppliers ──── suppliers
 │                             │
 ├── inventory_levels          │
 ├── stock_movements           │
 ├── orders (demand history)   │
 │                             │
 └── purchase_order_lines ─── purchase_orders
                                    │
workflow_runs ─── agent_results ────┘
     │
     └── exception_events
```

11 tables, WAL mode, foreign key constraints, 15+ indexes.

---

## 🗺 Build Roadmap

- [x] **Phase 1** — Foundation: Pydantic models, LangGraph state, SQLite + 180-day seed data
- [x] **Phase 2** — State Machine: 7 agent nodes, StateGraph, HITL checkpointer
- [x] **Phase 3** — Deep Tools: 22 → 36 @tools, ReAct upgrades, 54+ tests
- [x] **Phase 4** — Advanced: Parallel fan-out, split orders, bulk scoring, inventory KPIs
- [x] **Phase 5** — Fulfillment: Priority routing, cascading exceptions, auto-escalation
- [x] **Phase 6** — Full Stack: FastAPI (11 endpoints + SSE) + Streamlit (9 sections)
- [x] **Phase 7** — Production: Docker Compose, README, Interview Prep

