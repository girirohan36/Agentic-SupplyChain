# Architecture — Supply Demand Execution (Agentic AI)

A deep-dive technical reference for the multi-agent supply chain system. This
document explains *what* was built, *how* it works, and — most importantly —
*why* each decision was made. Written for senior-engineer / staff-level review.

---

## 1. System at a Glance

```
                              ┌──────────────────────────────────────┐
                              │           CLIENTS                      │
                              │  Streamlit Dashboard   ·   cURL / SDK  │
                              └────────────────┬─────────────────────┘
                                               │ HTTP + SSE
                              ┌────────────────▼─────────────────────┐
                              │        FastAPI (api/main.py)          │
                              │  /workflow  ·  /dashboard  ·  /health │
                              │  SSE streaming · HITL resume endpoint │
                              └────────────────┬─────────────────────┘
                                               │ graph.invoke / astream
                              ┌────────────────▼─────────────────────┐
                              │   LangGraph StateGraph (graph/)       │
                              │   7 agents · 4 conditional edges      │
                              │   fan-out/fan-in · HITL interrupt     │
                              └────────────────┬─────────────────────┘
                          ┌────────────────────┼────────────────────┐
                          ▼                    ▼                    ▼
                  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
                  │  36 @tools   │    │  Checkpointer │    │  GPT-4o LLM  │
                  │ (tools/*.py) │    │  Memory/Sqlite│    │ (bind_tools) │
                  └──────┬───────┘    └──────┬───────┘    └──────────────┘
                         │                   │
                         ▼                   ▼
                  ┌─────────────────────────────────────┐
                  │  SQLite + SQLAlchemy (WAL mode)       │
                  │  11 app tables + 3 checkpoint tables  │
                  └─────────────────────────────────────┘
```

**Stack:** LangGraph 0.2 · GPT-4o · FastAPI · Streamlit · Prophet · Pydantic v2 ·
SQLite + SQLAlchemy · Docker Compose · Pytest.

**Scale of the build:** 7 agents · 36 tools · 11 DB tables · ~40 `WorkflowState`
fields · 54+ tests · 11 API endpoints · 9 dashboard sections. Seeded with 20 SKUs,
5 suppliers, and 3,600 rows of demand history (180 days × 20 SKUs).

---

## 2. The Core Abstraction — `WorkflowState`

Everything in the system flows through a single `TypedDict`,
`graph/state.py::WorkflowState`. It is the *single source of truth* that every
node reads from and writes to.

### Design principles

1. **`total=False`** — every field is optional. A node returns only the *partial*
   dict it owns; LangGraph merges that patch into the running state (reducer
   pattern). This is why every agent returns `dict[str, Any]` containing just its
   own keys, never the full state.
2. **Plain dicts, not Pydantic models** — nested objects are dict-serialised so the
   built-in `MemorySaver` / `SqliteSaver` checkpointers can pickle them. Pydantic
   models are the *contract* at the edges (validation, API I/O); inside the graph
   we travel as dicts. Conversion happens at write/read boundaries.
3. **Naming convention** encodes data flow:
   - `*_input`  → data fed *into* an agent
   - `*_result` → data produced *by* an agent
   - `*_status` → per-agent lifecycle (`pending → success | failed | skipped`)
   - boolean flags → control signals consumed by conditional edges
4. **Reducers** — `messages` uses LangGraph's `add_messages` reducer so message
   history auto-appends. `per_sku_results` accumulates partial results from
   parallel workers the same way.

### Field groups

| Group | Key fields | Owner |
|-------|-----------|-------|
| Run metadata | `run_id`, `triggered_by`, `sku_ids`, `current_sku` | Orchestrator |
| Messages | `messages` (add_messages reducer) | All agents |
| Per-agent I/O | `<agent>_input`, `<agent>_result`, `<agent>_status` | Each agent |
| HITL | `hitl_required`, `hitl_checkpoint`, `hitl_response`, `hitl_approved` | Procurement / API |
| Control flags | `replenishment_needed`, `exception_detected`, `fulfillment_ready`, `run_complete` | Routers |
| Parallel accumulator | `per_sku_results`, `sku_processing_complete` | sku_worker / aggregator |
| Aggregated output | `summary` | Orchestrator (finalise) |

> **Why a TypedDict instead of a Pydantic model for the state itself?**
> LangGraph's reducer/merge machinery operates on dict patches. A `TypedDict`
> gives us static-type ergonomics in the editor while staying a plain `dict` at
> runtime — no serialization friction, no `.model_dump()` on every hop.

---

## 3. The Graph Topology

Built in `graph/workflow.py::build_graph()`. Compiled `StateGraph` over
`WorkflowState`.

```
START
  │
  ▼
orchestrator_plan ──► demand_forecast ──► supply_planning
                                              │
                                              ▼
                                       sku_dispatcher  ◄── fan-out (Send API)
                                       ┌──────┼──────┐
                                       ▼      ▼      ▼
                              sku_worker  sku_worker  …×20   (parallel)
                                       └──────┼──────┘
                                              ▼
                                       sku_aggregator  ◄── fan-in (reducer merge)
                                              │
                                              ▼
                                         inventory
                                              │
                        ┌── route_after_inventory ──┐
            replenishment_needed?                   else
                        ▼                             │
                  procurement                         │
                        │                             │
          ┌─ route_after_procurement ─┐               │
       hitl_required?                  else            │
              ▼                         │              │
            END (interrupt)            ▼              ▼
                                    fulfillment ◄──────┘
                                          │
                        ┌─ route_after_fulfillment ─┐
                  exception_detected?               else
                          ▼                          │
                  exception_handler                  │
                          │                          │
                          └──────────┬───────────────┘
                                     ▼
                            orchestrator_finalise ──► END
```

### Nodes

| Node | File | Responsibility |
|------|------|----------------|
| `orchestrator_plan` | agents/orchestrator.py | LLM produces execution plan, resets statuses |
| `demand_forecast` | agents/demand_forecast.py | Prophet + LLM ReAct forecasting |
| `supply_planning` | agents/supply_planning.py | EOQ, reorder point, safety stock |
| `sku_dispatcher` | graph/workflow.py | Fan-out — emits one `Send` per SKU |
| `sku_worker` | graph/workflow.py | Per-SKU inventory scoring (parallel) |
| `sku_aggregator` | graph/workflow.py | Fan-in — merges per-SKU results |
| `inventory` | agents/inventory.py | Health scoring, KPIs, expiry risk |
| `procurement` | agents/procurement.py | PO generation, supplier selection, split orders |
| `fulfillment` | agents/fulfillment.py | Order priority, DC routing, dispatch, backorders |
| `exception_handler` | agents/exception_handler.py | Cascade detection, escalation, notifications |
| `orchestrator_finalise` | agents/orchestrator.py | Compile run summary, mark complete |

### Conditional edges (the 4 routers)

All four live in `agents/orchestrator.py` and are wrapped by thin adapters in
`workflow.py` (the adapters translate the router's string verdict into a concrete
node name or `END`):

1. **`route_after_inventory`** — `replenishment_needed` → `procurement`, else
   `fulfillment`.
2. **`route_after_procurement`** — HITL enabled *and* `hitl_required` →
   `hitl_interrupt` (mapped to `END`, pausing the graph), else `fulfillment`.
3. **`route_after_fulfillment`** — `exception_detected` → `exception_handler`,
   else `orchestrator_finalise`.
4. **`route_after_exception`** — always → `orchestrator_finalise` (the handler
   resolves in place).

> **Why split planning and finalising into two orchestrator nodes?**
> It gives the graph a single clean entry and a single clean exit. The plan node
> owns "what should happen"; the finalise node owns "what *did* happen" (the
> summary). Keeping them apart avoids a single god-node that has to branch on
> whether the run is starting or ending.

---

## 4. The Agent Pattern (ReAct + Fallback)

Every agent follows an identical, deliberately boring shape. This consistency is
the system's most important reliability property.

```python
def <agent>_node(state: WorkflowState) -> dict[str, Any]:
    try:
        # ── HAPPY PATH: LLM ReAct loop ──────────────────────
        llm_with_tools = _get_llm_with_tools()          # bind_tools(...)
        messages = [SystemMessage(...), HumanMessage(...)]
        messages, final = _run_react_loop(llm_with_tools, messages, TOOL_MAP)
        result = json.loads(_strip_fences(final))       # structured output
    except Exception as exc:
        # ── FALLBACK: deterministic direct tool calls ───────
        result = _fallback_<agent>(...)                 # no LLM needed
        result["_fallback_reason"] = str(exc)

    return { "<agent>_result": ..., "<agent>_status": "success", ... }
```

### `_run_react_loop()` — the shared engine

Defined once in `agents/demand_forecast.py` and **imported by every other
agent**. It is a minimal, dependency-free ReAct implementation:

```
for _ in range(MAX_ITERATIONS):           # safety ceiling (default 10)
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    if not response.tool_calls:           # LLM done reasoning
        return messages, response.content
    for tc in response.tool_calls:        # execute each requested tool
        fn = tool_map[tc["name"]]
        result = fn(**tc["args"])          # ← direct Python call
        messages.append(ToolMessage(json.dumps(result), tc["id"]))
```

Two non-obvious decisions are baked in here:

- **Tools are invoked as `fn(**args)` — plain Python calls — *not*
  `tool.invoke({dict})`.** The `@tool` decorator is used purely to give the LLM a
  schema for function-calling; the local execution path calls the underlying
  function directly. This avoids LangChain's invoke wrapper overhead and keeps
  stack traces readable. Each agent maintains an explicit `TOOL_MAP` dict mapping
  the tool name the LLM emits → the local callable.
- **A hard iteration ceiling** (`MAX_AGENT_ITERATIONS`, default 10) guarantees the
  loop can never run away on a model that keeps calling tools.

### Why every agent has a fallback

The fallback path runs the same tools **deterministically, in a fixed order,
without an LLM**. This buys three things:

1. **The system runs with no `OPENAI_API_KEY`** — every agent degrades to a
   rules-based pipeline. Critical for local dev, CI, and sandbox testing where no
   LLM is reachable.
2. **Resilience** — a transient LLM error (rate limit, timeout, malformed JSON)
   doesn't fail the run; it silently falls back and records `_fallback_reason`.
3. **Testability** — 54+ tests exercise the deterministic paths without mocking an
   LLM round-trip.

> This is the single most defensible design choice in the project: *the LLM is an
> optimisation, not a dependency.* The supply chain still functions if GPT-4o is
> down.

---

## 5. Fan-out / Fan-in — Parallel SKU Processing

Inventory scoring is embarrassingly parallel: each SKU's health is independent.
Phase 4 exploits this with LangGraph's **Send API**.

### Fan-out (`sku_dispatcher`)

```python
def sku_dispatcher(state) -> list[Send]:
    from langgraph.types import Send
    sends = []
    for sku_id in state["sku_ids"]:
        per_sku_state = { "current_sku": sku_id, "sku_ids": [sku_id], ... }
        sends.append(Send(NODE_SKU_WORKER, per_sku_state))
    return sends          # LangGraph runs all of them concurrently
```

Returning a `list[Send]` from a conditional edge tells LangGraph to spawn one
`sku_worker` instance per `Send`, each with its **own isolated state slice**
(just that SKU's forecast + plan). 20 SKUs → 20 parallel workers.

### Fan-in (`sku_aggregator`)

Each worker writes a one-element list to `per_sku_results`. Because that field
uses an accumulating reducer, LangGraph merges all 20 partials into one list. The
aggregator then computes the rollup:

```python
critical = sum(1 for r in per_sku if r["health_status"] in ("critical","stock_out"))
at_risk  = sum(1 for r in per_sku if r["health_status"] == "at_risk")
inventory_result = { "records": per_sku, "critical_count": critical, ... }
```

### Graceful degradation

`sku_dispatcher` and `_route_after_supply_planning` both wrap
`from langgraph.types import Send` in a `try/except ImportError`. On older
LangGraph versions where `Send` is unavailable, the graph skips the fan-out and
falls through to a direct single-pass `inventory` node — same result, no
parallelism. The topology is self-healing across library versions.

> **Why Send instead of `asyncio.gather` inside one node?**
> Send keeps each parallel branch as a *first-class graph node*. That means every
> worker is independently checkpointed, independently visible in the LangSmith
> trace, and independently resumable. Hiding the parallelism inside a single node
> would make it opaque to the checkpointer and the observability layer.

---

## 6. Human-in-the-Loop (HITL)

HITL is the mechanism that keeps a human in control of high-stakes spend.

### Trigger

The Procurement agent sets `hitl_required = True` when a generated PO crosses the
**\$5,000** value threshold. `route_after_procurement` then routes to
`hitl_interrupt`, which the workflow maps to `END` — pausing the graph mid-run at
a checkpoint.

### Pause → Resume lifecycle

```
POST /workflow/run
   │  graph.invoke(...)
   ▼
…runs through procurement…
   │  PO ≥ $5,000  →  hitl_required = True
   ▼
graph pauses at __interrupt__  (state persisted by checkpointer)
   │
   ▼
API returns 200 with hitl_checkpoint payload (the question + context)
   │
   …human reviews in dashboard HITL widget…
   │
   ▼
POST /workflow/{run_id}/resume   { "approved": true, "response": "..." }
   │  graph.invoke(resume_with_approval(True), config=thread_config(run_id))
   ▼
graph resumes from the exact checkpoint → fulfillment → … → END
```

The checkpointer is what makes this possible: the *entire* graph state is
persisted at the interrupt, so resuming is a true continuation, not a re-run.
`thread_config(run_id)` ties the resume call to the correct thread via
`{"configurable": {"thread_id": run_id}}`.

### Helpers (`graph/checkpointer.py`)

- `build_interrupt_payload(...)` — constructs the structured question + context +
  timeout that the API surfaces (matches the `HITLCheckpoint` schema).
- `resume_with_approval(approved, response)` — builds the partial state patch
  (`hitl_required=False`, `hitl_approved`, `hitl_response`) that is merged back in
  on resume.

### Auto-escalation

A second, automatic HITL trigger fires when **≥3 critical exceptions** accumulate
in one run. The exception handler forces an interrupt *and* fans out
notifications (email + Slack). This protects against the "death by a thousand
cuts" scenario where no single exception is severe but the aggregate is alarming.

---

## 7. Checkpointer Strategy

`graph/checkpointer.py` selects the persistence layer by environment:

| Env | Checkpointer | Why |
|-----|-------------|-----|
| development / `TESTING` | `MemorySaver` | In-process, zero setup, fast tests |
| staging / production | `SqliteSaver` | Durable, survives restarts, enables true HITL resume |

`SqliteSaver` writes to the *same* `supply_demand.db` as app data and
auto-creates its own tables (`checkpoints`, `checkpoint_blobs`,
`checkpoint_writes`) on first use. This co-location means one file is the
complete system of record: business data *and* every state transition.

The checkpointer delivers four properties for free:

1. State persistence across async API turns.
2. HITL interrupts (pause/resume).
3. Fault tolerance (resume from last good checkpoint).
4. A full audit trail of every state transition — invaluable for a supply chain
   where "why did we order 10,000 units?" must be answerable.

---

## 8. The Tool Layer (36 tools)

Tools are plain Python functions decorated with `@tool`. Grouped by domain in
`tools/`:

| Module | Tools | Examples |
|--------|-------|----------|
| `forecast_tools.py` | 5 | `run_prophet_forecast`, `compute_mape`, `detect_demand_anomalies`, `get_demand_trend`, `load_demand_history` |
| `inventory_tools.py` | ~10 | `bulk_score_all_skus`, `get_inventory_kpis`, `check_expiry_risk`, `update_stock_level`, `record_stock_movement` |
| `procurement_tools.py` | ~7 | `compare_suppliers`, `calculate_split_order`, EOQ/PO generation |
| `fulfillment_tools.py` | 6 | `get_open_orders`, `score_order_priority`, `route_to_best_dc`, `create_dispatch`, `manage_backorder`, `calculate_fill_rate` |
| `notification_tools.py` | several | email + Slack alert dispatch |

**Two invariants for every tool:**

1. **Direct-call internally** — `tool_fn(arg=val)`, never `.invoke({dict})`.
2. **`DB_PATH` must be patched** in each tool module before testing, so tests run
   against an isolated temp DB rather than the seeded production file.

### Key domain formulas

These are implemented in the tool layer and are the quantitative heart of the
system:

- **Health score** (composite, 0–1):
  `score = 0.50·dos + 0.30·rop + 0.20·ss`
  where `dos` = days-of-supply factor, `rop` = reorder-point factor, `ss` =
  safety-stock factor.
  Bands: `> 0.75` healthy · `0.40–0.75` at_risk · `< 0.40` critical.

- **EOQ** (Economic Order Quantity):
  `EOQ = √(2·D·S / H)` where `H = unit_cost × 0.25` (25% annual holding cost),
  `D` = annual demand, `S` = order setup cost.

- **Split-order combined reliability** (critical urgency, 70/30 across two
  suppliers):
  `reliability = 1 − (1 − p₁)(1 − p₂)`
  — the probability that at least one supplier delivers, which is why splitting a
  critical order across two suppliers is strictly safer than a single source.

---

## 9. Data Layer

### Schema (`data/schema.sql`)

11 application tables (SKUs, suppliers, inventory, demand history, purchase
orders, orders, dispatches, exceptions, workflow_runs, etc.) plus the 3
checkpoint tables created by `SqliteSaver`.

- **WAL mode** (`PRAGMA journal_mode = WAL`) — Write-Ahead Logging lets the API
  read while the graph writes, without lock contention. Essential when the
  dashboard polls KPIs while a workflow is mid-run.
- **Foreign keys ON** (`PRAGMA foreign_keys = ON`) — referential integrity
  enforced at the DB level.
- **Parenthesis-depth-aware semicolon splitter** — the schema loader in
  `api/main.py` does *not* naively split on `;`. It tracks `(`/`)` depth and only
  splits at depth 0, so multi-line `CREATE TABLE` statements containing
  `CHECK(... ; ...)`-style content survive intact.

### Seed data (`data/seed_data.py`)

20 SKUs · 5 suppliers · 180 days of per-SKU demand history → **3,600 demand
rows**. Enough volume to make Prophet forecasts meaningful and to populate every
health band (healthy / at_risk / critical) for demo and test coverage.

---

## 10. API Layer (FastAPI)

`api/main.py` is an app factory with 11 endpoints across two routers plus core
endpoints.

| Concern | Implementation |
|---------|----------------|
| Lifespan | Initialises DB (idempotent schema load) on startup |
| CORS | Allows Streamlit (`:8501`), React (`:3000`), same-origin |
| Timing | Middleware injects `X-Process-Time` header on every response |
| Errors | Global 404 / 500 handlers return structured JSON |
| Health | `/health` pings the DB and returns sku/run counts + uptime |
| Streaming | SSE endpoint streams live agent events as the graph runs |
| HITL | `POST /workflow/{run_id}/resume` continues a paused graph |
| Docs | Swagger at `/docs`, ReDoc at `/redoc`, schema at `/openapi.json` |

Routers: `routers/workflow.py` (run / stream / resume / status) and
`routers/dashboard.py` (KPIs, inventory, exceptions, PO tracker feeds for the
Streamlit UI).

---

## 11. Configuration

`config/settings.py` uses `pydantic-settings` with an `lru_cache`'d singleton
(`get_settings()`). All values come from env / `.env`. Notable knobs:

| Setting | Default | Purpose |
|---------|---------|---------|
| `OPENAI_MODEL` | `gpt-4o` | LLM for all agents |
| `OPENAI_TEMPERATURE` | `0.0` | Determinism for reproducible plans |
| `MAX_AGENT_ITERATIONS` | `10` | ReAct loop ceiling |
| `ENABLE_HUMAN_IN_THE_LOOP` | `True` | Master HITL switch |
| `FORECAST_HORIZON_DAYS` | `90` | Prophet horizon |
| `DEFAULT_SAFETY_STOCK_DAYS` | `14` | Safety stock baseline |
| `DEFAULT_LEAD_TIME_DAYS` | `7` | Replenishment lead time |
| `LOW_STOCK_THRESHOLD_PCT` | `0.20` | Low-stock flag threshold |

`temperature=0.0` is deliberate: in a supply chain you want the *same* inputs to
produce the *same* plan. Creativity is a liability here.

---

## 12. Deployment

Two-stage Docker build, two services via Compose:

- **`Dockerfile.api`** — multi-stage (builder + slim runtime), runs as a non-root
  user, exposes the FastAPI service.
- **`Dockerfile.dashboard`** — Streamlit service.
- **`docker-compose.yml`** — both services + a named volume for the SQLite DB +
  an API healthcheck the dashboard waits on.
- **`.dockerignore`** — keeps the build context lean (no `__pycache__`, no `.db`,
  no tests).

```bash
cp .env.example .env          # add OPENAI_API_KEY
docker compose up --build
```

The named volume is what makes the SQLite-as-system-of-record choice viable in
containers: the DB (app data *and* checkpoints) survives container recreation.

---

## 13. Why These Choices — Trade-offs Summary

| Decision | Alternative considered | Why we chose this |
|----------|----------------------|-------------------|
| LangGraph StateGraph | Plain function pipeline / CrewAI | Need conditional routing, checkpointing, HITL interrupts, and parallel Send — LangGraph gives all four natively |
| TypedDict state (dicts inside) | Pydantic model as state | Reducer merging + picklable checkpoints; Pydantic stays at the edges |
| ReAct + deterministic fallback | LLM-only agents | System must run with no API key and survive LLM outages |
| Direct `fn(**args)` tool calls | `tool.invoke({dict})` | Less overhead, readable traces, simpler test seams |
| Send API fan-out | `asyncio.gather` in one node | First-class checkpointing + observability per branch |
| SQLite + WAL | Postgres | Single-file simplicity for a demo-scale system; WAL handles read/write concurrency |
| Two orchestrator nodes | One god-node | Clean entry/exit; separates "plan" from "summary" |
| HITL at \$5k + 3-exception escalation | Always-on or never | Balances autonomy with human control where money/risk is highest |

---

## 14. Where It Would Go Next

Honest limitations and the natural evolution path:

- **SQLite → Postgres** for true multi-writer concurrency at production scale; the
  SQLAlchemy layer makes this a config change.
- **Per-SKU forecast map** — `demand_forecast_node` currently surfaces the last
  SKU's forecast as the "primary"; a per-SKU result map is the documented next
  step.
- **Distributed checkpointer** (Redis/Postgres) to run the graph across multiple
  API workers.
- **Real supplier/ERP integrations** behind the existing tool interfaces — the
  tool abstraction is already the right seam for this.
- **Auth + multi-tenancy** on the API (currently open CORS for dev).
- **LangSmith tracing** is wired via settings but off by default — flip on for
  production observability.

---

*This document reflects the implementation as of Phase 7. Every formula, node
name, threshold, and file path above is taken directly from the codebase.*
