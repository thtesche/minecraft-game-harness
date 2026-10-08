# Architecture: Laya-Gated Minecraft Harness

**Applies to:** the harness described in [harness_idea.md](harness_idea.md)
**Status:** design · **Repo:** `minecraft-game-harness`

The harness is a standalone application. It owns the decision loop around
mine-ai-mcp, gates the repetitive majority of decisions through Laya, and escalates
the minority to a frontier LLM.

This document specifies the components, the data flow, the protocol handling, the
failure modes, and the build order. Decisions D1–D8 from the idea document are
referenced throughout and are not restated.

---

## 1. Runtime topology

```
┌─────────────────────────────────────────────────────────────────────┐
│  harness  (Python 3.12, standalone process, survives session end)     │
│                                                                     │
│   ┌──────────────┐   ┌───────────────┐   ┌──────────────────────┐   │
│   │ StateReader  │──▶│ DecisionNode  │──▶│ ObjectiveRunner      │   │
│   │ SQL + views  │   │ Laya / LLM    │   │ protocol state machine│  │
│   └──────────────┘   └───────────────┘   └──────────┬───────────┘   │
│          ▲                  │                        │               │
│          │                  │ escalate               │ MCP call     │
│          │                  ▼                        ▼               │
│          │            ┌───────────────┐   ┌──────────────────────┐   │
│          │            │  LLMNode      │   │ Ledger (JSONL+SQLite)│  │
│          │            │  frontier     │   └──────────────────────┘   │
│          │            └───────────────┘                              │
│          └──────────────────────┘                                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ Streamable HTTP, one foreground slot
                                 │ tool timeout 3600000 ms
┌────────────────────────────────▼────────────────────────────────────┐
│  mine-ai-mcp  (Bun ≥ 1.4, prebuilt/pinned)                           │
│    ├── survival reflex   (deterministic, no LLM, no gate)           │
│    ├── 34 tools          (high-level objectives, own verification)   │
│    └── SQLite            (bot_status, bot_inventory, bot_tools,     │
│                           action_executions, action_requests, …)    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ Mineflayer + Pathfinder
┌────────────────────────────────▼────────────────────────────────────┐
│  Minecraft Java 1.21.4                                               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  Laya  (in-process, English checkpoint, MPS or CPU, preloaded)      │
│    LayaDecision / LayaRouter  ·  hooks: Memo cache + Counter        │
└─────────────────────────────────────────────────────────────────────┘
```

**Three processes, three roles, no chat dependency.** The harness is the only
component under our control; the server is pinned by commit, and the model runs
inside the harness process so no extra IPC hop sits on the decision path (D5).

## 2. Components

### 2.1 `StateReader`

Produces the decision state from the server's durable records. It reads, in order:

1. **`view_status`** — vitals, position, heading, carried items, best tool and
   armour tiers, nearby entities, the `mobility` section. This also refreshes
   `bot_status`, `bot_inventory` and `bot_tools` in SQLite.
2. **`query_bot_data`** — one read-only `SELECT`/`WITH` per query, against the
   refreshed tables plus chunk records for world-scale facts the bot cannot see
   from where it stands.
3. **Cheap deterministic calls, before any model is consulted** (D7):
   - `view_frontier` for unexplored-chunk distance and bearing
   - the `mobility` verdict from `view_status` for what the current inventory permits

**Not `view_crafting_requirements`.** It was planned here as the goal-decomposition
step and has been removed on measurement (D18): the reference run called it **zero**
times in 388 calls and instead asked `craft_item` for the items it wanted, reading the
recipe tree, the leaf materials and the workstation requirement out of the *craft
reply*. A harness that plans first pays for information the answer already carries. A
goal is an item name, and the server does the decomposition.

A view failure is **not** a default value. It is an unverified read that escalates
(D-recall from ai-minebot §8.9.3: a parser that stops matching must fail the
command, not hand the caller a plausible number).

### 2.2 `DecisionNode`

Laya, in-process, over a compact state (D4). Three questions, each a small option
set, per [harness_idea.md §3](harness_idea.md#3-the-idea):

```python
decide = LayaDecision(
    ObjectivePlan,          # pydantic; built at construction, validated then
    max_len=512,            # English checkpoint window
    head_max_len=192,       # default for `laya`; raise only for a wide question
    lang="en",
    min_confidence=THRESHOLD,   # fitted in Phase 3, not assumed
    hooks=[Memo(), Counter()],
    hooks_timeout=0.25,
    return_details=True,
)
```

```python
class ObjectivePlan(BaseModel):
    category: Literal["gather", "craft", "smelt", "store",      # ≤ 6 options (D1)
                      "navigate", "defend", "wait", "escalate"]
    finished: bool          # is the current objective done?
    escalate: bool          # hand this to the frontier LLM?
```

`Literal` enums are what makes this work: `LayaDecision` validates the schema at
construction, so an unsupported field raises `SchemaError` immediately rather than
after the graph has paid for earlier steps.

**Constraints that are not negotiable**, from Laya's own documentation:

- **Option count.** Past ~20 options, labels are trimmed until similar ones reach
  the model as identical text — a wrong answer, not an error. Measured: 48 options
  score **1/48** at the default budget, **43/48** with `head_max_len=512`. Eight
  categories is nowhere near that cliff.
- **State size.** 4,000 tokens costs **~1.7 s** on an Apple GPU and accuracy falls
  to 8–17/20 past ~4,000 tokens. The state is a derived feature vector, not a raw
  inventory dump. If it exceeds the window, that is a design bug, not a bigger
  `max_len`.
- **Device concurrency.** Concurrent MPS forwards abort the process
  (`failed assertion _status < MTLCommandBufferStatusCommitted`). `max_concurrency=1`,
  or use `batch()` — one shared forward pass, ~2.2× on MPS.
- **Caching.** `Memo` skips the forward pass on a repeated state: 24 states over 4
  distinct tickets measured 24 passes / 2109 ms → 4 passes / 330 ms cold → 0 / 0.3 ms
  warm, with all 24 routes identical. State-keyed caching is the cheapest speedup
  available and it is free.

### 2.3 `LLMNode`

The frontier model, reached only on escalation. It gets the same tools as the
objective runner, bound as LangChain tools with `bind_tools()`.

It does **not** get the long-running protocol. It emits one objective; the runner
submits it. A frontier model improvising `submission_id` discipline, a single-slot
foreground gate, and `RESULT_NOT_RETRIEVED` handling deadlocks or double-submits
— see the protocol section.

### 2.4 `ObjectiveRunner`

The deterministic state machine around one foreground action (D8). This is the
component that makes the harness an application rather than a prompt.

```
submit(tool, args, submission_id, wait_timeout_ms=5000)
   │
   ├─ accepted + settled ──────────────▶ verify evidence ──▶ ledger ──▶ next decision
   ├─ pending ─────────────────────────▶ wait_for_action(action_id, 2000)
   │     ├─ settled ───────────────────▶ verify evidence ──▶ ledger ──▶ next decision
   │     └─ pending ───────────────────▶ poll with budget, or escalate
   │
   ├─ RESULT_NOT_RETRIEVED ────────────▶ wait_for_action(previous action_id) first
   ├─ ACTION_BUSY ─────────────────────▶ wait_for_action, do not resubmit
   ├─ SUBMISSION_CONFLICT ─────────────▶ recover the original handle, change nothing
   └─ transport closed ────────────────▶ re-read /health.foreground, do not resubmit
```

**Protocol rules that come from the server's contract, not from convenience:**

- Every foreground call needs a unique `submission_id`. Retry the *same* arguments
  with the *same* id to recover a lost reply; change either and it is refused.
- **One logical objective at a time.** A refusal never queues. A second submission
  before retrieval is refused with the preceding action id.
- A timeout is **not** cancellation. The bot keeps working. Inspect status before
  retrying; the README's own warning is that a timed-out client request does not
  prove the action stopped.
- Initial waits are bounded (5 s default), follow-ups short (0–30 s qualified in
  the SDK's live playtests), both capped at the client's transport limit.
- Disconnecting does not stop admitted work. Reconnect and read the Foreground
  section of `view_status`.
- **Verify settled evidence, never acceptance.** `partial`, `failed` and `cancelled`
  outcomes are terminal and keep their available evidence. A runtime failure
  explicitly names missing domain evidence — treat a missing leg as unverified,
  not as success.

**Budgets the runner enforces**, because every one of these has been measured to run
away: explicit time budget per objective, a step/attempt ceiling, consecutive-failure
tolerance, and per-goal attempt caps. A refusal is not a retry; a fact about the
world is not a hiccup.

### 2.5 `Ledger`

Append-only JSONL plus a SQLite index. One row per decision:

| Field | Purpose |
|---|---|
| `state_vector`, `state_hash` | what the model saw; the hash is the cache key |
| `question`, `options`, `raw_scores` | the exact prompt and the full distribution |
| `answer`, `answer_confidence`, `escalated` | the decision and whether it was trusted |
| `objective_tool`, `objective_args` | what was actually submitted |
| `outcome`, `evidence_ok`, `duration_ms` | what the server settled with |

Two jobs: **the eval set** (§2.2's quality gate) and **the baseline** (§7). Both
need the same rows, and both are unobtainable after the fact if the fields are not
written at decision time. Record the full distribution, not just the argmax —
calibration work needs the probabilities.

## 3. Decision flow

```
        ┌──────────────┐
        │ StateReader  │  view_status + query_bot_data + deterministic layers
        └──────┬───────┘
               │ compact state vector
               ▼
        ┌──────────────┐   answer_confidence ≥ THRESHOLD ?
        │  Laya node   │────── yes ──────────────────────────┐
        └──────┬───────┘                                     │
               │ no                                          ▼
               │                                    ┌──────────────┐
               ▼                                    │ submit       │  submission_id,
        ┌──────────────┐                           │ objective    │  bounded wait
        │  LLM node    │──── objective ───────────▶└──────┬───────┘
        │ (frontier)   │                                   │ settled
        └──────────────┘                                   ▼
                                                 ┌──────────────────┐
                                                 │ verify evidence  │── incomplete ──▶ back to decision
                                                 └────────┬─────────┘
                                                          ▼
                                                        ledger ──▶ next decision
```

**Escalation triggers**, in the order they are checked:

1. `answer_confidence` below the fitted threshold
2. A view returned unverified, so the state itself is not trustworthy
3. The objective is outside the declared category set
4. The server's own reflex has taken over and the bot is no longer where the plan
   assumed — position drifted, or an action settled `partial`

Trigger 4 is not in the idea document and is worth stating: a plan is only valid
for the world it was made in. The measured failure mode is a recipe drifting the
bot 209 → 393 blocks over six runs with no way back.

## 4. Failure handling

| Failure | Handling | Never |
|---|---|---|
| Client timeout | Read status before retrying; the action may still be running | Assume cancellation |
| `ACTION_BUSY` | `wait_for_action` on the active id | Resubmit |
| `RESULT_NOT_RETRIEVED` | Retrieve the previous result, then submit | Treat as a new decision point |
| `SUBMISSION_CONFLICT` | Recover the original handle from the identical call | Change arguments to force it through |
| Transport closed | Re-read `/health.foreground`, reconnect | Assume work stopped |
| `RUNTIME_INTERRUPTED` | Explicit failure with stale checkpoints — never auto-replay | Replay a durable row |
| Death mid-objective | An ordinary step outcome: absorb, report, continue | Crash the run |
| Unparsed read | Report unverified, escalate | Substitute a plausible default |
| Model emits an out-of-set label | Discard, treat as escalation | Map it to the nearest option |
| MPS concurrent forward | `max_concurrency=1`, or `batch()` | Thread-pool the graph |
| Slow action (10+ min smelt) | Budget 3600000 ms at the client, poll with short waits | Retry the objective |

## 5. Configuration

| Key | Default | Note |
|---|---|---|
| `mcp.url` | `http://localhost:25575/mcp` | Streamable HTTP |
| `mcp.health_url` | `http://localhost:25575/health` | Health endpoint |
| `mcp.tool_timeout_ms` | `3600000` | **Not optional.** A 10-minute smelt exceeds a 5-minute idle window |
| `mcp.initial_wait_ms` | `5000` | Bounded initial wait |
| `mcp.poll_ms` | `2000` | Follow-up wait |
| `mcp.max_polls` | `900` | No unbounded polling |
| `mcp.max_sse_event_bytes` | `8388608` | Max SSE event size to avoid OOM |
| `mcp.submission_prefix` | `""` | One prefix per process run for attribution |
| `ledger.path` | `runs/ledger.jsonl` | JSONL append-only log |
| `ledger.sqlite_path` | `runs/ledger.sqlite` | SQLite index (optional) |
| `budget.objective_ms` | `900000` | Time budget per objective |
| `budget.max_attempts_per_goal` | `12` | Per-goal attempts ceiling |
| `budget.max_consecutive_failures` | `3` | Tolerate consecutive failures |
| `budget.gate_wait_ms` | `180000` | Wait for survival reflex before giving up |
| `laya.model` | `laya` | English checkpoint |
| `laya.device` | `mps` | **Precondition, not an optimisation.** CPU at 88 ms is not a loop budget |
| `laya.preload` | `true` | Preload checkpoint to avoid switch cost |
| `laya.lang` | `en` | Language; `multilingual` only if state exceeds English window |
| `laya.max_len` | `512` | Max token length for state |
| `laya.head_max_len` | `192` | Option-prompt budget |
| `laya.max_concurrency` | `1` | MPS aborts on concurrent forwards |
| `laya.min_confidence` | **fitted in Phase 3** | No default until eval exists (D3) |
| `llm.model` | `""` | Frontier model, escalation only |
| `llm.api_key_env` | `OPENROUTER_API_KEY` | Env var for API key |
| `llm.base_url` | `https://openrouter.ai/api/v1` | OpenRouter compatible endpoint |
| `llm.temperature` | `0.0` | Sampling temperature |
| `run_id` | `""` | One prefix per process run (fallback to "harness") |

## 6. Build order

Each phase has an exit criterion. A phase that cannot meet its criterion does not
proceed — that is the difference between a measurement and an assumption.

### Phase 0 — Skeleton

Connect to mine-ai-mcp, read `view_status`, run one objective end to end, write a
ledger row. Own process, config file, no chat session.

**Exit:** one objective completes unattended, and a ledger row exists for it.

### Phase 1 — The runner

The protocol state machine of §2.4, complete: bounded waits, `RESULT_NOT_RETRIEVED`,
`ACTION_BUSY`, `SUBMISSION_CONFLICT`, cancellation, evidence verification, death
absorption. Multiple objectives in sequence.

Two of these are named omissions rather than gaps. **`SUBMISSION_CONFLICT` has no
recovery** (D15): the refusal names no action, so the submission holding that id cannot
be found and may still be running. **Reconnection has detection but no recovery** (D28):
`runtime_unavailable` reads `/health` and stops, because
`mine-ai-mcp/src/server/runtime-host.ts:143` states that "a new Minecraft connection
requires an explicit service restart" — there is no reconnect for the harness to drive.
An unbuildable half is recorded as unbuildable rather than stubbed to look finished.

`cancellation` (`cancel_foreground_action`) is the one item in this phase still
unbuilt, and it is worth real money rather than just time: a decision costs 20–37 s
and a cancelled objective wastes it.

**Exit:** a scripted multi-objective run completes with every ledger row carrying
verified evidence. **This phase is the harness.** Phases 2–4 are an optimisation
layer on top of it; if Phase 1 is skipped, nothing else has anything to sit on.

### Phase 2 — Baseline

Run the scenario set with the LLM node on every decision. Record LLM calls per
objective, success rate, wall clock, and objective failure distribution.

A scenario is `scenarios/*.json`: a name, an **unordered goal list**, and a **checker**.
The checker is written before the run (D19) and is named code from
`harness.scenario.CHECKS`, not prose. A scenario with no checks is refused at load
time, and an unknown check kind is refused too — a scenario that cannot fail is
indistinguishable from one that passed. The checkers read the world themselves, from
a fresh `view_status` after the loop, because the loop's own vector is deliberately
lossy (twelve stacks) and a checker reading a lossy view reports "no pickaxe" for a
bot holding one in slot twenty.

`run-scenario scenarios/first-pickaxe.json --decider llm` exits **0 only when every
check passed and the loop finished cleanly**, so a suite can be run unattended.

**Exit:** a baseline number exists, measured on a named scenario set. This is the
comparison that makes "dramatically fewer LLM calls" falsifiable.

### Phase 3 — Laya eval gate

Build ≥ 200 labelled decisions from the Phase 2 ledger. `laya-evals` for accuracy and
ECE. Refit temperature, record ECE before and after.

**Decision rule:**

| Baseline accuracy | Consequence |
|---|---|
| ≥ 0.8 | Deploy Laya as the gate on all three questions |
| 0.6 – 0.8 | Deploy only behind a fitted confidence gate |
| < 0.6 | Fine-tuning is a project of its own, not a phase |

**Exit:** a documented decision — planner, gatekeeper, or fine-tune first — with the
reasoning attached. `laya-evals` exits non-zero on threshold violation, so this drops
into CI unchanged.

### Phase 4 — The gated loop

Wire Laya into the decision point. Fit the threshold from the Phase 3 eval and state
the coverage it costs. Enable the `Memo` cache and `Counter` hooks. Re-run the Phase 2
scenario set.

**Exit:** LLM calls per objective drop by the stated factor, and task success rate
does not regress. Both numbers reported, on the same scenario set.

### Phase 5 — Fine-tuning (conditional)

Only if Phase 3 says so. Behaviour cloning on ledger rows from the gated loop, with
the calibration split held out from training. The notebook's calibration samples come
from its own training items; evaluating on them is not an eval.

## 7. What "reduced LLM calls" is measured against

The claim has to be a number or it is a slogan. Three metrics, on a fixed scenario
set, LLM-only versus gated:

1. **LLM calls per completed objective** — the headline.
2. **Task success rate** — must not regress. A cheap gate that quietly fails tasks
   is not a saving.
3. **Escalation rate** — the share of decisions that reached the frontier model.
   This is the one that says whether the split is real.

The reference point for the frontier model is the published *beat-the-game* run:
`claude-opus-5[1m]`, effort `high`, seed `97996358`, 1,121 transcript rows. Reusing
that seed and prompt makes the comparison direct rather than anecdotal.

## 8. Sources

- [mine-ai-mcp tools](https://github.com/thtesche/mine-ai-mcp/blob/main/docs/mcp/tools.md) —
  every tool's arguments and verification evidence
- [Async actions](https://github.com/thtesche/mine-ai-mcp/blob/main/docs/mcp/async-actions.md) —
  the submit/wait/retrieve contract this design implements
- [Limitations](https://github.com/thtesche/mine-ai-mcp/blob/main/docs/mcp/limitations.md) —
  what the server deliberately does not do
- [Bot data](https://github.com/thtesche/mine-ai-mcp/blob/main/docs/mcp/bot-data.md) —
  SQLite tables and query bounds behind `StateReader`
- [Laya LangChain/LangGraph guide](https://nandhakishorm.github.io/laya/langchain/) —
  `LayaRouter`, `LayaDecision`, per-call budgets, hooks, MPS concurrency
- [Laya model card](https://github.com/NandhaKishorM/laya) — checkpoints, latency,
  long-context accuracy, calibration, honest limits
- [ai-minebot `docs/PLAN.md`](https://github.com/thtesche/ai-minebot) — measured
  latency on this hardware, the executor's failure modes, the Phase 3 gate
- [beat-the-game dataset](https://huggingface.co/datasets/aibengineering/beat-the-game-minecraft) —
  the frontier-model baseline