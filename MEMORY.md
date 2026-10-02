# MEMORY

Working memory for `minecraft-game-harness`. Read this first in a new session.

## What this project is

A standalone application that plays Minecraft through
[mine-ai-mcp](https://github.com/thtesche/mine-ai-mcp), with
[Laya](https://nandhakishorm.github.io/laya/) gating the decisions so a frontier
LLM is called far less often.

The lever is the ratio. A survival run is mostly repetitive, low-stakes decisions —
collect the next log, walk to the ore, keep smelting. A frontier model is the expensive
way to answer those. Laya handles the majority; the frontier model is reached only for
the minority that genuinely needs deliberation.

The success metric is three numbers on a fixed scenario set, against the
*beat-the-game* reference (seed `97996358`, `claude-opus-5[1m]`, effort high, 1121
transcript rows):

1. LLM calls per completed objective
2. Task success rate — **must not regress**
3. Escalation rate

## Current state

Phase 0 is **live-verified** at `0d4bca7`. 78 tests pass, working tree clean.

```
$ .venv/bin/python -m pytest -q
78 passed in 3.82s
```

Against a real Minecraft 1.21.4 host: `health` → 37 tools · `state` →
trustworthy, `unverified: []` · `run-one collect_block` → `settled:succeeded`,
`evidenceOk: true`, `polls: 16`, 41.6 s · `ledger` → `settled:succeeded` with a
unique row id. The 41.6 s was spent waiting out a survival reflex that held the
body — see D9.

`mcp` SDK is **2.2.0**. Python 3.12 in `.venv` (uv-managed).

## The lesson this session cost

**A stand-in must be pinned to the server's contract, never to the client's
assumptions.** `tests/fake_host.py` modelled `view_status` with the submission
envelope and `tools` as `{"best": {...}}` — the two shapes the real host does
*not* use. So the client and the fixture agreed, 50 tests passed, and
`harness state` was broken live in two independent ways. A fake that encodes what
the code under test happens to believe amplifies exactly the bug it exists to
catch.

Five defects surfaced only against the live world; all five passed the suite:

| | Defect |
|---|---|
| 1 | `tools/list` is **2.91 MiB in one SSE event**; httpx2 caps at 1 MiB, the SDK hardcodes `EventSource(response)`, and swallows the error into a dead-socket message |
| 2 | Two reply shapes in one session — 27 enveloped, 10 direct (`view_status` direct). One assumption → empty world reported as fact |
| 3 | `situation.tools` is **one row per class**, not `{"best": …}`; live `tools.best` is null → crash |
| 4 | `fake_host.py` encoded the client's assumptions, not the contract |
| 5 | `ACTION_BUSY` with no action id (a survival reflex) was treated as terminal |

## Live host facts, measured 2026-10-02

- **Reply shapes.** Envelope: `{state, actionId, output}`. Direct:
  `{action, durationMs, result, survival, survivalPolicy}`, no `state`.
  27 foreground / 10 direct. `wait_for_action` is a *control* tool that still
  speaks the envelope.
- **`foreground` is not on the wire** — internal to `describe.ts:144`. Only
  `submission_id` + `wait_timeout_ms` in the advertised schema hint at the split.
- **Tool table**: `{"tools": [12 rows], "armour": [4 rows]}`, one row per class
  whether or not held, empty ones carrying `tier: "none"`, `item: null`.
- **Tier vocabulary** (`src/world/tool-tiers.ts:28`): `wooden stone iron golden
  diamond netherite leather chainmail turtle other none`. Enum order is **not** a
  ranking — armour materials sit after netherite, so the harness declares
  Minecraft's own material order and reports anything outside it.
- **Free body** = `activity.owner == "idle"` *and* `activeAction is null`
  (`action-runner.ts`). Owners: `idle | foreground | yielding | takeover`.
- `tools/list` = 3,054,268 bytes. `wait_for_action` outputSchema alone = 913 KB.

## Ground rules for this repo

- **All code, docs and commit messages in English.** There is no exception.
- **Push after every commit.** The user asked for this explicitly.
- MIT licence, `Copyright (c) 2026 Thomas Tesche`, copied verbatim from
  `ai-minebot/LICENSE`.
- Docs live under `docs/`, not the repo root.
- `runs/` is gitignored. Ledger data is measurement substrate — commit it deliberately
  when keeping a run, not while iterating on a smoke test.

## Architecture in one paragraph

mine-ai-mcp publishes 34 objective tools that verify their own outcomes and
deliberately refuse to plan — the bot stands still until the caller issues the next
command. So the only missing piece is **which objective to submit next**. The harness
reads state, asks Laya which objective category fits, submits, waits for settled
evidence, and logs the decision. The submit → wait → retrieve loop is a deterministic
state machine, never a model improvisation.

## Decisions that must not be silently reversed

These came out of analysis against the Laya model card and the mine-ai-mcp tool
contracts. Full reasoning in `docs/harness_idea.md` §5.

| | Decision |
|---|---|
| D1 | Laya gates (≤6 option choices). It never selects among the 34 tools. Above ~20 options labels collapse — 48 options score **1/48** at the default head budget. |
| D2 | Gate on `answer_confidence`, never `action.act_probability`. AUROC 0.77 vs 0.30. |
| D3 | The threshold is **fitted from eval data**. `0.85` was an invented constant. |
| D4 | English checkpoint by default. `laya-multilingual` only against a measured need; it ships no fitted temperatures. |
| D5 | Laya in-process, not over MCP. |
| D6 | Laya runs at **objective** cadence, not tick cadence. The server stands still between actions; latency is not the constraint, **calibration** is. |
| D7 | Deterministic server-side layers (recipe trees, frontier map, mobility) run before any model call. |
| D8 | The long submit→wait→retrieve protocol stays a deterministic loop. The model is consulted at decision points *inside* it. |
| D9 | `ACTION_BUSY` **with no `activeActionId`** is a survival reflex holding the body. Wait it out via `view_status`, bounded by `budget.gate_wait_ms`. Treating it as terminal aborted every objective for as long as a mob lived — most of the night. A status read that cannot be parsed is **not** read as freedom: that submits into a reflex and reports the refusal as an answer. |
| D10 | `harness.sse` is the single sanctioned place that reaches into the SDK. A ceiling default chosen below a *measured* workload is a silent outage, so `MCMEASURED_TOOLS_LIST_BYTES` is a constant and the default is asserted above it in a test. |

## Laya operating limits

- `head_max_len` 192 (`laya`) / 256 (multilingual) — do not raise casually
- 8k context is multilingual-only, costs ~1.7 s on Apple GPU, and accuracy falls to 8–17/20 past ~4k tokens
- Concurrent MPS forwards abort the process (`_status < MTLCommandBufferStatusCommitted`) → `max_concurrency=1` or `batch()`
- Measured latency on this hardware: 39.3 ms median MPS, 87.6 ms CPU
- `Memo` hook caches identical states (4 distinct of 24 states → 4 passes / 330 ms cold, 0 / 0.3 ms warm)
- `laya-evals` exists in 0.3.23 and exits non-zero on threshold violation, so the Phase 3 gate drops into CI unchanged

## Protocol contract (from `mine-ai-mcp/docs/mcp/async-actions.md`)

Encoded in `src/harness/objective.py`. These rules are in **no tool description**, which
is exactly why they are code rather than prompt text.

- Every foreground call needs a unique `submission_id`. Retry the identical args + id to
  recover a lost reply. Change either → `SUBMISSION_CONFLICT`.
- One objective admitted at a time.
- `RESULT_NOT_RETRIEVED` refuses the next submission until retrieval.
- **A client timeout is not cancellation.** The bot keeps working.
- `wait_timeout_ms` / `timeout_ms` bounded 0–120000. Initial wait 5000 ms, follow-ups 0–30 s.
- **Client timeout must be 3600000 ms** — one stack smelt exceeds a 5-minute idle window.
- States: `accepted`, `pending`, `settled`, `refused`, `storage_failed`.
- Refusal codes: `INVALID_ARGUMENTS`, `SUBMISSION_CONFLICT`, `RUNTIME_UNAVAILABLE`,
  `ACTION_BUSY`, `RESULT_NOT_RETRIEVED`, `ADMISSION_STORAGE_FAILED`, `ACTION_NOT_FOUND`.
- Acceptance is not a physical result. `partial`, `failed`, `cancelled` are terminal; a
  missing evidence leg is **unverified**, never success.

## Code layout

| File | Role |
|---|---|
| `src/harness/config.py` | Every bound, each existing because something was measured to run away without it. Unknown config keys **raise** — a typo that is ignored produces a harness that does not do what its author believes. |
| `src/harness/mcp_client.py` | Streamable HTTP, forces `response_format: json`, builds its own timeouts. |
| `src/harness/objective.py` | The protocol state machine. |
| `src/harness/state.py` | `view_status` → compact decision vector. Missing sections land in `unverified`, never as zeros. |
| `src/harness/ledger.py` | JSONL + SQLite, one row per decision, flushed per row. |
| `src/harness/errors.py` | `UnverifiedRead` / `ProtocolError` / `ObjectiveFailed` / `BudgetExceeded`. |
| `src/harness/cli.py` | `health`, `state`, `run-one`, `ledger`. |
| `src/harness/sse.py` | The **only** module that patches the SDK. Raises if the SDK's call site moves; distinguishes "raise the ceiling" from "the patch stopped working". |

`ToolReply` is deliberately **shape-aware**: `is_protocol`, `state` (which may be
`None`), `require_state()`, and a `result` accessor that resolves the result
object from either reply shape. A direct reply and an enveloped reply appear in
the same session, so no call site may pick one.

`tests/fake_host.py` now reproduces the **direct** reply shape, the row-per-class
tool table, and the body-ownership refusal (`hostile.reflex`, `reflex_reads_left`).

`StateVector` is deliberately small — a 4000-token state costs ~1.7 s on an Apple GPU and
accuracy degrades past ~4k tokens. Shrinking the state is a design fix, not a bigger
`max_len`.

## Four SDK/contract bugs the integration tests caught

Worth remembering because unit tests passed on all four and each would have failed
against a live host at 3am:

1. `streamable_http_client()` takes **no `timeout` kwarg** — it takes `http_client`. The
   transport returns a **2-tuple** `(read, write)`, not a 3-tuple.
2. The SDK exposes `structuredContent` as **`structured_content`**. The inner payload
   keeps the server's own camelCase spelling.
3. mcp 2.x has **no `FastMCP`** — it is `mcp.server.mcpserver.MCPServer`.

Default SDK read timeout is **300 s**, shorter than one stack smelt, so the client
builds its own via `create_mcp_http_client(timeout=...)`.

4. `httpx2` caps one SSE event at **1 MiB** and the SDK constructs its parser as
   `EventSource(response)` with no knob. The live host's `tools/list` is 2.91 MiB.
   The SDK's `except Exception` around the event loop swallows httpx2's
   `SSEError`, so the ceiling surfaces as a *dead socket*. Fixed in
   `src/harness/sse.py`.

## One correctness subtlety

`RESULT_NOT_RETRIEVED` only fires for a **new** `submission_id` — an identical retry
short-circuits earlier in the server. So the owed result **always belongs to a different
action**. Retrieving it releases the gate; it does not satisfy the objective being
submitted. The runner drains the blocking action and then submits for real. Returning the
owed action's result would claim a physical outcome the harness never asked for — the
exact failure this project measures around. There is a test pinning this.

## Tests

- `tests/test_objective.py`, `test_state.py`, `test_config.py` — unit, scripted client
- `tests/fake_host.py` — a real MCP server over Streamable HTTP reimplementing the
  submission protocol from the contract
- `tests/test_live_client.py` — the real client against that stand-in

The stand-in reproduces the **transport and protocol only**. It reports no physical
world outcome that a test then believes. It is not Minecraft.

## Verified this session

Live, against Minecraft 1.21.4 with the bot joined as `MineAI`: `health` → 37
tools, SSE patch live · `state` → trustworthy, `unverified: []` · `run-one
collect_block` → `settled:succeeded`, `evidenceOk: true`, `polls: 16` · `ledger`
→ `settled:succeeded`, unique row ids.

Also verified: the ceiling error path, by setting `max_sse_event_bytes` to
1,000,000 and confirming the failure **names its cause** rather than reporting a
dead socket.

## Not yet verified

- **Death absorption.** The live run ended with the bot at 4.5 health under a
  spider. It never died, so `lastDeath` handling is still unexercised. The
  standing `lastDeath` in the live world (`shot by Skeleton`, 2026-09-30) is from
  an earlier session, not this run.
- **Long objectives.** Only `collect_block` has run. A smelt is the reason
  `tool_timeout_ms` is an hour and nothing has yet exercised it.
- **`state_hash` is never populated.** Every ledger row records an empty hash. It
  is the key the `Memo` cache needs, so it must be filled from a real state read
  before Phase 4 — and its absence is why the D10 memo figures cannot be
  reproduced from harness data.

Still unbuilt, part of Phase 1 per `docs/architecture.md` §2.4 and §6:

- cancellation
- reconnection (re-read `/health.foreground`, reconnect)
- death absorption (death mid-objective is an ordinary step outcome: absorb, report,
  continue — never crash the run)
- multiple objectives in sequence — **this is the loop itself, i.e. the bulk of Phase 1**
- `SUBMISSION_CONFLICT` recovery (identifying the runner has, in `objective.py`; the
  state machine does not yet)
- `mcp.submission_prefix` per-process run prefix (documented in architecture §5, not yet
  in `config.py`; `run_id` exists but `submission_id` is still a bare uuid4)

## Next move

1. **Phase 1.** The runner loop: several objectives in sequence, plus the failure modes
   above. Exit criterion is a scripted multi-objective run with every ledger row carrying
   verified evidence. Start by reading the real state vector before each decision and
   **stamping `state_hash`** — the loop is what makes the hash obtainable, and Phase 4's
   memo cache needs it.
2. **Resolve duplication against `laya-mine`** before building further — it already has a
   reflex dataset builder (`build_reflex_dataset.py`, 293 labelled rows in `reflex.jsonl`)
   and baseline measurements
   (`measurements/baseline-original-super120b.json`). Its rows predict *which survival
   directive the reflex should run*; the reflex is already deterministic server-side and
   needs no gating, so those rows answer a **different question** than the harness does.
   Reuse them as a baseline, or treat as a separate experiment — open question.
3. **Fix the tool-advertisement bloat in mine-ai-mcp** (its own repo). `wait_for_action`'s
   913 KB `outputSchema` is `$ref` inlining. Capping at the client is correct but only
   treats the symptom; a 2.91 MiB handshake is a real cost on every session.

## Running a live leg

`mine-ai-mcp` needs `bun install` first (no `node_modules` in a fresh checkout):

```sh
cd /Users/thtesche/VibeCoding/mine-ai-mcp && bun install
bun src/server/host.ts --minecraft-host 127.0.0.1 --minecraft-port 25565 \
  --username MineAI --version 1.21.4
```

Then `harness health` is the reachability check. The host is spawned **per session**
and does not survive a restart; `~/.mine-ai/bot-data` persists the frontier across
runs, so a restart resumes rather than restarting.

## Related files elsewhere

- `/Users/thtesche/VibeCoding/mine-ai-mcp` — the server. `docs/mcp/async-actions.md` is the
  protocol contract.
- `/Users/thtesche/VibeCoding/mine-ai-mcp-local/README-LAYA.md` — quantifies the real LLM
  failure modes: invented `action_id`s, `pending` waits 20% of the time, the gate rule
  missing from schemas, `collect_block` median 50.8 s / max 120 s.
- `/Users/thtesche/VibeCoding/mine-ai-mcp-local/run-patched.sh` — A/B switch
  `original|laya` via `MINE_AI_TOOL_DESCRIPTIONS`.
- `/Users/thtesche/VibeCoding/ai-minebot/docs/PLAN.md` — measured Laya latency and
  executor failure modes. Phase 3 eval gate: ≥0.8 deploy / 0.6–0.8 gate / <0.6 fine-tune.
- `/Users/thtesche/VibeCoding/ai-minebot/.venv` — separate env, `laya` 0.3.23.
- `/Users/thtesche/VibeCoding/laya-mine` — the duplication risk above.