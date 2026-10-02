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

Phase 0 committed and pushed at `1d3fe08`. 50 tests pass. Working tree was clean.

```
$ .venv/bin/python -m pytest -q
50 passed in 2.93s
```

`mcp` SDK is **2.2.0**. Python 3.12 in `.venv` (uv-managed).

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

`StateVector` is deliberately small — a 4000-token state costs ~1.7 s on an Apple GPU and
accuracy degrades past ~4k tokens. Shrinking the state is a design fix, not a bigger
`max_len`.

## Three SDK bugs the integration tests caught

Worth remembering because unit tests passed on all three and each would have failed
against a live host at 3am:

1. `streamable_http_client()` takes **no `timeout` kwarg** — it takes `http_client`. The
   transport returns a **2-tuple** `(read, write)`, not a 3-tuple.
2. The SDK exposes `structuredContent` as **`structured_content`**. The inner payload
   keeps the server's own camelCase spelling.
3. mcp 2.x has **no `FastMCP`** — it is `mcp.server.mcpserver.MCPServer`.

Default SDK read timeout is **300 s**, shorter than one stack smelt, so the client
builds its own via `create_mcp_http_client(timeout=...)`.

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

CLI end to end against the fake host: `health` → tool count 3; `state` → trustworthy
vector; `run-one collect_block` → `settled:succeeded`, `evidenceOk: true`, `polls: 0`;
`ledger --counts` → one row.

## Not yet verified

**Phase 0's live leg is unproven.** The exit criterion "one objective completes
unattended, one ledger row exists" has only been met against the stand-in. No
mine-ai-mcp host was running (`curl localhost:25575/health` empty). `mine-ai-mcp` is
present and runnable at `/Users/thtesche/VibeCoding/mine-ai-mcp`.

Also unbuilt, and part of Phase 1's scope per `docs/architecture.md` §2.4 and §6:

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

1. **Verify Phase 0 live.** Start mine-ai-mcp, run `harness health` → `harness state` →
   `harness run-one` against the real world, confirm a ledger row with verified evidence.
2. **Phase 1.** The runner loop: several objectives in sequence, plus the missing
   failure modes above. Exit criterion is a scripted multi-objective run with every
   ledger row carrying verified evidence.
3. **Resolve duplication against `laya-mine`** before building further — it already has a
   reflex dataset builder (`build_reflex_dataset.py`, 293 labelled rows in `reflex.jsonl`)
   and baseline measurements
   (`measurements/baseline-original-super120b.json`). Its rows predict *which survival
   directive the reflex should run*; the reflex is already deterministic server-side and
   needs no gating, so those rows answer a **different question** than the harness does.
   Reuse them as a baseline, or treat as a separate experiment — open question.

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