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

Phase 1 slice one is **live-verified** at `35302ae`; the Phase 2 model decider is
live-verified at `df7af20`. The **goal list and scenario checker** are done and
pushed at `7ac306a`/`dd2188a` — the mechanism, not a passing run. 233 tests pass.

```
$ .venv/bin/python -m pytest -q
233 passed in 4.20s
```

Against a real Minecraft 1.21.4 host: `health` → 37 tools · `state` →
trustworthy, `unverified: []` · `run-loop` with two objectives → both
sequenced, `plan_exhausted`, exit 0:

```
[0] collect_block: settled:succeeded evidence=ok 3.5s
[1] collect_block: failed:failed       evidence=ok 0.0s
```

Both ledger rows carry a **unique** `state_hash`, `source: scripted`,
`confidence: null`, and the full vector. The hashes differ because the
inventory did, which is the evidence that the bins are not so coarse that
distinct states collapse together.

`mcp` SDK is **2.2.0**. Python 3.12 in `.venv` (uv-managed).

## The lesson this session cost

**A stand-in must be pinned to the server's contract, never to the client's
assumptions.** `tests/fake_host.py` modelled `view_status` with the submission
envelope and `tools` as `{"best": {...}}` — the two shapes the real host does
*not* use. So the client and the fixture agreed, 50 tests passed, and
`harness state` was broken live in two independent ways. A fake that encodes what
the code under test happens to believe amplifies exactly the bug it exists to
catch.

**A second one, from mine-ai-mcp:** `tools/list` was 3,054,268 bytes because
Zod's `toJSONSchema` defaults `reused` to `"inline"` — a shared schema used in
two places is written out twice. `wait_for_action` referenced the foreground
output union from *both* its `settled` and `storage_failed` branches (394 KB
twice), and every checkpoint action's request variant was anonymous (one 22 KB
block inlined eight times). Fixed in mine-ai-mcp at `a27d93a` by naming the
schemas — `3,054,268 → 2,485,102`, `wait_for_action 913,523 → 444,049`. The
client-side ceiling was correct as a stopgap but treated only the symptom.

Five defects surfaced only against the live world; all five passed the suite:

| | Defect |
|---|---|
| 1 | `tools/list` is **2.91 MiB in one SSE event**; httpx2 caps at 1 MiB, the SDK hardcodes `EventSource(response)`, and swallows the error into a dead-socket message |
| 2 | Two reply shapes in one session — 27 enveloped, 10 direct (`view_status` direct). One assumption → empty world reported as fact |
| 3 | `situation.tools` is **one row per class**, not `{"best": …}`; live `tools.best` is null → crash |
| 4 | `fake_host.py` encoded the client's assumptions, not the contract |
| 5 | `ACTION_BUSY` with no action id (a survival reflex) was treated as terminal |

**A third, and the sharpest one yet: a check that stopped running is
indistinguishable from a check that passes.** `ScriptedDecider`'s guard against a
mistyped argument name read `tool["inputSchema"]`. The MCP specification says
`inputSchema`; mine-ai-mcp publishes `input_schema`, and `list_tools` dumps the
SDK model without `by_alias`, so snake_case is what every consumer here has ever
seen. The guard therefore **never ran against the live host** — verified before
changing anything: `block_typo` was accepted on `collect_block`, and the server
*drops* an unrecognised argument rather than refusing it, so the objective ran
against the wrong thing and reported success.

Two unit tests kept it green because both **hand-wrote** the tool table with
`inputSchema`. That is the same mistake as #4 in miniature: a fixture states what
the author believes the server sends. The replacement reads an advertisement off
a real MCP server, so it cannot drift again. **Write fixtures from the wire, not
from memory** — and when a check has never been observed failing, assume it is
not running until you have seen it fail.

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
- `tools/list` = 3,054,268 bytes → **2,485,102** after mine-ai-mcp `a27d93a`.
  `wait_for_action` outputSchema 913 KB → 444 KB.
- **`lastDeath`** is a top-level `situation` key, present only after a death:
  `{dimension, position, observedAt, cause}`. Absent on a fresh world, and
  **not** a missing section — requiring it would report every new session
  unverified. The live bot died twice this session (Zombie 12:30:36, Skeleton
  14:28:19).
- **The input side of `tools/list` is tiny.** All 37 tools' name + description
  + `input_schema` = **28,799 bytes** (~7.2K tokens), against a 2.4 MB total that
  is almost entirely output schemas. A decision needs only the input side, so
  the whole catalogue ships on every call with no filtering heuristic: **16,767
  prompt tokens** measured live. `submission_id` is stripped from it (D15).
- **The advertised schema key is `input_schema`,** not the specified
  `inputSchema` — and `McpClient.list_tools` dumps the SDK model without
  `by_alias`, so snake_case is what every consumer here sees. See the lesson
  below; this already cost one silently-disabled guard.
- **`nvidia/nemotron-3-ultra-550b-a55b:free` behaviour, measured.** 262K context,
  advertises `response_format` and `structured_outputs`, reports `cost: 0`.
  **It ignores an advisory schema and needs `strict: true`** — the first live
  call returned chain-of-thought in `content` with the constraint silently
  dropped. It **truncates**: ~300–430 reasoning tokens come before the answer, so
  a 1024-token ceiling cut the object off mid-brace with
  `finish_reason: "length"`; the ceiling is now 4096. It sometimes returns the
  object **wrapped in a JSON string**, which parses cleanly and then fails every
  field lookup. And its free provider is **overloaded a lot** — three of four
  probes returned 503 — so retryable failures retry and the attempt count is
  recorded. Latency **20–37 s per decision**; that is the cost of this baseline.
- **Remaining advertisement bloat** is not inlining. 38% of what is left
  (1,016,964 B) is four identical shared definitions repeated into 36 of 37
  tool schemas — `MineAiSurvivalPolicySnapshot` alone is 533,880 B. `$defs`
  cannot span documents, and each tool's schema is its own document.
  `wait_for_action` also republishes a union each of the 27 foreground tools
  already advertises verbatim. Getting under 1 MiB means removing published
  information — an API decision for that repo's owner, not a mechanical fix.

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
| D11 | **The loop is deterministic; the decider is not.** Every protocol rule lives in code and the model is consulted only for *which* objective to submit. Swapping a model in changes `decide.py` and nothing else. |
| D12 | **An escalation stops the run.** `min_confidence` is unset, so per `LayaConfig`'s own docstring the honest answer is "escalate rather than guess" — and with no model to escalate *to*, downgrading it into a guess is the exact failure the threshold exists to prevent. An escalation is recorded as a full row, because it is an answer. |
| D13 | **`state_hash` is quantised, and the claim it makes is deliberately weak.** Hashed verbatim it never repeats — two reads a second apart differ in position — so a Memo cache keyed on it records one hit in a thousand rows and is indistinguishable from a model ignoring it. Vitals bin to a whole point, position to half a chunk, distances to a block; discrete facts hash exactly. Two states with the same hash are identical *in every respect a decision could turn on*, which is the property a cache needs and a stronger claim than this can honestly make. |
| D14 | **A death is not a stop condition.** The bot respawns and remaining objectives are still answerable, so the loop absorbs it, records it against the objective it interrupted, and continues. It costs one extra state read, and only after an objective that did *not* succeed — death cannot make a succeeded objective not have succeeded. |
| D15 | **`SUBMISSION_CONFLICT` has no recovery, on purpose.** The refusal names no action, so the earlier submission holding that id cannot be found from it, and that work may still be running. The only defence is making the collision impossible; `mcp.submission_prefix` is for attribution, so an id seen twice is identifiable as ours. |
| D16 | **The model decider is not gated, and records `confidence: null`.** `confidence_gate` is a Laya policy about a classifier with fitted temperatures. A frontier model asked for a JSON objective has no such number. A fixed `1.0` was proposed to unblock the loop and is refused twice over: the harness would be inventing a certainty it has no evidence for, and it would pass *any* threshold fitted in Phase 3, so the calibration mechanism would go in green reporting that calibration happened when nothing was measured. Not applying the gate gets the loop running exactly as asked, with `None` in the ledger meaning what it means. |
| D17 | **The model chooses from the 27 tools that submit work, not all 37.** The ten reads and controls are excluded by a list declared in `llm.py`: an advertisement says nothing about being an objective, and the server's tier enums are not a ranking of these (D13). Default is *open*, so a tool the server adds later works without a change here. |
| D18 | **`craft_item` is the decomposition engine, so the harness holds no recipe knowledge and calls no planner.** Measured: the reference run called `view_crafting_requirements` **zero** times in 388 calls. It asked for items by name and read the recipe tree, leaf materials and workstation requirement out of the `craft_item` *result* — 23 calls, 27 distinct items, 66 units. An earlier plan to call the planner once per goal is therefore dropped: it would pay for information the craft reply already carries. A goal is an **item name**, not a procedure. |
| D19 | **A scenario is a prompt, a world, and a checker; without the third it is a demo.** The checker is written *before* the run. "It reached a wooden pickaxe" is a fact; "the run looked reasonable" is not. This is the same line `docs/harness_idea.md:152` draws for phases, applied to individual runs. |
| D20 | **The reference run's per-call cost is not a target and its $0.240 is not comparable to ours.** Opus list price against a `cost: 0` free model compares two different things; only token counts transfer. Its 388 calls are a *ceiling*, not a target. Recorded because the temptation to quote the cheaper number is exactly what makes a baseline unreadable. |
| D21 | **The goal list is not pre-ordered.** `GoalSet` accepts a set and refuses to act on an order it was not given, because a harness that sorts the goals has written the plan itself and the decider's ordering mistakes become invisible. Ordering is the measurement. |
| D22 | **Checkers read the world themselves, not the loop's report.** The loop's `StateVector` is deliberately lossy — twelve stacks, to keep the prompt small — so a checker reading it would report "no pickaxe" for a bot holding one in slot twenty, and that failure is indistinguishable from the bot failing. |
| D23 | **An unknown check kind is refused at load time, never evaluated as true.** A scenario naming a check that does not exist must not run green; a scenario that cannot fail looks exactly like one that passed. The refusal lists the kinds that do exist. |
| D24 | **The loop refuses two kinds of off-goal objective, and both stop the run.** `goal_unknown` when the decider names no goal at all or one outside the set — counting progress toward a goal nobody asked for corrupts the measurement the scenario exists to make. `goal_attempts_exhausted` when past `budget.max_attempts_per_goal` — spending a real objective, and 20–37 s of model, on work the harness has already declared it will stop doing. Both in `INCOMPLETE_STOPS`, so neither reads as success. |
| D25 | **The per-goal cap counts attempts, not failures.** A long recipe is not a stuck loop; eight failed attempts at one goal and eight attempts at a goal that needs six steps are different things, and only the second is a cap. |
| D26 | **A run-shaped check refuses to pass on a run that made no call.** `no_death`, `within_calls` and `evidence_verified` are universals over the set of steps, and over an empty set all three are vacuously true. Found by running scenario 1 live against a disconnected bot: the loop stopped correctly and two checks still printed green for a run in which the bot never moved. A check that passes on nothing has not passed. |
| D27 | **Unreadable and unrankable are different, and the two consumers disagree on purpose.** `StateVector.unverified` holds both, because a decider that cannot rank what it is holding should be told so. `WorldFacts.unverified` holds only sections the harness could not parse; an equipment tier outside the harness order is a *warning*, because "is the best tool at least stone" is still answerable when the bot also holds a trident. Folding the two together would fail a passing run for a reason that is not the reason. |

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
| `src/harness/mcp_client.py` | Streamable HTTP, forces `response_format: json`, builds its own timeouts. `input_schema_of` / `argument_names` read the advertised schema under either spelling — see the lesson below. |
| `src/harness/objective.py` | The protocol state machine. |
| `src/harness/state.py` | `view_status` → compact decision vector. Missing sections land in `unverified`, never as zeros. |
| `src/harness/ledger.py` | JSONL + SQLite, one row per decision, flushed per row. |
| `src/harness/loop.py` | The deterministic loop: read a trusted state, ask the decider, run, record. Three stop conditions (plan exhausted / escalated / unreadable); a death is not one. |
| `src/harness/decide.py` | The model seam. `Proposal | Escalation | None` — three answers, and the third is the one that gets forgotten. `ScriptedDecider` is the Phase 1 stand-in and marks every row `source: "scripted"`. |
| `src/harness/llm.py` | `OpenRouterDecider`: the Phase 2 baseline. Every decision is one HTTP call. Validates the model's tool and arguments against the advertisement, records cost, retries only what is worth retrying, and deliberately does **not** apply `confidence_gate` (D16). Also `load_dotenv`. |
| `src/harness/errors.py` | `UnverifiedRead` / `ProtocolError` / `ObjectiveFailed` / `BudgetExceeded`. |
| `src/harness/cli.py` | `health`, `state`, `run-one`, `run-loop` (`--decider script\|llm`), `ledger`. |
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
- `tests/test_loop.py` — the loop and the decider seam. Weighted towards what the loop
  *refuses*: an unreadable state is re-read once then the run ends, an escalation is
  never re-asked until it becomes a guess, a death is not a stop condition, the same
  death timestamp is not counted twice
- `tests/fake_host.py` — a real MCP server over Streamable HTTP reimplementing the
  submission protocol from the contract
- `tests/test_live_client.py` — the real client against that stand-in
- `tests/test_llm.py` — the model decider, 37 tests, and **37 of them refuse
  something**: no key, no model named, an advertised tool that names no
  arguments, a 401, a 429 that is retried, an overload that never clears, a
  truncated answer, reasoning with no answer, prose, an object wrapped in a JSON
  string, an unadvertised tool, an invented argument name, an omitted required
  argument. No test opens a socket; the transport is injected.

The stand-in reproduces the **transport and protocol only**. It reports no physical
world outcome that a test then believes. It is not Minecraft.

`mine-ai-mcp` gained `src/server/advertised-size.test.ts` for the same reason: the
advertisement bloat regresses *silently* — no tool changes behaviour, the payload just
grows. It asserts the size relationship directly rather than re-deriving it from the
code, and asserts that an unnamed union is **still** inlined per use, so if that ever
stops holding, the measurement the fix was justified by is known to be stale rather
than quietly assumed.

## Verified this session

Live, against Minecraft 1.21.4 with the bot joined as `MineAI`: `health` → 37
tools, SSE patch live · `state` → trustworthy, `unverified: []` · `run-loop` (two
objectives, scripted) → both sequenced, `plan_exhausted`, exit 0 · `ledger` →
two rows with unique `state_hash`, `source: scripted`, verified evidence on both.

**`run-loop --decider llm`, end to end.** The model read a live state (health
13.2, food 17.0, `best_tool: null`, carrying `dirtx2`) and chose
`collect_block {block_name: logs, count: 4}` — wood for a pickaxe, which is the
correct first move with nothing held. It settled **succeeded**, evidence
verified, 34.5 s. Ledger row: `source: "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free"`,
`answer_confidence: null`, `escalated: false`, `state_hash: ce8cc645e60ddc46`.
Cost **$0**, 16,767 prompt + 429 completion tokens (315 of them reasoning),
0 retries.

Also verified: the ceiling error path, by setting `max_sse_event_bytes` to
1,000,000 and confirming the failure **names its cause** rather than reporting a
dead socket.

Death absorption is **implemented and unit-tested but not yet seen live** — the
bot died twice during this session's probing (Zombie, then Skeleton) but not
during a `run-loop` step. The mechanism is a comparison of
`situation.lastDeath.observedAt` across a step, so a re-death is not
double-counted.

## Reference run, measured 2026-10-02

Full extraction, with its validation shown, in **`docs/reference_run.md`**. Dataset
`aibengineering/beat-the-game-minecraft`; the three files used were SHA256-verified
against the published `SHA256SUMS`. `claude-opus-5[1m]` at effort high, zero subagents,
fresh spawn → confirmed Ender Dragon kill.

| | |
|---|---|
| Tool calls / turns | **388** / 389 |
| Wall clock | 3:56:35 |
| Cost | **$93.17** = **$0.240 per call** |
| Cache-read input | 163,325,079 = **99.4% of all input**, 420,941 per call |
| Context growth | 37,417 (turn 1) → 779,049 (turn 385), mean 426,246 |
| Output / thinking | 148,383 (382 per call) / 67,071 |
| Tools used | **34 of 37** — never called: `barter`, `send_message`, `view_crafting_requirements` |
| `wait_for_action` | **59 calls, 15.2% of the run** |
| Reads vs action | 44 vs 344 |

**The comparison number for scenario 1: four calls from an empty inventory to a held
wooden pickaxe**, two of them reads — `view_status`, `note_read`, `collect_block logs×16`,
`craft_item[crafting_table, wooden_pickaxe, wooden_axe, stick×8]`. Stone tier including a
furnace is call six.

The structural claim, now with a number: the reference re-reads a growing transcript and
pays 420,941 input tokens per call on average. Our harness reads a state vector and paid
**16,767** prompt tokens for its one live decision — 25× fewer, and flat rather than
compounding.

Two extractions nearly produced false findings and both are recorded in the doc: a
substring search for `"wooden_pickaxe"` reported the item was *never acquired* when it was
acquired on call 4 (results are markdown, not JSON), and an argument map keyed by tool name
showed every `craft_item` asking for `shears`. A check that finds nothing is not evidence of
absence, and a check that stops running is indistinguishable from one that passes.

## Scenario 1, first live attempt 2026-10-02 — refused, and that is the result

`run-scenario scenarios/first-pickaxe.json` against the live host produced:

```
scenario: first-pickaxe  ->  FAIL
  [FAIL] world: the world could not be read after the run
  [FAIL] goal_succeeded: the run recorded no goal attempts
  [FAIL] evidence_verified: the run made no tool call, so evidence_verified has nothing to measure
  [FAIL] within_calls: the run made no tool call, so within_calls has nothing to measure
  loop stopped: unverified_state: the world could not be read after 2 attempt(s)
```

**Why:** the bot's Minecraft connection ended. `minecraft.connected: false`,
`vitals: null`, health 5.4, food 5, holding 4 dirt + 1 oak_log + 3 acacia_log.
The MCP host itself is alive (PID 15601) and the Minecraft server is up (25565
accepts); only the bot's socket closed. `mine-ai-mcp/src/server/runtime-host.ts:143`
says so in its own comment: *"A new Minecraft connection requires an explicit
service restart."* There is no reconnect path in the server and none in the
harness, so this is the `RUNTIME_UNAVAILABLE` recovery from Phase 1 slice two,
still unbuilt. Restarting the host is the operator's call.

Two things worth keeping from the failure:

1. **The read refused rather than defaulted.** No situation was found in either
   reply shape and the run stopped instead of submitting into whatever the server
   had. The world checks then failed saying the read failed — never passed on
   absent evidence.
2. **Two checks passed vacuously and were caught by looking.** `evidence_verified
   0/0` and `within_calls 0 of 8` printed green for a run in which the bot never
   moved. The overall verdict was already correct, because `ScenarioReport.passed`
   also requires a clean loop — but a reader scanning the report saw green under a
   dead run. Fixed as D26.

**Two real bugs came out of writing the fixture from the wire** rather than from
memory, and the wire is `mine-ai-mcp/src/actions/view-status/contract.ts`
(`inventoryStackSchema`, `toolSnapshotEntrySchema`, `clock`, `lastDeath`):

- The tools rows were keyed by `row["item"] or row["tier"]`, which produces keys
  like `"none"` and `"wooden_pickaxe"` for a row the server keys by `class`.
  `world_facts` now calls `state.best_equipment()`, so the prompt and the checker
  cannot disagree about which tool is best — a disagreement that would be
  indistinguishable from the bot not holding one.
- A reply with **no `inventory` key at all** read as an empty inventory, because
  only the shape was checked, not the presence. Now `unverified`, so a broken read
  cannot become a red run that looks like a bot that failed.

## Not yet verified

- **Scenario 1 passing, live.** Never run against a bot that was connected. The
  target is ≤8 objective submissions to hold `wooden_pickaxe`; the reference did it
  in 2 submissions / 4 calls. Watch for the model re-proposing `collect_block logs`,
  which it did three times in Phase 1.
- **Death absorption, live.** Tested against the fake only. The `lastDeath` read
  and the cross-step comparison both work against the real payload (the Skeleton
  death at 14:28:19 is in the live vector), but no `run-loop` step has yet
  absorbed one. Scenario 2 (survive one night) is what will finally exercise it.
- **Long objectives.** Only `collect_block` has run, and it settles in seconds. A
  smelt is the reason `tool_timeout_ms` is an hour and nothing has yet exercised
  it.
- **A real decider.** `min_confidence` is unset, so every *Laya*-backed decision
  would escalate and stop the run. There is no Laya client in `src/` at all —
  `LayaConfig` is configuration and nothing imports it. `LlmConfig` is now live
  through `llm.py`; see D16 for why that path is deliberately not gated.

Still unbuilt, part of Phase 1 per `docs/architecture.md` §2.4 and §6:

- a Laya-backed decider (the `Decider` seam is done and filled by the model one)
- cancellation
- reconnection (re-read `/health.foreground`, reconnect) — `RUNTIME_UNAVAILABLE`
  currently stops the run
- `SUBMISSION_CONFLICT` recovery — deliberately **not** built; see D15

## Next move

0. **Blocked on an operator action: restart the mine-ai-mcp host.** The bot's
   Minecraft connection ended and the server will not reconnect on its own. Until
   it is restarted no scenario can run, and a run attempted against a disconnected
   world measures the disconnection. Note that restarting does *not* restore
   "from nothing" — the bot rejoins holding 4 dirt and 4 logs — so scenario 1's
   premise needs either accepting that starting state (recorded in the scenario's
   `reference` block) or a fresh world with a named seed. **The user's call, not
   mine: it is their service and their world.**
1. **Scenario 1 — reach a wooden pickaxe from nothing — plus the checker mechanism.**
   Mechanism done and pushed (`7ac306a`, `dd2188a`); the run is what is missing.
   This is the make-or-break for goal-list mode: if the model cannot sequence
   logs → crafting table → wooden pickaxe from an unordered goal list, the rest is
   wasted effort. Its baseline is **four calls**, of which two are reads. A scenario
   is prompt + world + checker (D19); the checker is written first. Note D18: the
   goal is an item name and `craft_item` does the decomposition, so there is no
   planner call and no recipe knowledge in the harness.
2. **Cancellation and reconnection.** The decider seam is filled — `ScriptedDecider`
   and `OpenRouterDecider` both work — so what is left of Phase 1 slice two is
   `cancel_foreground_action` and `RUNTIME_UNAVAILABLE` recovery (re-read
   `/health.foreground`, reconnect). Note the baseline's shape when judging them:
   **20–37 s per decision**, so a cancelled objective wastes real money, not just
   time. The reference run's shape says the same thing louder: **59 of its 388 calls
   were `wait_for_action`**, 15.2% of a $93 run spent polling.
3. **Then: survive one night** (time-bounded, cheap, and it finally exercises **death
   absorption live**, unverified across three sessions), and **stone tier** (both its
   items come back `missing_materials` naming `cobbled_deepslatex`).
4. **Resolve duplication against `laya-mine`** before building further — it already has a
   reflex dataset builder (`build_reflex_dataset.py`, 293 labelled rows in `reflex.jsonl`)
   and baseline measurements
   (`measurements/baseline-original-super120b.json`). Its rows predict *which survival
   directive the reflex should run*; the reflex is already deterministic server-side and
   needs no gating, so those rows answer a **different question** than the harness does.
   Reuse them as a baseline, or treat as a separate experiment — open question.
5. **The remaining advertisement bloat in mine-ai-mcp** is an API decision, not a
   mechanical fix: 38% of what is left is four shared definitions copied into 36 of
   37 tool schemas, and `definitions` cannot span documents. `a27d93a`'s commit
   message carries the numbers so they need not be re-derived. Relevant to us now that
   it is quantified how much of a prompt is dead weight: **3 of 37 tools were never called
   once in 388 calls**, and their schemas were in every one of those contexts.
6. **A Laya-backed decider**, when the checkpoint is available. `min_confidence`
   still needs the Phase 3 eval before it can be set — and per D16 the gate will
   escalate every decision until it is.

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

`run-loop --decider llm` needs a key: `cp .env.example .env`, fill in
`OPENROUTER_API_KEY`, and export it. `.env` is gitignored, `.env.example` is
tracked, and `load_dotenv` never overwrites a variable already in the
environment. The key goes in the `Authorization` header and nowhere else — not
the config file, not the ledger, not an error message.

```sh
.venv/bin/python -m harness.cli --config harness.config.json run-loop \
  --decider llm --max-steps 8
```

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