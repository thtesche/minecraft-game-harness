# minecraft-game-harness

A standalone application that plays Minecraft through
[mine-ai-mcp](https://github.com/thtesche/mine-ai-mcp), with
[Laya](https://nandhakishorm.github.io/laya/) gating the decisions so that a frontier
LLM is called far less often.

> **Status: Phase 0 complete, verified live.** The skeleton connects, reads state, runs
> one objective end to end, and records a ledger row — against a real Minecraft 1.21.4
> world, not only a stand-in. That run found five defects the whole suite had passed; see
> [Running it](#running-it) and [Roadmap](#roadmap).

## The problem

mine-ai-mcp already beats Minecraft, but it does so with a frontier model driving every
step. The published [*beat the game* run](https://huggingface.co/datasets/aibengineering/beat-the-game-minecraft)
used `claude-opus-5[1m]` at effort `high` — 1,121 transcript rows for one playthrough.

That cost is not evenly spread. In a survival run most decisions are repetitive and
low-stakes: collect the next log, walk to the next ore, keep smelting. A frontier model
is the expensive way to answer those. The minority that genuinely needs deliberation —
which research target now, whether a structure is reachable, how to sequence a
progression — is where the model earns its cost.

**The lever is the ratio.** Cheap, correct handling of the repetitive majority lowers
cost per objective without touching quality where quality matters.

## The approach

mine-ai-mcp publishes 34 high-level objective tools that verify their own outcomes, and
deliberately refuses to plan: the bot stands still between actions until the caller
issues the next command. So the missing piece is exactly one thing — **the choice of
which objective to submit next.**

```
                       ┌──────────────────────────────────────┐
   mine-ai-mcp ───────▶│  harness                             │
   (Bun, Streamable    │                                      │
    HTTP /mcp)         │   1. read state      (SQL + views)   │
                       │   2. decide         (Laya, ~70 ms)   │──▶ high
   Laya                │      ├─ confident ──▶ submit tool    │    confidence
   (in-process,        │      └─ unsure ─────▶ LLM node       │──▶ low
    English ckpt,      │                     (frontier)      │
    MPS/CPU)           │   3. wait for settled evidence       │
                       │   4. log (state, question, answer)   │
                       └──────────────────────────────────────┘
```

Laya answers three small questions per objective — which category next, is the current
one finished, escalate or not — and a frontier LLM is reached only for the minority
that warrants deliberation. Every decision and every settled outcome is logged, because
that log is the only honest source for the baseline and for the model's own evaluation.

## Documentation

| Document | Contents |
|---|---|
| [docs/harness_idea.md](docs/harness_idea.md) | The idea, the eight decisions it rests on, and the figures an earlier draft got wrong |
| [docs/architecture.md](docs/architecture.md) | Components, protocol state machine, failure handling, configuration, build order |

## Roadmap

Each phase has an exit criterion, and a phase that cannot meet its criterion does not
proceed — that is the difference between a measurement and an assumption.

| Phase | Deliverable | Exit criterion | State |
|---|---|---|---|
| 0 — Skeleton | Connect, read status, run one objective, write a ledger row | One objective completes unattended, one ledger row exists | **done, live-verified** |
| 1 — Runner | The submit → wait → retrieve protocol state machine | A scripted multi-objective run, every row with verified evidence | protocol core in `objective.py`; loop not built |
| 2 — Baseline | LLM on every decision, on a named scenario set | A measured calls-per-objective number | next |
| 3 — Eval gate | ≥ 200 labelled decisions, `laya-evals`, temperature refit | A documented planner/gatekeeper/fine-tune decision | |
| 4 — Gated loop | Laya wired in at the decision point | Fewer LLM calls per objective, success rate not regressed | |
| 5 — Fine-tune | Conditional on Phase 3 | Held-out calibration, not training-split accuracy | |

Phase 1 is the harness proper. Everything after it is an optimisation layer on top;
without it there is nothing for a model to sit on.

## Running it

```bash
uv venv && uv pip install -e ".[dev]"
cp config.example.json harness.config.json   # then point mcp.url at your host

harness health                              # /health plus the published tool count
harness state                               # the derived decision vector
harness run-one collect_block --arguments '{"block_name": "dirt"}'
harness ledger --counts
```

Start mine-ai-mcp first; `harness health` is the check that the host is reachable.

```sh
cd ../mine-ai-mcp && bun install          # a fresh checkout has no node_modules
bun src/server/host.ts --minecraft-host 127.0.0.1 --minecraft-port 25565 \
  --username MineAI --version 1.21.4
```

`harness state` exits non-zero when the vector is not trustworthy. That is deliberate: a
section the contract promises but the reading did not find is reported, never replaced
with a zero. An invented number is invisible, an escalation is not.

### Two reply shapes, and why the live leg mattered

The server answers in two shapes within one session: foreground tools return the
submission envelope (`{state, actionId, output}`) and information tools return their
result directly (`{action, durationMs, result, survival, survivalPolicy}`, no `state`).
`view_status` is a direct tool. A client that assumes one shape reads nothing and reports
an empty world as fact, so `ToolReply` resolves the result from either and insists on the
envelope only where the protocol is required.

The live host also advertises its 37 tools in a single 2.91 MiB server-sent event, which
is over httpx2's 1 MiB ceiling — the SDK offers no way to raise it, and the resulting
error reports itself as a dead socket. `src/harness/sse.py` is the one module that reaches
into the SDK to bound it, and `harness health` reports whether that patch is in effect.

### Tests

```bash
.venv/bin/python -m pytest
```

The protocol rules are tested twice over. Unit tests drive a scripted client, which
proves the runner's logic. Integration tests run an actual MCP server over Streamable
HTTP that reimplements the submission protocol from the server's contract — unique
`submission_id`, `SUBMISSION_CONFLICT` on a changed retry, `RESULT_NOT_RETRIEVED` on an
owed result, `pending` on an expired wait — and drive the real client against it. That
second layer exists because the client's SDK call signature and the `structuredContent`
envelope are exactly the kind of assumption that passes every unit test and fails at
three in the morning against a live world.

It is a stand-in for the transport and the protocol, not for Minecraft. Nothing in it
reports a physical world outcome that a test then believes.

Worth stating plainly, because the live run proved it the hard way: **a stand-in has to
be pinned to the server's contract, never to the client's assumptions.** This file once
modelled `view_status` with the submission envelope and `tools` as `{"best": {...}}` —
neither of which the real host uses. The client and the fixture therefore agreed, every
test passed, and `harness state` was broken live in two independent ways. A fake that
encodes what the code under test believes amplifies exactly the bug it exists to catch.

## A note on the numbers

Figures in these documents were checked against the Laya model card and the mine-ai-mcp
tool contracts rather than taken from a marketing summary. The ones that moved:

- **33 ms** is a Tesla T4 reference for a short state. Measured on Apple hardware: 39.3 ms
  median on MPS, 87.6 ms on CPU. A 4,000-token state costs **~1.7 s** on an Apple GPU.
- **`min_confidence=0.85`** was an invented constant. Both Laya checkpoints ship
  over-confident and `laya-multilingual` ships no fitted temperatures at all, so the
  threshold has to be fitted from measured accuracy at a chosen coverage.
- **8192-token states** are a real capability, but Laya's own benchmark puts multilingual
  accuracy at 16–18/20 up to ~4,000 tokens and 8–17/20 beyond.
- **Routing over the tool list** is Laya's measured collapse case: at 48 options the
  default head budget scores **1/48**, because similar labels get trimmed until they reach
  the model as the same text.

[harness_idea.md §5](docs/harness_idea.md#5-what-this-document-corrects) records all of
them, including why each looked reasonable in the first place.

## Related work

- [mine-ai-mcp](https://github.com/thtesche/mine-ai-mcp) — the MCP server providing tools
  and SQLite state
- [ai-minebot](https://github.com/thtesche/ai-minebot) — the predecessor project; its
  `docs/PLAN.md` carries the measured Laya latency on this hardware and the executor
  failure modes reused here
- [Laya](https://nandhakishorm.github.io/laya/) — the decision model, with a
  [LangChain/LangGraph integration](https://nandhakishorm.github.io/laya/langchain/)

## Licence

[MIT](LICENSE)