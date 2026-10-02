# minecraft-game-harness

A standalone application that plays Minecraft through
[mine-ai-mcp](https://github.com/thtesche/mine-ai-mcp), with
[Laya](https://nandhakishorm.github.io/laya/) gating the decisions so that a frontier
LLM is called far less often.

> **Status: design complete, implementation not started.** The repository currently
> holds the design documents only. See [Roadmap](#roadmap).

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

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0 — Skeleton | Connect, read status, run one objective, write a ledger row | One objective completes unattended, one ledger row exists |
| 1 — Runner | The submit → wait → retrieve protocol state machine | A scripted multi-objective run, every row with verified evidence |
| 2 — Baseline | LLM on every decision, on a named scenario set | A measured calls-per-objective number |
| 3 — Eval gate | ≥ 200 labelled decisions, `laya-evals`, temperature refit | A documented planner/gatekeeper/fine-tune decision |
| 4 — Gated loop | Laya wired in at the decision point | Fewer LLM calls per objective, success rate not regressed |
| 5 — Fine-tune | Conditional on Phase 3 | Held-out calibration, not training-split accuracy |

Phase 1 is the harness proper. Everything after it is an optimisation layer on top;
without it there is nothing for a model to sit on.

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