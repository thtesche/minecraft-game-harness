# Harness Idea: Mine AI MCP with Laya-Gated LLM Calls

**Status:** design under review · **Repo:** `minecraft-game-harness`
**Purpose:** use [mine-ai-mcp](https://github.com/thtesche/mine-ai-mcp) and its tools as
the execution layer, and cut the number of frontier-LLM calls per Minecraft objective
dramatically by putting a small, fast decision model ([Laya](https://nandhakishorm.github.io/laya/))
in front of it.

This document states the idea and the decisions it rests on.
[architecture.md](architecture.md) holds the design that follows from it.

> **This document supersedes an earlier, thinner draft.** That draft proposed the
> reflex/deliberation split with `min_confidence=0.85` and an 8192-token JSON state
> per decision. Four of its numbers did not survive checking against the Laya model
> card and the mine-ai-mcp tool contracts. The corrections are listed in
> [§5](#5-what-this-document-corrects) rather than quietly dropped.

---

## 1. Problem

mine-ai-mcp already beats the game, but it does so with a frontier model driving
every step: the published *beat the game* run used `claude-opus-5[1m]` at effort
`high`, and its 1,121-row transcript is the best available baseline for what a
frontier agent costs per objective.

That cost is not evenly spread. Most decisions in a survival run are repetitive and
low-stakes: collect the next log, walk to the next ore, keep smelting, do not open
the chest. Those do not need deliberation. The expensive minority — which research
target now, whether this structure is reachable, how to sequence a multi-stage
progression — does.

**The lever is the ratio, not the total.** A decision layer that is cheap and
correct on the repetitive majority moves cost per objective without touching the
quality of the minority that actually needs a frontier model.

## 2. What already exists in mine-ai-mcp

Two properties of the server make this a wiring job rather than a reimplementation:

**1. The tools are high-level objectives, not primitives.** `collect_block`,
`navigate`, `craft_item`, `smelt_item`, `explore_frontier` each own a whole objective,
verify their own outcome, and report typed evidence. 34 tools are published, grouped
by job. A caller picks the next objective; it does not script the steps.

**2. The server refuses to plan.** Its own limitations page is explicit: no
autonomous reasoning, no ambient roaming, no meta-game planning. When an action
finishes, the bot stands still until the caller issues the next command. All strategy
belongs to the external caller.

So the decision layer is genuinely missing, and it is missing in exactly one place:
**the choice of which objective to submit next.**

**3. A survival reflex already covers the reactive layer.** The server fights hostile
contact on its own, within eight blocks, with a documented ladder, and applies
hunger, fire, breath and footing responses. This is deterministic System-1 behaviour
that costs no LLM call and needs no gating. It is the reason the harness only has to
decide about *objectives*, not about reflexes.

## 3. The idea

A standalone application — the harness — that owns the decision loop:

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

The harness is an application, not a chat session. It runs unattended, holds its own
queue and state, and survives the terminal closing. That is the whole point of
separating it from an MCP host like Claude Code or OpenCode.

**Decision points, and what each one is for:**

| Decision | Type | Options | Component |
|---|---|---|---|
| Which objective category next | `choice` | ≤ 6 | Laya |
| Continue, or is this objective finished? | `choice` | 2 | Laya |
| Escalate to the LLM? | `choice` | 2 | Laya |
| Which concrete tool + arguments | — | many | LLM, or deterministic binding |

The last row is the one that matters for cost. The first three are small option sets
with a deterministic pre-check — a good fit for a 0.4 B decision model. Tool selection
is not, and must not be, a single Laya question.

## 4. The decisions this rests on

| # | Decision | Because |
|---|---|---|
| D1 | Laya gates, it does not choose tools | A 34-option `choice` question is Laya's measured collapse case: 48 options score 1/48 at the default head budget. Small option sets do not collapse. |
| D2 | Gate on `answer_confidence`, never `action.act_probability` | `act_probability` reads ~1.0 for almost everything; AUROC 0.30 against correctness vs 0.77 for `answer_confidence` (issue #185). |
| D3 | The threshold is **fitted**, not assumed | Both checkpoints ship over-confident and `laya-multilingual` has no fitted temperatures at all. 0.85 is not a defensible constant before the eval exists. |
| D4 | English checkpoint by default | The state is English or JSON. `laya-multilingual` is only needed if state length exceeds the English window, and it is the checkpoint with no shipped calibration. |
| D5 | In-process Laya, not Laya over MCP | The MCP round-trip sits on top of model time and `laya_shortlist` is missing from the shipped MCP server. The harness is Python and already links torch. |
| D6 | Laya runs at **objective** cadence, not tick cadence | The server executes one objective at a time and stands still between them. There is no tick to keep up with. Objective cadence is seconds to minutes; 70 ms is free. |
| D7 | Deterministic layers run **before** Laya | The server already computes recipe trees (`view_crafting_requirements`), chunk maps (`view_frontier`), and a mobility analysis. Asking Laya to re-derive those spends a frontier call on arithmetic. |
| D8 | Only the long-running loop lives in the harness | A smelt takes over ten minutes. The submit → wait → retrieve protocol stays a deterministic loop; the model is consulted at decision points inside it. |

D6 is the one that changes the shape of the problem. An earlier reading of this idea
assumed a per-tick reflex loop, which made 33 ms look load-bearing. It is not: the
server is a foreground/async action machine, and between objectives the bot is
stationary. Laya's latency is not the constraint here. Its **calibration** is.

## 5. What this document corrects

Recorded rather than removed, because each was a reasonable-looking wrong turn.

| Earlier claim | Reality | Source |
|---|---|---|
| "Laya analyses the state in a forward pass (ca. 33 ms)" | 33 ms is a Tesla T4 reference figure for a short state. Measured on this class of machine: **39.3 ms** median `choice` on MPS, 87.6 ms on CPU. A 4,000-token state takes **~1.7 s** on an Apple GPU — 50× the fast path. | ai-minebot `docs/PLAN.md` §7; Laya README, long-context section |
| "`min_confidence=0.85`" | A threshold is a policy chosen from measured accuracy at a chosen coverage on your own data. Both checkpoints ship over-confident; `laya-multilingual` ships **no fitted temperatures at all**. | Laya README, calibration + long-document notes |
| "Large standardized JSON dump, `max_len=8192`, `laya-multilingual`" | `max_len=8192` is real for the multilingual checkpoint, but its own benchmark shows 16–18/20 correct up to ~4,000 tokens and **8–17/20 beyond**. Trade 1.7 s and degraded accuracy for a state that did not need to be that large. | Laya README, long-context section |
| "Laya picks the tool from a list" | Above ~20 options, labels are trimmed until similar ones reach the model as the same text — a wrong answer, not an error. At 48 options the default budget scores **1/48**; widening `head_max_len` recovers 43/48. | Laya LangChain guide §7; README "Honest limits" |
| "Convert the MCP calls into LangChain tools and let the LLM bind them" | Fine for the LLM node. Not for the objective loop: 34 tools with `submission_id`, a one-slot foreground gate, `RESULT_NOT_RETRIEVED` refusals and `SUBMISSION_CONFLICT` identity are a protocol, and a model improvising it deadlocks or double-submits. | mine-ai-mcp `docs/mcp/async-actions.md` |
| "Do not write our own harness" | Correct for the Laya side and wrong for this project. The decision loop is the missing component the server explicitly leaves out. | mine-ai-mcp `docs/mcp/limitations.md` |

## 6. Non-goals for v1

- ❌ **No fine-tuning before the eval demands it.** Zero-shot Laya on this task is
  unmeasured. Measure it first.
- ❌ **No Laya tool-selection question.** D1.
- ❌ **No per-tick loop.** There is no tick to run on, and none is needed.
- ❌ **No claiming an objective succeeded.** Success is read back from the server's
  settled evidence, never inferred from acceptance.
- ❌ **No reimplementing what the server already computes.** D7.
- ❌ **No chat automation.** A frontier model in a chat session is manual automation
  at roughly one call per second, and it dies with the session.

## 7. Success criteria

The harness is not finished when the loop runs. It is finished when these hold:

1. **A measured baseline.** LLM calls per completed objective, recorded on a fixed
   scenario set, before any Laya code exists.
2. **Laya evaluated as data, not as hope.** ≥ 200 labelled decisions; `laya-evals`
   accuracy and ECE recorded; a temperature refit with before/after ECE.
3. **A threshold derived from that eval**, with the coverage it costs stated.
4. **A demonstrated reduction** in LLM calls per objective on the same scenario set,
   with task success rate not regressing.
5. **Escalation as the exception.** The LLM is reached for the minority of decisions
   that warrant deliberation, and the ratio is reported.

Criterion 5 is the one that makes this a system rather than a demo. A harness that
calls the frontier model on every objective has saved nothing.

## 8. Open questions

1. **Which scenario set defines the baseline?** The published *beat-the-game* seed
   and prompt would make the comparison against the reference run direct.
2. **Is a two-tier decision set enough, or does goal decomposition need to be a third
   tier?** "Make a stone pickaxe" may not reduce to one ≤ 6-option category choice.
3. **How are labels produced for the eval set?** Reranking a frontier transcript is
   the cheap route; hand-labelling is the trustworthy one. This decision gates Phase 3.
4. **Does the harness ever call Laya at all once fine-tuned?** If its calibrated
   confidence covers the objective-level distribution, the frontier model may only be
   needed for genuinely open-ended goals.

## Sources

- [mine-ai-mcp](https://github.com/thtesche/mine-ai-mcp) — [tools](https://github.com/thtesche/mine-ai-mcp/blob/main/docs/mcp/tools.md),
  [async actions](https://github.com/thtesche/mine-ai-mcp/blob/main/docs/mcp/async-actions.md),
  [limitations](https://github.com/thtesche/mine-ai-mcp/blob/main/docs/mcp/limitations.md)
- [ai-minebot](https://github.com/thtesche/ai-minebot) `docs/PLAN.md` — measured Laya
  latency, the executor design, the Phase 3 eval gate
- [Laya](https://nandhakishorm.github.io/laya/) — [LangChain/LangGraph guide](https://nandhakishorm.github.io/laya/langchain/),
  model card and "honest limits"
- [beat-the-game dataset](https://huggingface.co/datasets/aibengineering/beat-the-game-minecraft) —
  the reference frontier-model baseline