# The reference run: measured, not estimated

Every number here was extracted from the published dataset
[`aibengineering/beat-the-game-minecraft`](https://huggingface.co/datasets/aibengineering/beat-the-game-minecraft).
Nothing on this page is a recollection. Where an extraction could have silently failed,
the extraction was validated against a known-true case first and that validation is shown.

Provenance for the files used (`transcript.jsonl`, `stream.full.jsonl`, `result.json`): all
three SHA256 sums match the dataset's published `SHA256SUMS`. The published world is
`world.zip`, 65,517,451 B, `sha256:6a2dbf68b3189e9a7ef0f3ab8e3520fcbfb4579fa81d1f152637afdf294616d6`
— so a bit-comparable rerun is possible, at the cost of a 65 MB download.

The run: `claude-opus-5[1m]` at effort high, Claude Code 2.1.268, single agent, **zero
subagents**, fresh spawn to a confirmed Ender Dragon kill.

## Headline

| | |
|---|---|
| Tool calls | **388** |
| Model turns (`num_turns`) | **389** |
| Distinct requests (`request_id`) | 385 |
| Transcript rows | 1,121 |
| Wall clock | **3:56:35** |
| API time | 1,066,036 ms (17.8 min) |
| Total cost | **$93.1681255** |
| Distinct tools advertised | 37 |
| Distinct tools used | **34** |

**$0.240 per tool call.**

## Where the money went

| | total | per call |
|---|---|---|
| cache-read input | 163,325,079 (**99.4%** of all input) | 420,941 |
| cache-creation input | 779,047 | 2,009 |
| output | 148,383 | 382 |
| of which thinking | 67,071 | 173 |

The cost is not the reasoning. It is the transcript being re-read on every single call.
Context grows monotonically:

```
turn   1:     37,417      turn 201:    464,171
turn  11:     55,418      turn 301:    642,679
turn  51:    118,619      turn 385:    779,049
turn 101:    232,453      mean:        426,246
```

164,104,896 context tokens billed across 385 requests. Tool results are not small
(median 4,393 B, p90 12,948 B, max 76,125 B), so the curve is driven by the world, not
by the prompt.

**This is the structural claim the harness rests on, and it is now a number:** an agent
that re-reads its own history pays a cost that grows with the length of the run. The
harness reads a state vector instead, so its prompt does not grow. Measured on the live
model path, one decision cost 16,767 prompt tokens — against a reference average of
420,941. Same order of tools, **25× fewer input tokens per call**, and flat rather than
compounding.

## What it spent calls on

| count | tool | | count | tool |
|---|---|---|---|---|
| 59 | `wait_for_action` | | 13 | `attack_dragon_perch` |
| 39 | `navigate` | | 9 | `sleep` |
| 34 | `collect_mob_drop` | | 9 | `set_survival_policy` |
| 25 | `collect_block` | | 8 | `smelt_item` |
| 24 | `view_status` | | 5 | `note_save` |
| 23 | `craft_item` | | 5 | `cancel_foreground_action` |
| 18 | `equip` | | 5 | `locate_stronghold` |
| 17 | `drop_item` | | 4 | `place_block` |
| 15 | `view_blocks` | | 4 | `pick_up_items` |
| 15 | `destroy_end_crystal` | | 4 | `explore_frontier` |
| 15 | `prepare_dragon_perch` | | 4 | `enter_end_portal` |
| 14 | `query_bot_data` | | 4 | `shoot_dragon` |

344 non-read calls, 44 reads. It barely looked at the world — 44 reads across 388 calls.

Three advertised tools were never called once: **`barter`, `send_message`,
`view_crafting_requirements`**. Their schemas were in every one of the 388 contexts.

**`wait_for_action` is 59 calls, 15.2% of the run.** Polling for an action to settle was
the single largest line item in the reference run.

## The opening: four calls to a wooden pickaxe

This is the comparison number for scenario 1, so it is worth stating exactly. Call 3
collects 16 logs; call 4 batches the craft. Those are the first four tool calls of the run,
with arguments as sent:

```
1. view_status    {}
2. note_read      {"n": 20}
3. collect_block  {"block_name": "logs", "count": 16}
4. craft_item     {"items": [{"crafting_table",1},{"wooden_pickaxe",1},{"wooden_axe",1},{"stick",8}]}
   -> gained crafting_table, wooden_pickaxe, wooden_axe, stick
5. collect_block  {"block_name": "stone", "count": 32}
6. craft_item     {"stone_pickaxe":2, "stone_sword":1, "stone_axe":1, "stone_shovel":1, "furnace":1}
   -> gained all five
```

**Four calls from an empty inventory to a held wooden pickaxe**, two of which are reads.
Stone tier, including a furnace, is call six.

### Validation of this extraction

The `gained` lines are the server's own progress reporting, and pairing each `tool_use`
block to its `tool_use_result` by index is sound because both lists are exactly 388 long.
The pairing was checked rather than assumed: in the 302 results whose body names the tool,
the name in the request matches the name in the result body in 269 cases. The 33 mismatches
are `wait_for_action` results, which report the action they settled rather than themselves.
The extractor was validated by finding 49 `- item: gained n/n` lines resolving to 27
distinct items and 66 total units — a shape consistent with the narrated progression. An
earlier attempt at this extraction searched for `"wooden_pickaxe"` as a quoted JSON key and
reported that the item was *never acquired*. It was in fact acquired on call 4; the results
are markdown, not JSON. A check that finds nothing is not evidence of absence.

## Three design consequences

**1. `craft_item` is the decomposition engine, so do not add a planner call.**
`view_crafting_requirements` was never invoked in 388 calls. The reference asked for the
items it wanted by name and read the recipe tree, the leaf materials, and the workstation
requirement out of the `craft_item` *result*. 23 `craft_item` calls yielded 27 distinct
items and 66 units. A harness that calls a planner first is paying for something the
craft reply already contains.

**2. The model batches, and one call's count matters.**
The reference asked for 16 logs in one call and 5 item types in one craft. The harness's
first live model call asked for `logs x4`. Against a 4-call baseline, under-batching is
visible immediately, and it is a prompt issue rather than a capability limit.

**3. The reference keeps notes; the harness does not.**
`note_read` and `note_save` appear 6 times. The very first action after `view_status` is
`note_read {n: 20}`. Anything the harness expects a decider to remember across decisions
has to be in the harness, because there is no conversation to remember it in.

## What is *not* comparable

Stated plainly so the number is not over-read:

- **Cost.** The reference is Opus list price. The harness baseline runs a free model
  (`cost: 0`). Comparing $0.240/call to our spend compares two different things; only the
  token counts transfer.
- **World.** The published run used `world.zip`. Our live host is a different world, so
  timings and distances differ.
- **Prompt.** Entirely different: 388 growing messages against one system prompt plus a
  state vector.
- **Model.** Frontier at effort high against a free small model. The reference's call count
  is a *ceiling to compare against*, not a target.

What does transfer is the shape: tool calls per milestone, input tokens per call, and the
proportion of calls spent waiting rather than acting.