"""Goals, scenarios, and the checkers that grade them.

Weighted towards refusal. A goal set that silently drops a key, or a checker that
passes because the world could not be read, produces a green run that measured
nothing - and a green run is exactly the artefact nobody re-reads. The positive
paths here are short on purpose; the refusals are the specification.
"""

from __future__ import annotations

import json

import pytest
from conftest import FakeClient, settled, status

from harness.config import BudgetConfig, McpConfig
from harness.decide import Proposal
from harness.goals import GoalBoard, GoalError, GoalSet, count_carried
from harness.ledger import Ledger
from harness.loop import (
    STOP_ESCALATED,
    STOP_GOAL_ATTEMPTS,
    STOP_GOAL_UNKNOWN,
    STOP_PLAN_EXHAUSTED,
    LoopReport,
    RunLoop,
    StepReport,
)
from harness.objective import ObjectiveResult, ObjectiveRunner
from harness.scenario import (
    Check,
    CheckContext,
    Scenario,
    ScenarioError,
    ScenarioReport,
    Verdict,
    evaluate,
    format_report,
    verdicts_for,
    world_facts,
)
from harness.state import StateReader

TOOL = "collect_block"
LOG_ARGS = {"block_name": "logs", "count": 4}


# --------------------------------------------------------------------------
# Goal sets
# --------------------------------------------------------------------------


def test_a_goal_is_an_item_name_and_nothing_more():
    """D18: the harness holds no recipe knowledge, so a goal cannot be a plan."""
    goals = GoalSet.from_any(["wooden_pickaxe"])
    assert goals.items == ("wooden_pickaxe",)
    assert goals.goals[0].count == 1


def test_both_goal_file_shapes_read_the_same():
    assert GoalSet.from_any(["furnace"]).items == GoalSet.from_any({"goals": ["furnace"]}).items


def test_an_unknown_goal_key_is_refused():
    """A typo'd key is a goal the run silently does not chase."""
    with pytest.raises(GoalError) as caught:
        GoalSet.from_any({"goals": [{"item": "furnace", "quanity": 2}]})
    assert "quanity" in str(caught.value)


def test_an_unknown_top_level_goal_key_is_refused():
    with pytest.raises(GoalError) as caught:
        GoalSet.from_any({"goals": ["furnace"], "order": "wooden first"})
    assert "order" in str(caught.value)


def test_an_empty_goal_set_is_refused():
    with pytest.raises(GoalError):
        GoalSet.from_any({"goals": []})


def test_a_goal_with_no_item_name_is_refused():
    with pytest.raises(GoalError) as caught:
        GoalSet.from_any([{"count": 2}])
    assert "item name" in str(caught.value)


@pytest.mark.parametrize("count", [0, -1, "two", 1.5, True])
def test_a_nonsensical_count_is_refused(count):
    with pytest.raises(GoalError):
        GoalSet.from_any([{"item": "furnace", "count": count}])


def test_a_goal_set_that_is_not_a_list_or_object_is_refused():
    with pytest.raises(GoalError) as caught:
        GoalSet.from_any("wooden_pickaxe")
    assert "list of goals" in str(caught.value)


def test_a_missing_goal_file_names_the_file():
    with pytest.raises(GoalError) as caught:
        GoalSet.load("does/not/exist.json")
    assert "does/not/exist.json" in str(caught.value)


def test_malformed_json_in_a_goal_file_is_refused_by_name(tmp_path):
    path = tmp_path / "goals.json"
    path.write_text('["wooden_pickaxe",')
    with pytest.raises(GoalError) as caught:
        GoalSet.load(path)
    assert "not valid JSON" in str(caught.value)


# --------------------------------------------------------------------------
# The board: what the decider is shown, and the cap that now means something
# --------------------------------------------------------------------------


def board(goals, cap=3) -> GoalBoard:
    return GoalBoard.of(GoalSet.from_any(goals), max_attempts=cap)


def test_max_attempts_per_goal_is_now_read_by_something():
    """It was dead config: declared, defaulted, asserted > 0, read by nothing."""
    b = board(["furnace"], cap=2)
    assert not b.exhausted("furnace")
    b.record("furnace", ok=False, outcome="failed:failed")
    assert not b.exhausted("furnace")
    b.record("furnace", ok=False, outcome="failed:failed")
    assert b.exhausted("furnace")


def test_a_successful_attempt_still_counts_against_the_cap():
    """The cap is on attempts, not failures: a long recipe is not a stuck loop."""
    b = board(["furnace"], cap=1)
    b.record("furnace", ok=True, outcome="settled:succeeded")
    assert b.exhausted("furnace")


def test_the_decider_is_shown_attempts_and_exhaustion():
    b = board(["furnace", "shield"], cap=2)
    b.record("furnace", ok=False, outcome="failed:failed")
    view = {row["item"]: row for row in b.view()}
    assert view["furnace"]["attempts"] == 1
    assert view["furnace"]["succeeded"] is False
    assert view["furnace"]["last_outcome"] == "failed:failed"
    assert view["furnace"]["exhausted"] is False
    assert view["shield"]["attempts"] == 0
    assert [g.item for g in b.open()] == ["furnace", "shield"]


def test_an_exhausted_goal_leaves_the_open_set():
    b = board(["furnace", "shield"], cap=1)
    b.record("furnace", ok=False, outcome="failed:failed")
    assert [g.item for g in b.open()] == ["shield"]


def test_a_board_with_no_goals_is_disabled_rather_than_empty():
    b = GoalBoard.of(None, max_attempts=3)
    assert not b.enabled
    assert b.view() == []


def test_count_carried_reads_the_rendered_stack_names():
    assert count_carried(["acacia_logx4", "stickx8"], "acacia_log") == 4
    assert count_carried(["acacia_logx4", "stickx8"], "stick") == 8
    assert count_carried(["shield"], "shield") == 1
    assert count_carried(["acacia_logx4"], "log") == 0
    assert count_carried([], "shield") == 0


def test_count_carried_does_not_split_a_name_that_ends_in_x():
    assert count_carried(["max_ender_pearlx2"], "max_ender_pearl") == 2
    assert count_carried(["max_ender_pearlx2"], "max_ender_pearl_") == 0


# --------------------------------------------------------------------------
# Scenario files
# --------------------------------------------------------------------------


MINIMAL = {
    "name": "s",
    "goals": ["furnace"],
    "checks": [{"kind": "holds_item", "item": "furnace"}],
}


def scenario_file(tmp_path, payload):
    path = tmp_path / "s.json"
    path.write_text(json.dumps(payload))
    return path


def test_a_minimal_scenario_loads(tmp_path):
    scenario = Scenario.load(scenario_file(tmp_path, MINIMAL))
    assert scenario.name == "s"
    assert scenario.goals.items == ("furnace",)
    assert scenario.max_steps == 24
    assert scenario.reference == {}


def test_a_scenario_with_no_checks_is_refused(tmp_path):
    """A scenario with no checker is a demo (D19)."""
    with pytest.raises(ScenarioError) as caught:
        Scenario.load(scenario_file(tmp_path, {**MINIMAL, "checks": []}))
    assert "demo" in str(caught.value)


def test_an_unknown_check_kind_is_refused_and_names_the_known_ones(tmp_path):
    """An unknown check must not evaluate to true: a scenario that cannot fail
    looks exactly like one that passed."""
    with pytest.raises(ScenarioError) as caught:
        Scenario.load(scenario_file(tmp_path, {**MINIMAL, "checks": [{"kind": "holds_itemm"}]}))
    message = str(caught.value)
    assert "holds_itemm" in message
    assert "holds_item" in message


def test_an_unknown_scenario_key_is_refused(tmp_path):
    with pytest.raises(ScenarioError) as caught:
        Scenario.load(scenario_file(tmp_path, {**MINIMAL, "maxStepss": 4}))
    assert "maxStepss" in str(caught.value)


def test_a_scenario_with_no_name_is_refused(tmp_path):
    with pytest.raises(ScenarioError):
        Scenario.load(scenario_file(tmp_path, {**MINIMAL, "name": "  "}))


@pytest.mark.parametrize("bad", [0, -1, "four", 2.5])
def test_a_nonsensical_step_ceiling_is_refused(tmp_path, bad):
    with pytest.raises(ScenarioError):
        Scenario.load(scenario_file(tmp_path, {**MINIMAL, "maxSteps": bad}))


def test_a_check_with_no_kind_is_refused(tmp_path):
    with pytest.raises(ScenarioError):
        Scenario.load(scenario_file(tmp_path, {**MINIMAL, "checks": [{"item": "furnace"}]}))


def test_a_reference_that_is_not_an_object_is_refused(tmp_path):
    with pytest.raises(ScenarioError):
        Scenario.load(scenario_file(tmp_path, {**MINIMAL, "reference": "the other run"}))


def test_malformed_json_in_a_scenario_is_refused(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("{not json")
    with pytest.raises(ScenarioError) as caught:
        Scenario.load(path)
    assert "not valid JSON" in str(caught.value)


def test_a_missing_scenario_file_is_named_rather_than_a_traceback():
    """A wrong path is a typo; the reader needs the filename, not a pathlib stack."""
    with pytest.raises(ScenarioError) as caught:
        Scenario.load("does/not/exist.json")
    assert "does/not/exist.json" in str(caught.value)
    assert "could not be read" in str(caught.value)


def test_the_shipped_scenarios_parse():
    """The files in `scenarios/` are what the exit criterion is measured on."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "scenarios"
    files = sorted(root.glob("*.json"))
    assert files, "no scenario files; the scenario set is the baseline"
    for path in files:
        scenario = Scenario.load(path)
        assert scenario.checks, f"{path.name} has no checks"
        assert scenario.goals.goals


def test_the_first_scenario_records_the_reference_numbers():
    """D19: the comparison target is written down before the run, not after."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scenarios" / "first-pickaxe.json"
    scenario = Scenario.load(path)
    assert scenario.reference["toolCalls"] == 4
    assert scenario.reference["objectiveSubmissions"] == 2
    assert scenario.goals.items == ("wooden_pickaxe",)
    kinds = [check.kind for check in scenario.checks]
    assert "holds_item" in kinds and "within_calls" in kinds


# --------------------------------------------------------------------------
# World facts
# --------------------------------------------------------------------------


#: A ``view_status`` situation, transcribed from the server's own schema rather
#: than from recollection: ``inventoryStackSchema`` and ``toolSnapshotEntrySchema``
#: in ``mine-ai-mcp/src/actions/view-status/contract.ts``, with ``clock`` and
#: ``lastDeath`` from the same file. The rows that hold nothing are present and
#: carry ``tier: "none"`` with ``item: null`` - the server emits one row per
#: equipment class whether or not anything is held, and a fixture that omitted
#: them would not have caught the reader disagreeing about which class is which.
def tool_row(item, tier, cls):
    return {
        "class": cls,
        "tier": tier,
        "item": item,
        "slot": 0 if item else None,
        "durabilityLeft": 59 if item else None,
        "maximumDurability": 59 if item else None,
    }


def stack(name, count, slot=36):
    return {
        "slot": slot,
        "location": "hotbar",
        "name": name,
        "count": count,
        "held": False,
        "durability": None,
    }


SITUATION = {
    "clock": {
        "timeOfDay": 1000,
        "phase": "day",
        "ticksUntilChange": 11000,
        "minutesUntilChange": 9.1,
        "sleeping": False,
        "raining": False,
    },
    "inventory": {
        "usedSlots": 3,
        "freeSlots": 33,
        "stacks": [
            stack("wooden_pickaxe", 1, 36),
            stack("acacia_log", 16, 37),
            stack("stick", 8, 38),
        ],
    },
    "tools": {
        "tools": [
            tool_row("wooden_pickaxe", "wooden", "pickaxe"),
            tool_row(None, "none", "axe"),
        ],
        "armour": [tool_row(None, "none", "helmet")],
    },
    "lastDeath": {
        "dimension": "overworld",
        "position": {"x": 1.0, "y": 64.0, "z": 2.0},
        "observedAt": "2026-10-01T00:00:00.000Z",
        "cause": "zombie",
    },
}


def test_world_facts_counts_every_stack_not_the_first_twelve():
    """The bounded carry list would hide slot twenty; the checker must not."""
    stacks = [{"name": f"item_{i}", "count": 1} for i in range(25)]
    facts = world_facts({**SITUATION, "inventory": {"stacks": stacks}})
    assert len(facts.inventory) == 25
    assert facts.holds("item_24")


def test_world_facts_ranks_the_best_tool_by_tier_not_by_row_order():
    """The reader picks the highest *tier*, so a lower row cannot win on position."""
    facts = world_facts({**SITUATION, "tools": {"tools": [
        tool_row("wooden_pickaxe", "wooden", "pickaxe"),
        tool_row("iron_pickaxe", "iron", "shovel"),   # better tier, listed later
        tool_row("shield", "other", "shield"),       # a tier the order does not know
        tool_row(None, "none", "axe"),
    ], "armour": []}})
    assert facts.best_tool == "iron_pickaxe"
    assert facts.best_tool_tier == "iron"
    assert facts.tool_tier_rank() == 2


def test_world_facts_reports_an_unrankable_tier_as_a_warning_not_a_bad_read():
    """Ranking it wrongly would report the wrong best tool with total confidence.

    A warning, not `unverified`: the read was complete, and a bot holding a
    stone pickaxe *and* a trident has passed a "stone tier" check - folding the
    two together would fail that run for a reason that is not the reason.
    """
    facts = world_facts({**SITUATION, "tools": {"tools": [
        tool_row("stone_pickaxe", "stone", "pickaxe"),
        tool_row("trident", "other", "shovel"),
    ], "armour": []}})
    assert not facts.unverified
    assert any("not in the harness tier order" in note for note in facts.warnings)
    verdict = run(Check("best_tool_at_least", {"tier": "stone"}), facts)
    assert verdict.passed
    assert "could not rank" in verdict.detail


def test_an_unrankable_tier_alone_does_not_fail_the_hold_check():
    facts = world_facts({**SITUATION, "tools": {"tools": [
        tool_row("wooden_pickaxe", "wooden", "pickaxe"),
        tool_row("trident", "other", "shovel"),
    ], "armour": []}})
    assert run(Check("holds_item", {"item": "wooden_pickaxe"}), facts).passed


def test_world_facts_reads_nothing_held_as_no_answer_not_as_an_error():
    """`tier: "none"` with `item: null` is a fact about the world, not a bad read."""
    facts = world_facts({**SITUATION, "tools": {"tools": [
        tool_row(None, "none", "pickaxe"),
    ], "armour": [tool_row(None, "none", "helmet")]}})
    assert facts.best_tool is None
    assert facts.tool_tier_rank() == -1
    assert not [n for n in facts.warnings if "tier order" in n]
    assert not facts.unverified


def test_a_world_read_with_no_stacks_is_marked_unverified():
    facts = world_facts({**SITUATION, "inventory": {}})
    assert facts.unverified
    assert not facts.holds("wooden_pickaxe")


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def ctx(facts, *, steps=(), deaths=0, goal_attempts=None) -> CheckContext:
    report = LoopReport(steps=list(steps), deaths_absorbed=deaths)
    report.goal_attempts = dict(goal_attempts or {})
    return CheckContext(facts=facts, loop=report, params={})


def run(check: Check, facts, **kwargs) -> Verdict:
    return evaluate(check, ctx(facts, **kwargs))


def a_step(index=0, *, goal=None, ok=True, evidence=True, state=None, status_=None):
    """A step with a settled result.

    ``ok=False`` means a *failed* settlement rather than a success with a flag
    set False - the two are indistinguishable in :class:`ObjectiveResult`, so the
    flag is what derives the state. Passing ``state``/``status_`` explicitly
    overrides that, which is how the never-settled case is built.
    """
    if state is None:
        state, status_ = ("settled", "succeeded") if ok else ("settled", "failed")
    return StepReport(
        index=index,
        tool=TOOL,
        state_hash="x",
        goal=goal,
        result=ObjectiveResult(
            tool=TOOL, submission_id="s", action_id="a", state=state, status=status_,
            output={}, evidence_ok=evidence, polls=1, duration_ms=1,
        ),
    )


def test_holds_item_passes_and_names_the_count():
    facts = world_facts(SITUATION)
    ok = run(Check("holds_item", {"item": "wooden_pickaxe"}), facts)
    assert ok.passed and "holds 1x wooden_pickaxe" in ok.detail
    short = run(Check("holds_item", {"item": "wooden_pickaxe", "count": 2}), facts)
    assert not short.passed and "needed 2x" in short.detail


def test_holds_item_without_an_item_name_fails_rather_than_passing():
    verdict = run(Check("holds_item", {}), world_facts(SITUATION))
    assert not verdict.passed
    assert "needs an `item`" in verdict.detail


@pytest.mark.parametrize("count", [0, -2, "two", None, True])
def test_holds_item_with_a_nonsensical_count_fails(count):
    verdict = run(Check("holds_item", {"item": "furnace", "count": count}), world_facts(SITUATION))
    assert not verdict.passed


@pytest.mark.parametrize(
    "kind,params",
    [
        ("holds_item", {"item": "wooden_pickaxe"}),
        ("best_tool_at_least", {"tier": "stone"}),
        ("time_phase_is", {"phase": "day"}),
    ],
)
def test_a_world_check_fails_when_the_world_could_not_be_read(kind, params):
    """No evidence is not a pass."""
    verdict = run(Check(kind, params), None)
    assert not verdict.passed, kind
    assert "could not be read" in verdict.detail, kind


def test_a_world_check_fails_on_an_incomplete_read():
    facts = world_facts({"clock": {"phase": "day"}})  # no inventory section
    verdict = run(Check("holds_item", {"item": "furnace"}), facts)
    assert not verdict.passed
    assert "incomplete" in verdict.detail


def test_an_unrankable_tier_is_refused_rather_than_assumed():
    verdict = run(Check("best_tool_at_least", {"tier": "obsidian"}), world_facts(SITUATION))
    assert not verdict.passed
    assert "not in the harness tool tier order" in verdict.detail


def test_best_tool_at_least_compares_tiers_not_names():
    facts = world_facts(SITUATION)
    assert run(Check("best_tool_at_least", {"tier": "wooden"}), facts).passed
    assert not run(Check("best_tool_at_least", {"tier": "stone"}), facts).passed


def test_time_phase_is_compares_the_phase():
    facts = world_facts(SITUATION)
    assert run(Check("time_phase_is", {"phase": "day"}), facts).passed
    assert not run(Check("time_phase_is", {"phase": "night"}), facts).passed


def test_within_calls_measures_objectives_not_reads():
    steps = [a_step(i) for i in range(3)]
    assert run(Check("within_calls", {"calls": 3}), None, steps=steps).passed
    over = run(Check("within_calls", {"calls": 2}), None, steps=steps)
    assert not over.passed and "used 3" in over.detail


@pytest.mark.parametrize("budget", [{}, {"calls": 0}, {"calls": -1}, {"calls": "four"}])
def test_within_calls_without_a_sane_budget_fails(budget):
    assert not run(Check("within_calls", budget), None).passed


def test_no_death_uses_the_run_not_the_world():
    """A death from a previous run must not fail this one."""
    facts = world_facts(SITUATION)  # the world carries a lastDeath
    assert run(Check("no_death", {}), facts, steps=[a_step(0)]).passed
    assert not run(Check("no_death", {}), facts, steps=[a_step(0)], deaths=1).passed


def test_no_run_shaped_check_passes_on_a_run_that_did_nothing():
    """"Never died" over zero steps is an empty set, not a clean run.

    Every one of these would read green on a loop that stopped at the first
    unreadable world, which is the failure this repo cares about most: a check
    that passes on nothing has not passed.
    """
    for kind in ("no_death", "within_calls", "evidence_verified"):
        verdict = run(Check(kind, {"calls": 8}), None)
        assert not verdict.passed, kind
        assert "made no tool call" in verdict.detail, kind


def test_a_step_that_ran_is_enough_for_the_run_shaped_checks():
    """The guard is about an empty run, not about failures inside a real one.

    One failed step is still a run. `no_death` and `within_calls` then answer
    normally - a failure is not a death, and one call is within eight - while
    `evidence_verified` still fails, because that objective carried no evidence.
    """
    for kind, expected in (("no_death", True), ("within_calls", True), ("evidence_verified", False)):
        verdict = run(Check(kind, {"calls": 8}), None, steps=[a_step(0, ok=False, evidence=False)])
        assert verdict.passed is expected, (kind, verdict.detail)
        assert "made no tool call" not in verdict.detail, kind


def test_evidence_verified_notices_a_settled_step_with_no_evidence():
    good = run(Check("evidence_verified", {}), None, steps=[a_step(0)])
    assert good.passed
    bad = run(Check("evidence_verified", {}), None, steps=[a_step(0, evidence=False), a_step(1)])
    assert not bad.passed
    assert "steps [0]" in bad.detail


def test_evidence_verified_ignores_a_step_that_never_settled():
    """A refused step is not a settled objective missing evidence.

    Asserted via the step's own state, so the helper cannot quietly turn a
    refused step into a settled one and make this pass for the wrong reason.
    """
    refused = a_step(0, state="refused", status_=None)
    assert refused.result is not None and refused.result.state == "refused"
    verdict = run(Check("evidence_verified", {}), None, steps=[refused])
    assert verdict.passed
    assert "0/0" in verdict.detail


def test_evidence_verified_does_not_miss_a_failed_settlement():
    """A failed settlement is still a settlement, and its evidence counts."""
    failed = a_step(0, ok=False)
    verdict = run(Check("evidence_verified", {}), None, steps=[failed])
    assert verdict.passed and "1/1" in verdict.detail
    missing = run(Check("evidence_verified", {}), None, steps=[a_step(0, ok=False, evidence=False)])
    assert not missing.passed and "steps [0]" in missing.detail


def test_goal_succeeded_needs_a_success_not_merely_an_attempt():
    good = run(Check("goal_succeeded", {}), None,
               steps=[a_step(0, goal="furnace")], goal_attempts={"furnace": 1})
    assert good.passed
    bad = run(Check("goal_succeeded", {}), None,
              steps=[a_step(0, goal="furnace", ok=False)], goal_attempts={"furnace": 1})
    assert not bad.passed and "furnace" in bad.detail


def test_goal_succeeded_fails_when_the_run_recorded_no_attempts():
    verdict = run(Check("goal_succeeded", {}), None, steps=[])
    assert not verdict.passed
    assert "no goal attempts" in verdict.detail


# --------------------------------------------------------------------------
# The loop's goal refusals
# --------------------------------------------------------------------------


def build(decider, ledger_config, goals=None) -> RunLoop:
    client = FakeClient(script={
        "view_status": [status()],
        TOOL: [settled("a1")],
    })
    ledger = Ledger(ledger_config, "goal-test")
    return RunLoop(
        StateReader(client),
        ObjectiveRunner(client, McpConfig(initial_wait_ms=10, poll_ms=10, max_polls=5),
                        BudgetConfig(objective_ms=5_000, max_attempts_per_goal=3, gate_wait_ms=0),
                        ledger),
        decider,
        ledger=ledger,
        goals=goals,
    )


def always(goal):
    """A decider that proposes the same objective, naming ``goal``, forever."""

    class Stubborn:
        goals = None

        def __init__(self) -> None:
            self.n = 0

        async def propose(self, vector, *, step):
            self.n += 1
            return Proposal(tool=TOOL, arguments=dict(LOG_ARGS), goal=goal)

    return Stubborn()


def exactly(goal, count):
    class Once:
        goals = None

        def __init__(self) -> None:
            self.n = 0

        async def propose(self, vector, *, step):
            self.n += 1
            if self.n > count:
                return None
            return Proposal(tool=TOOL, arguments=dict(LOG_ARGS), goal=goal)

    return Once()


async def test_the_loop_refuses_a_goal_outside_the_set(ledger_config):
    """Counting progress toward a goal nobody asked for is not progress."""
    loop = build(always("diamond_sword"), ledger_config,
                 board(["wooden_pickaxe"], cap=3))
    report = await loop.run(max_steps=4)
    assert report.stop_reason == STOP_GOAL_UNKNOWN
    assert "diamond_sword" in report.detail
    assert "wooden_pickaxe" in report.detail
    assert report.steps == [], "the objective must not have run"


async def test_the_loop_refuses_an_objective_that_names_no_goal(ledger_config):
    """A run with goals measures calls per goal; an unnamed objective is uncountable."""
    loop = build(always(None), ledger_config, board(["wooden_pickaxe"], cap=3))
    report = await loop.run(max_steps=4)
    assert report.stop_reason == STOP_GOAL_UNKNOWN
    assert "named no goal" in report.detail
    assert report.steps == []


async def test_a_run_with_no_goals_is_unaffected_by_any_of_it(ledger_config):
    """`run-loop` has no goal set, so it must not start refusing objectives."""
    report = await build(exactly(None, 2), ledger_config).run(max_steps=6)
    assert report.stop_reason == STOP_PLAN_EXHAUSTED
    assert len(report.steps) == 2


async def test_an_empty_board_is_the_same_as_no_board(ledger_config):
    report = await build(exactly(None, 2), ledger_config, GoalBoard.of(None, max_attempts=3)).run()
    assert report.stop_reason == STOP_PLAN_EXHAUSTED
    assert len(report.steps) == 2


async def test_attempts_are_banked_against_the_goal_the_step_named(ledger_config):
    b = board(["wooden_pickaxe"], cap=5)
    report = await build(exactly("wooden_pickaxe", 2), ledger_config, b).run(max_steps=6)
    assert report.goal_attempts == {"wooden_pickaxe": 2}
    assert b.attempts["wooden_pickaxe"] == 2
    assert all(step.goal == "wooden_pickaxe" for step in report.steps)


async def test_the_cap_stops_a_run_that_keeps_at_an_exhausted_goal(ledger_config):
    """`max_attempts_per_goal` finally does something."""
    report = await build(always("wooden_pickaxe"), ledger_config,
                         board(["wooden_pickaxe"], cap=2)).run(max_steps=10)
    assert report.stop_reason == STOP_GOAL_ATTEMPTS
    assert report.goal_attempts == {"wooden_pickaxe": 2}
    assert len(report.steps) == 2, "the third attempt must not have been submitted"


async def test_the_decider_is_handed_the_board_by_the_loop(ledger_config):
    """Handed over, not left to be found: a decider that ignores it is visibly ignoring it."""
    decider = exactly("wooden_pickaxe", 1)
    await build(decider, ledger_config, board(["wooden_pickaxe"], cap=3)).run(max_steps=1)
    assert isinstance(decider.goals, GoalBoard)


async def test_the_goal_reaches_the_ledger_row(ledger_config):
    """A row a reader cannot attribute to a goal cannot be counted per goal."""
    with Ledger(ledger_config, "goal-test") as ledger:
        client = FakeClient(script={"view_status": [status()], TOOL: [settled("a1")]})
        loop = RunLoop(
            StateReader(client),
            ObjectiveRunner(client, McpConfig(initial_wait_ms=10, poll_ms=10, max_polls=5),
                            BudgetConfig(objective_ms=5_000, gate_wait_ms=0), ledger),
            exactly("wooden_pickaxe", 1),
            ledger=ledger,
            goals=board(["wooden_pickaxe"], cap=3),
        )
        await loop.run(max_steps=2)
        rows = ledger.read_all()
    objective_rows = [row for row in rows if row.get("objective_tool")]
    assert objective_rows
    assert objective_rows[0]["question"]["goal"] == "wooden_pickaxe"


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def test_a_report_passes_only_when_the_loop_also_finished_cleanly():
    """All checks green on a run that stopped on an escalation is still a failure."""
    loop = LoopReport(steps=[], deaths_absorbed=0)
    passing = [Verdict("holds_item", True, "holds 1x wooden_pickaxe")]
    assert ScenarioReport("s", loop, passing).passed
    loop.stop_reason = STOP_ESCALATED
    assert not ScenarioReport("s", loop, passing).passed


def test_a_report_with_no_checks_never_passes():
    assert not ScenarioReport("s", LoopReport(), []).passed


def test_a_report_with_one_failing_check_fails():
    loop = LoopReport(steps=[])
    assert not ScenarioReport(
        "s", loop, [Verdict("a", True, ""), Verdict("b", False, "no")]
    ).passed


def test_verdicts_are_evaluated_in_the_order_the_scenario_lists_them(tmp_path):
    scenario = Scenario.load(scenario_file(tmp_path, {
        **MINIMAL,
        "checks": [{"kind": "within_calls", "calls": 1}, {"kind": "evidence_verified"}],
    }))
    kinds = [v.kind for v in verdicts_for(scenario, LoopReport(), world_facts(SITUATION))]
    assert kinds == ["within_calls", "evidence_verified"]


def test_the_reference_numbers_are_carried_into_the_report(tmp_path):
    scenario = Scenario.load(scenario_file(tmp_path, {**MINIMAL, "reference": {"toolCalls": 4}}))
    report = ScenarioReport(
        "s", LoopReport(), verdicts_for(scenario, LoopReport(), world_facts(SITUATION)),
        reference=scenario.reference,
    )
    assert report.summary()["reference"] == {"toolCalls": 4}


def test_the_text_report_names_a_failing_check_and_a_bad_stop():
    text = format_report(ScenarioReport(
        "s",
        LoopReport(stop_reason=STOP_ESCALATED, detail="no model"),
        [Verdict("holds_item", False, "holds 0x wooden_pickaxe, needed 1x")],
    ))
    assert "FAIL" in text
    assert "holds 0x wooden_pickaxe" in text
    assert "escalated" in text
    assert "no model" in text


def test_the_text_report_shows_attempts_per_goal():
    loop = LoopReport(steps=[a_step(0, goal="wooden_pickaxe")])
    loop.goal_attempts = {"wooden_pickaxe": 1}
    assert "wooden_pickaxe=1" in format_report(
        ScenarioReport("s", loop, [Verdict("holds_item", True, "")])
    )