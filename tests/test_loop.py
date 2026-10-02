"""Tests for the decision loop and the decider seam.

The loop is where Phase 1's evidence comes from, so what is under test is mostly
*what it refuses to do*: it will not decide from a state that would not verify, it
will not downgrade an escalation into a guess, and it will not treat a death as the
end of the run. Each is a case where the convenient behaviour - carry on with what
you have - would produce a run that looks fine and means nothing.
"""

from __future__ import annotations

import pytest
from conftest import FakeClient, settled, situation, status

from harness.config import BudgetConfig, McpConfig
from harness.decide import (
    Escalation,
    Proposal,
    ScriptedArgumentError,
    ScriptedDecider,
    confidence_gate,
)
from harness.ledger import Ledger
from harness.loop import (
    STOP_ESCALATED,
    STOP_PLAN_EXHAUSTED,
    STOP_REFUSED,
    STOP_STEP_LIMIT,
    STOP_UNVERIFIED_STATE,
    RunLoop,
)
from harness.objective import ObjectiveRunner
from harness.state import StateReader, StateVector

TOOL = "collect_block"


def build(client: FakeClient, decider, ledger_config, **kwargs) -> RunLoop:
    """A loop over a fake client.

    The runner and the loop share one ledger rather than splitting the rows
    between them: the runner knows the objective's outcome, including the refusals
    it raises, and the loop knows about escalations and unreadable states. They are
    different events, so they append different rows to the same file - which is
    what makes the ledger a record of the run rather than of whichever component
    happened to notice.
    """
    ledger = Ledger(ledger_config, "loop-test")
    budget = BudgetConfig(objective_ms=5_000, max_attempts_per_goal=3,
                          max_consecutive_failures=2, gate_wait_ms=0)
    return RunLoop(
        StateReader(client),
        ObjectiveRunner(client, McpConfig(initial_wait_ms=10, poll_ms=10, max_polls=5),
                        budget, ledger),
        decider,
        ledger=ledger,
        **kwargs,
    )


def status_missing(section: str):
    """A situation with one promised section absent, so it will not verify."""
    return status({k: v for k, v in situation().items() if k != section})


# --- sequencing -------------------------------------------------------------


async def test_loop_runs_the_script_in_order(ledger_config):
    client = FakeClient(script={
        "view_status": [status()],
        TOOL: [settled("a1"), settled("a2")],
    })
    report = await build(client, ScriptedDecider([(TOOL, {}), (TOOL, {})]), ledger_config).run()

    assert report.stop_reason == STOP_PLAN_EXHAUSTED
    assert report.ok
    assert [step.tool for step in report.steps] == [TOOL, TOOL]
    assert report.objectives_succeeded == 2
    assert report.verified_evidence == 2


async def test_every_row_carries_the_state_the_decision_was_made_against(ledger_config):
    """The Phase 1 deliverable: a row is unreadable without the state it answers.

    Without this the ledger records what was submitted and what came back but not
    why, and the Phase 2 baseline has no input to compare against.
    """
    ledger = Ledger(ledger_config, "loop-test")
    client = FakeClient(script={"view_status": [status()], TOOL: [settled("a1")]})
    loop = RunLoop(
        StateReader(client),
        ObjectiveRunner(client, McpConfig(initial_wait_ms=10, poll_ms=10, max_polls=5),
                        BudgetConfig(objective_ms=5_000, gate_wait_ms=0), ledger),
        ScriptedDecider([(TOOL, {})]),
        ledger=ledger,
    )
    await loop.run()

    rows = [row for row in ledger.read_all() if row.get("objective_tool")]
    assert rows, "the objective must produce a row"
    row = rows[0]
    assert row["state_hash"], "the row must carry a state hash"
    assert row["state_vector"]["health"] == 20.0
    assert row["answer"] == TOOL
    assert row["question"]["source"] == "scripted", "a scripted row must say it was scripted"
    assert row["answer_confidence"] is None


async def test_state_reads_precede_each_objective(ledger_config):
    client = FakeClient(script={
        "view_status": [status()],
        TOOL: [settled("a1"), settled("a2")],
    })
    await build(client, ScriptedDecider([(TOOL, {}), (TOOL, {})]), ledger_config).run()

    order = [name for name, _ in client.calls]
    # Two objectives and a read before each. The plan runs dry before a fourth
    # read, so three is the exact count rather than "at least".
    assert order.count("view_status") == 3
    assert order.count(TOOL) == 2
    assert order.index("view_status") < order.index(TOOL)


# --- what the loop refuses --------------------------------------------------


async def test_a_state_that_will_not_verify_is_not_decided_against(ledger_config):
    """The loop re-reads, then stops. It never shows the decider a broken vector."""
    client = FakeClient(script={"view_status": [status_missing("mobility")]})
    report = await build(client, ScriptedDecider([(TOOL, {})]), ledger_config).run()

    assert report.stop_reason == STOP_UNVERIFIED_STATE
    assert report.steps == [], "no objective may run against an unread world"
    # One re-read, then give up: two failures is a contract change, not a blip.
    assert report.unverified_reads == 2


async def test_a_dropped_connection_is_ridden_out_by_the_re_read(ledger_config):
    """One bad read is a blip; the re-read is a fresh attempt, not a guess."""
    client = FakeClient(script={
        "view_status": [status_missing("mobility"), status()],
        TOOL: [settled("a1")],
    })
    report = await build(client, ScriptedDecider([(TOOL, {})]), ledger_config).run()

    assert report.stop_reason == STOP_PLAN_EXHAUSTED
    assert report.objectives_succeeded == 1
    assert report.unverified_reads == 1


async def test_the_step_ceiling_stops_a_loop_that_never_exhausts_its_plan(ledger_config):
    """Bounded, because an endless plan is a hang that reports as progress."""

    class Endless:
        async def propose(self, vector, *, step):
            return Proposal(tool=TOOL, arguments={})

    client = FakeClient(script={"view_status": [status()], TOOL: [settled("a1")]})
    report = await build(client, Endless(), ledger_config).run(max_steps=3)

    assert report.stop_reason == STOP_STEP_LIMIT
    assert not report.ok, "hitting the ceiling is not finishing the plan"
    assert len(report.steps) == 3


async def test_an_unresolvable_refusal_stops_the_run(ledger_config):
    from conftest import refused

    client = FakeClient(script={
        "view_status": [status()],
        TOOL: [refused("RUNTIME_UNAVAILABLE")],
    })
    report = await build(client, ScriptedDecider([(TOOL, {}), (TOOL, {})]), ledger_config).run()

    assert report.stop_reason == STOP_REFUSED
    assert len(report.steps) == 1, "the run must not continue past a refusal"


# --- escalation -------------------------------------------------------------


async def test_an_escalation_stops_the_run_and_is_recorded(ledger_config):
    """Refusing to guess is the right answer, so it must not be retried into one."""

    class Escalating:
        calls = 0

        async def propose(self, vector, *, step):
            Escalating.calls += 1
            return Escalation(reason="no calibrated threshold", question={"kind": "objective"})

    ledger = Ledger(ledger_config, "loop-test")
    client = FakeClient(script={"view_status": [status()]})
    loop = RunLoop(
        StateReader(client),
        ObjectiveRunner(client, McpConfig(initial_wait_ms=10, poll_ms=10, max_polls=5),
                        BudgetConfig(objective_ms=5_000, gate_wait_ms=0), ledger),
        Escalating(),
        ledger=ledger,
    )
    report = await loop.run()

    assert report.stop_reason == STOP_ESCALATED
    assert not report.ok
    assert Escalating.calls == 1, "an escalation must not be re-asked until it becomes a guess"
    row = ledger.read_all()[0]
    assert row["escalated"] is True
    assert row["outcome"] == "escalated"
    assert row["escalation_reason"] == "no calibrated threshold"
    assert row["state_hash"], "an escalation is still a decision made against a state"


# --- death ------------------------------------------------------------------


async def test_a_death_during_a_failed_objective_is_absorbed_and_the_run_continues(
    ledger_config,
):
    """The bot dying is a fact about the objective, not the end of the run.

    The bot respawns and the remaining objectives are still answerable, so the loop
    keeps going - but the death is recorded against the objective it interrupted,
    because "the objective failed" and "the objective failed because the bot died"
    are different facts and only one of them says anything about the objective.
    """
    dead = status(situation(lastDeath={
        "observedAt": "2026-10-02T14:00:00.000Z",
        "cause": "MineAI was slain by Zombie",
    }))
    # Read order within a failing step: the trusted state, then the death check.
    client = FakeClient(script={
        "view_status": [status(), dead, status()],
        TOOL: [settled("a1", status="failed", error="died"), settled("a2")],
    })
    report = await build(
        client, ScriptedDecider([(TOOL, {}), (TOOL, {})]), ledger_config
    ).run()

    assert report.deaths_absorbed == 1
    assert report.steps[0].death_cause == "MineAI was slain by Zombie"
    assert report.stop_reason == STOP_PLAN_EXHAUSTED, "a death must not end the run"
    assert report.objectives_succeeded == 1, "the second objective still ran"


async def test_no_death_is_reported_when_the_state_only_looks_different(ledger_config):
    """The same death timestamp is not a new death.

    Otherwise every failed objective in a world where the bot has already died
    would be attributed to that death, which would make the field useless.
    """
    already_dead = status(situation(lastDeath={
        "observedAt": "2026-10-02T13:00:00.000Z",
        "cause": "MineAI fell from a high place",
    }))
    client = FakeClient(script={
        "view_status": [already_dead, already_dead],
        TOOL: [settled("a1", status="failed", error="no path")],
    })
    report = await build(client, ScriptedDecider([(TOOL, {})]), ledger_config).run()

    assert report.deaths_absorbed == 0
    assert report.steps[0].death_cause is None


async def test_only_a_failed_objective_pays_for_a_death_check(ledger_config):
    """Death cannot make a succeeded objective not have succeeded, so a success
    skips the extra read that a failure spends.

    Both runs make one read per loop pass, and both discover the plan is exhausted
    on a final pass. Only the failing run adds the death check on top.
    """
    succeeded = FakeClient(script={"view_status": [status()], TOOL: [settled("a1")]})
    await build(succeeded, ScriptedDecider([(TOOL, {})]), ledger_config).run()

    failed = FakeClient(script={
        "view_status": [status(), status()],
        TOOL: [settled("a1", status="failed", error="no path")],
    })
    await build(failed, ScriptedDecider([(TOOL, {})]), ledger_config).run()

    assert len(succeeded.args_for("view_status")) == 2
    assert len(failed.args_for("view_status")) == 3


# --- submission identity ----------------------------------------------------


async def test_the_submission_prefix_lands_on_every_submitted_id(ledger_config):
    client = FakeClient(script={"view_status": [status()], TOOL: [settled("a1"), settled("a2")]})
    await build(client, ScriptedDecider([(TOOL, {}), (TOOL, {})]), ledger_config,
                submission_prefix="run-abc-").run()

    ids = [args["submission_id"] for args in client.args_for(TOOL)]
    assert ids and all(i.startswith("run-abc-") for i in ids)
    # The prefix must not cost uniqueness, which is what makes a SUBMISSION_CONFLICT
    # impossible rather than merely attributable.
    assert len(set(ids)) == len(ids)


# --- the decider seam -------------------------------------------------------


async def test_a_scripted_decider_marks_its_answers_as_scripted():
    decider = ScriptedDecider([(TOOL, {})])
    answer = await decider.propose(StateVector(), step=0)

    assert isinstance(answer, Proposal)
    assert answer.source == "scripted"
    assert answer.confidence is None, "a script has no confidence to offer and must not fake one"


async def test_an_exhausted_script_proposes_nothing():
    decider = ScriptedDecider([])
    assert await decider.propose(StateVector(), step=0) is None


async def test_a_scripted_decider_reports_what_is_left_and_rewinds():
    """``remaining`` drives progress output, and ``reset`` makes a retried run
    replay the same plan rather than a half-consumed one."""
    decider = ScriptedDecider([(TOOL, {}), (TOOL, {})])
    assert decider.remaining == 2
    await decider.propose(StateVector(), step=0)
    assert decider.remaining == 1
    decider.reset()
    assert decider.remaining == 2


async def test_a_script_names_arguments_from_the_advertised_schema():
    tools = [{
        "name": TOOL,
        "inputSchema": {"properties": {"block_type": {"type": "string"}}, "required": ["block_type"]},
    }]
    decider = ScriptedDecider([(TOOL, {"blocks": "dirt"})], tools)

    with pytest.raises(ScriptedArgumentError) as caught:
        await decider.propose(StateVector(), step=0)
    # A wrong argument name is dropped by the server rather than refused, so the
    # objective would run against the wrong thing and say nothing at all.
    assert "blocks" in str(caught.value)


async def test_a_correct_argument_name_passes():
    tools = [{"name": TOOL, "inputSchema": {"properties": {"block_type": {"type": "string"}}}}]
    decider = ScriptedDecider([(TOOL, {"block_type": "dirt"})], tools)

    answer = await decider.propose(StateVector(), step=0)
    assert isinstance(answer, Proposal)
    assert answer.arguments == {"block_type": "dirt"}


async def test_a_tool_the_server_never_advertised_is_left_to_the_server():
    """No local list means no local claim. The server's own refusal is the record."""
    decider = ScriptedDecider([("no_such_tool", {})])
    answer = await decider.propose(StateVector(), step=0)
    assert isinstance(answer, Proposal)


def test_an_unset_threshold_escalates_rather_than_assuming():
    """The policy that exists because both shipped checkpoints are over-confident."""
    escalation = confidence_gate(0.99, None)
    assert escalation is not None
    assert escalation.reason.startswith("laya.min_confidence is unset")


def test_a_threshold_is_enforced_in_both_directions():
    assert confidence_gate(0.30, 0.50) is not None
    assert confidence_gate(0.80, 0.50) is None


def test_a_missing_confidence_escalates_even_under_a_set_threshold():
    """A decider that omits its confidence has not earned the threshold."""
    assert confidence_gate(None, 0.50) is not None