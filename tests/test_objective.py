"""Protocol tests for the objective runner.

Each test names the rule from the server's contract that it pins down, because
these rules are not in any tool description and a model will not read them from
a prompt.
"""

from __future__ import annotations

import pytest
from conftest import FakeClient, accepted, pending, refused, settled

from harness.errors import BudgetExceeded, ObjectiveFailed
from harness.ledger import Ledger
from harness.mcp_client import ToolReply
from harness.objective import Objective, ObjectiveRunner


def no_evidence(action_id: str) -> ToolReply:
    """A settled output that asserts success but carries no observation."""
    return ToolReply(
        is_error=False,
        data={
            "state": "settled",
            "wakeReason": "settled",
            "actionId": action_id,
            "output": {"action": "collect_block", "result": {"kind": "task", "status": "succeeded"}},
        },
        notifications={},
    )


def make_runner(client, mcp_config, budget, ledger=None, clock=None, **kwargs) -> ObjectiveRunner:
    return ObjectiveRunner(client, mcp_config, budget, ledger, clock=clock or (lambda: 0.0), **kwargs)


def force_accepted(client: FakeClient, action_id: str) -> None:
    """Make the submission behave as if the wait timeout was omitted."""
    original = client.call

    async def call(tool, arguments=None, *, rationale, read_timeout_ms=None):
        if tool != "wait_for_action":
            client.calls.append((tool, dict(arguments or {})))
            return accepted(action_id)
        return await original(tool, arguments, rationale=rationale, read_timeout_ms=read_timeout_ms)

    client.call = call  # type: ignore[method-assign]


async def test_settled_on_initial_wait_needs_one_round_trip(mcp_config, budget):
    """A bounded initial wait returns the full result and releases the gate."""
    client = FakeClient(script={"collect_block": [settled("a1")]})
    runner = make_runner(client, mcp_config, budget)

    result = await runner.run(Objective(tool="collect_block", arguments={"block_name": "dirt"}))

    assert result.ok
    assert result.state == "settled"
    assert result.status == "succeeded"
    assert result.polls == 0, "a settled initial wait must not need a follow-up poll"
    assert [name for name, _ in client.calls] == ["collect_block"]


async def test_acceptance_is_not_a_physical_result(mcp_config, budget):
    """Omitting the wait yields a handle; the result still has to be retrieved."""
    client = FakeClient(script={"collect_block": [settled("a2")], "wait_for_action": [settled("a2")]})
    force_accepted(client, "a2")
    runner = make_runner(client, mcp_config, budget)

    result = await runner.run(Objective(tool="collect_block"))

    assert result.ok
    assert result.action_id == "a2"
    assert [name for name, _ in client.calls] == ["collect_block", "wait_for_action"]


async def test_expired_initial_wait_is_pending_and_still_owed(mcp_config, budget):
    """A bounded wait that expires returns `pending`, not `accepted`.

    Both mean admitted-and-unfinished, and both owe a retrieval. Polling only
    the `accepted` shape is how an objective silently ends as a bare handle.
    """
    client = FakeClient(script={"smelt_item": [pending("a10")], "wait_for_action": [settled("a10")]})
    runner = make_runner(client, mcp_config, budget)

    result = await runner.run(Objective(tool="smelt_item"))

    assert result.ok
    assert result.action_id == "a10"
    assert result.polls == 1
    assert [name for name, _ in client.calls] == ["smelt_item", "wait_for_action"]


async def test_pending_then_settled(mcp_config, budget):
    """Pending is progress, not failure: keep waiting."""
    client = FakeClient(script={"smelt_item": [pending("a3")], "wait_for_action": [settled("a3")]})
    runner = make_runner(client, mcp_config, budget)

    result = await runner.run(Objective(tool="smelt_item"))

    assert result.ok
    assert result.polls == 1
    assert client.args_for("wait_for_action")[0]["timeout_ms"] == mcp_config.poll_ms


async def test_polling_ceiling_fires(mcp_config, budget):
    """No unbounded polling: max_polls is a bound, and it fires."""
    client = FakeClient(script={"navigate": [pending("a4")], "wait_for_action": [pending("a4")]})
    runner = make_runner(client, mcp_config, budget)

    with pytest.raises(BudgetExceeded) as error:
        await runner.run(Objective(tool="navigate"))

    assert error.value.kind == "max_polls"


async def test_result_not_retrieved_is_released_then_the_objective_is_submitted(mcp_config, budget):
    """The owed result belongs to an earlier action, not to this objective.

    Retrieving it releases the gate. Reporting it as this objective's outcome
    would claim a physical result the harness never asked for.
    """
    client = FakeClient(
        script={
            "collect_block": [
                refused("RESULT_NOT_RETRIEVED", unretrievedActionId="owed"),
                settled("mine"),
            ],
            "wait_for_action": [settled("owed")],
        }
    )
    runner = make_runner(client, mcp_config, budget)

    result = await runner.run(Objective(tool="collect_block"))

    assert result.action_id == "mine", "the objective must be submitted for real, not proxied"
    assert [name for name, _ in client.calls] == [
        "collect_block",
        "wait_for_action",
        "collect_block",
    ]


async def test_gate_recovery_is_bounded(mcp_config, budget):
    """A permanently jammed gate terminates instead of looping."""
    client = FakeClient(
        script={
            "collect_block": [refused("RESULT_NOT_RETRIEVED", unretrievedActionId="owed")],
            "wait_for_action": [pending("owed")],
        }
    )
    runner = make_runner(client, mcp_config, budget, max_gate_recoveries=2)

    with pytest.raises((ObjectiveFailed, BudgetExceeded)):
        await runner.run(Objective(tool="collect_block"))

    assert len(client.args_for("collect_block")) <= 4, "recovery must be bounded"


async def test_result_not_retrieved_drains_the_owed_action_before_resubmitting(mcp_config, budget):
    """Half-recovering leaves the gate shut.

    The owed result is still pending on the first look, so the runner keeps
    waiting for it rather than resubmitting into a refusal that has nothing to
    do with the objective being submitted.
    """
    client = FakeClient(
        script={
            "collect_block": [refused("RESULT_NOT_RETRIEVED", unretrievedActionId="owed"), settled("mine")],
            "wait_for_action": [pending("owed"), settled("owed")],
        }
    )
    runner = make_runner(client, mcp_config, budget)

    result = await runner.run(Objective(tool="collect_block"))

    assert result.action_id == "mine"
    assert [name for name, _ in client.calls] == [
        "collect_block",
        "wait_for_action",
        "wait_for_action",
        "collect_block",
    ]


async def test_submission_identity_is_preserved_across_recovery(mcp_config, budget):
    """Recovery repeats the identical call, so the submission_id is carried.

    Changing either the id or the arguments turns a recovery into a
    SUBMISSION_CONFLICT, so identity has to survive the retry verbatim.
    """
    client = FakeClient(
        script={
            "collect_block": [refused("RESULT_NOT_RETRIEVED", unretrievedActionId="owed"), settled("mine")],
            "wait_for_action": [settled("owed")],
        }
    )
    runner = make_runner(client, mcp_config, budget)

    objective = Objective(tool="collect_block", arguments={"block_name": "dirt"})
    await runner.run(objective)

    submitted = client.args_for("collect_block")
    assert len(submitted) == 2, "the objective is submitted again after the gate opens"
    for args in submitted:
        assert args["submission_id"] == objective.submission_id
        assert args["wait_timeout_ms"] == mcp_config.initial_wait_ms
        assert args["block_name"] == "dirt", "recovery must repeat the call verbatim"


async def test_action_busy_waits_for_the_active_action_then_submits(mcp_config, budget):
    """ACTION_BUSY carries the active id; wait it out, then submit for real."""
    client = FakeClient(
        script={
            "collect_block": [refused("ACTION_BUSY", activeActionId="busy"), settled("mine")],
            "wait_for_action": [settled("busy")],
        }
    )
    runner = make_runner(client, mcp_config, budget)

    result = await runner.run(Objective(tool="collect_block"))

    assert result.action_id == "mine"
    assert [name for name, _ in client.calls] == [
        "collect_block",
        "wait_for_action",
        "collect_block",
    ]


async def test_action_busy_drains_the_active_action(mcp_config, budget):
    """Same rule as RESULT_NOT_RETRIEVED: wait it out, do not peek once."""
    client = FakeClient(
        script={
            "collect_block": [refused("ACTION_BUSY", activeActionId="busy"), settled("mine")],
            "wait_for_action": [pending("busy"), settled("busy")],
        }
    )
    runner = make_runner(client, mcp_config, budget)

    result = await runner.run(Objective(tool="collect_block"))

    assert result.action_id == "mine"
    assert [name for name, _ in client.calls] == [
        "collect_block",
        "wait_for_action",
        "wait_for_action",
        "collect_block",
    ]


async def test_action_busy_without_an_action_id_does_not_spin(mcp_config, budget):
    """Busy because the bot owns the body is not an action to wait out.

    There is nothing to drain, so resubmitting would burn the objective budget
    proving nothing.
    """
    client = FakeClient(script={"collect_block": [refused("ACTION_BUSY", error="body owned by movement")]})
    runner = make_runner(client, mcp_config, budget)

    with pytest.raises(ObjectiveFailed):
        await runner.run(Objective(tool="collect_block"))

    assert len(client.args_for("collect_block")) == 1


async def test_runtime_unavailable_does_not_retry_into_a_dead_connection(mcp_config, budget):
    """A dead bot connection is reported, not resubmitted against.

    Retrying would produce the same refusal every time and spend the whole
    objective budget proving nothing.
    """
    client = FakeClient(script={"collect_block": [refused("RUNTIME_UNAVAILABLE", error="bot disconnected")]})
    runner = make_runner(client, mcp_config, budget)

    with pytest.raises(ObjectiveFailed):
        await runner.run(Objective(tool="collect_block"))

    assert len(client.args_for("collect_block")) == 1


async def test_unresolvable_refusal_raises_rather_than_guessing(mcp_config, budget):
    """A refusal with no documented recovery raises instead of improvising."""
    client = FakeClient(script={"collect_block": [refused("INVALID_ARGUMENTS", error="bad block name")]})
    runner = make_runner(client, mcp_config, budget)

    with pytest.raises(ObjectiveFailed) as error:
        await runner.run(Objective(tool="collect_block"))

    assert error.value.state == "refused"


async def test_repeated_submission_uses_an_identical_call(mcp_config, budget):
    """Recovery depends on repeating the call verbatim, so arguments must match."""
    client = FakeClient(script={"collect_block": [settled("a5")]})
    runner = make_runner(client, mcp_config, budget)

    objective = Objective(tool="collect_block", arguments={"block_name": "dirt"})
    await runner.run(objective)

    submitted = client.args_for("collect_block")[0]
    assert submitted["block_name"] == "dirt"
    assert submitted["submission_id"] == objective.submission_id


async def test_pending_wait_does_not_trigger_a_resubmission(mcp_config, budget, clock):
    """A client timeout is not cancellation: poll the same action."""
    client = FakeClient(script={"collect_block": [pending("a6")], "wait_for_action": [settled("a6")]})
    runner = make_runner(client, mcp_config, budget, clock=clock)

    result = await runner.run(Objective(tool="collect_block"))

    assert result.ok
    assert len(client.args_for("collect_block")) == 1, "a pending wait must not cause a resubmission"


async def test_settled_without_evidence_is_unverified(mcp_config, budget):
    """Acceptance is not proof: a settled output with no observation is not success."""
    client = FakeClient(script={"collect_block": [no_evidence("a7")]})
    runner = make_runner(client, mcp_config, budget)

    result = await runner.run(Objective(tool="collect_block"))

    assert not result.ok
    assert result.evidence_ok is False


async def test_terminal_failure_is_reported_with_evidence(mcp_config, budget):
    """A partial outcome is terminal and keeps its evidence."""
    client = FakeClient(
        script={
            "collect_block": [
                settled(
                    "a8",
                    status="partial",
                    result={"kind": "task", "status": "partial", "error": "ran out of reach"},
                )
            ]
        }
    )
    runner = make_runner(client, mcp_config, budget)

    result = await runner.run(Objective(tool="collect_block"))

    assert not result.ok
    assert result.state == "partial"
    assert result.error == "ran out of reach"


async def test_ledger_row_is_written_with_the_outcome(mcp_config, budget, ledger_config):
    """The eval set and the baseline both come from this row, so it must exist."""
    with Ledger(ledger_config, "test-run") as ledger:
        client = FakeClient(script={"collect_block": [settled("a9")]})
        runner = make_runner(client, mcp_config, budget, ledger)
        await runner.run(Objective(tool="collect_block", arguments={"block_name": "dirt"}))

        rows = ledger.read_all()

    assert len(rows) == 1
    row = rows[0]
    assert row["objective_tool"] == "collect_block"
    assert row["objective_args"] == {"block_name": "dirt"}
    assert row["evidence_ok"] is True
    assert row["outcome"] == "settled:succeeded"
    assert row["run_id"] == "test-run"


async def test_failed_objective_is_recorded_before_it_raises(mcp_config, budget, ledger_config):
    """A refusal is a recorded outcome, not a lost one."""
    with Ledger(ledger_config, "test-run") as ledger:
        client = FakeClient(script={"collect_block": [refused("INVALID_ARGUMENTS", error="unknown block")]})
        runner = make_runner(client, mcp_config, budget, ledger)

        with pytest.raises(ObjectiveFailed):
            await runner.run(Objective(tool="collect_block"))

        rows = ledger.read_all()

    assert len(rows) == 1
    assert rows[0]["outcome"] == "refused"
    assert rows[0]["error"] == "unknown block", "the server's own words, not ours"