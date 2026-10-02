"""Tests for the model-backed decider.

The interesting behaviour here is refusal. A frontier model fills an object from
a description, so it invents tool names, misspells arguments, omits required
ones, and answers in prose when JSON was asked for. The server drops an argument
it does not recognise rather than refusing it, so every one of those is a run
that reports success having done the wrong thing. Each case below is a way that
silently wrong outcome is refused by name instead.
"""

from __future__ import annotations

import json
import os

import pytest
from conftest import FakeClient, settled, status

from harness.config import BudgetConfig, LlmConfig, McpConfig
from harness.decide import Completion, Proposal
from harness.goals import Goal, GoalBoard, GoalError, GoalSet
from harness.ledger import Ledger
from harness.llm import (
    LlmError,
    OpenRouterDecider,
    objective_tools,
    load_dotenv,
)
from harness.state import StateVector

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
KEY = "sk-or-v1-not-a-real-key-0123456789"

#: Every name this harness excludes from a decision, so the fixture mirrors the
#: live host's 27 objective / 10 read-or-control split rather than a sample of it.
#: A partial set would make the staleness check report the fixture's own gaps.
EXCLUDED = (
    "view_status", "view_blocks", "view_crafting_requirements", "view_frontier",
    "read_recent_events", "query_bot_data", "note_read", "wait_for_action",
    "cancel_foreground_action", "set_survival_policy",
)

TOOLS = [
    {
        "name": "collect_block",
        "description": "Collect one block type.",
        "input_schema": {
            "type": "object",
            "properties": {
                "block_name": {"type": "string", "description": "Exact block name."},
                "count": {"type": "integer", "minimum": 1, "maximum": 32, "default": 1},
                "submission_id": {"type": "string"},
            },
            "required": ["block_name"],
        },
    },
    {
        "name": "craft_item",
        "description": "Craft an item.",
        "input_schema": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string"},
                "count": {"type": "integer"},
                "submission_id": {"type": "string"},
            },
            "required": ["item_name"],
        },
    },
] + [{"name": name, "description": f"{name}.", "input_schema": {
    "type": "object", "properties": {"subject": {"type": "string"}}}} for name in EXCLUDED]


def config(**overrides) -> LlmConfig:
    defaults = {"model": MODEL, "api_key_env": "OPENROUTER_API_KEY", "temperature": 0.0}
    return LlmConfig(**{**defaults, **overrides})


def answer(**fields) -> dict:
    """A provider response carrying one model answer."""
    body = {"done": False, "tool": "collect_block", "arguments": {"block_name": "dirt"},
            "rationale": "no dirt yet"}
    body.update(fields)
    return {
        "model": MODEL,
        "choices": [{"message": {"content": json.dumps(body)}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.00012},
    }


def decider(tools=TOOLS, reply=None, **overrides) -> OpenRouterDecider:
    """A decider whose transport is a stub, so no test touches a network."""
    seen: list[dict] = []

    async def transport(payload: dict) -> dict:
        seen.append(payload)
        return reply if reply is not None else answer()

    instance = OpenRouterDecider(
        config(**overrides),
        tools,
        transport=transport,
        environ={"OPENROUTER_API_KEY": KEY},
    )
    instance.seen = seen  # type: ignore[attr-defined]
    return instance


# --- what it refuses ------------------------------------------------------

async def test_an_unnamed_model_is_refused():
    with pytest.raises(LlmError) as caught:
        OpenRouterDecider(config(model=""), TOOLS, transport=None, environ={})
    assert "llm.model" in str(caught.value)


async def test_no_advertised_input_schema_is_refused():
    with pytest.raises(LlmError) as caught:
        OpenRouterDecider(config(), [{"name": "view_status", "input_schema": {}}],
                          transport=None, environ={})
    assert "nothing a decision could choose" in str(caught.value)


async def test_a_missing_key_is_refused_by_name():
    instance = OpenRouterDecider(config(), TOOLS, environ={})

    with pytest.raises(LlmError) as caught:
        await instance.propose(StateVector(), step=0)
    assert "OPENROUTER_API_KEY" in str(caught.value)


async def test_the_key_never_appears_in_a_refusal():
    """The one secret in the process, and every error path is a way to leak it."""
    instance = OpenRouterDecider(config(), TOOLS, environ={"OPENROUTER_API_KEY": KEY})
    for bad in (
        {"error": {"code": 401, "message": f"bad key {KEY}"}},
        {"choices": []},
        {"choices": [{"message": {"content": "not json"}}]},
    ):
        async def transport(payload, bad=bad):
            return bad

        instance._transport = transport
        with pytest.raises(LlmError) as caught:
            await instance.propose(StateVector(), step=0)
        # The provider may echo the key back; the harness must not have sent it
        # in the request and must not repeat it. Assert on what we control.
        assert "Authorization" not in str(caught.value)


async def test_a_provider_error_is_refused_by_its_own_code():
    async def transport(payload):
        return {"error": {"code": 401, "message": "No auth credentials found"}}

    instance = OpenRouterDecider(config(), TOOLS, transport=transport,
                                 environ={"OPENROUTER_API_KEY": KEY}, backoff_s=0)
    with pytest.raises(LlmError) as caught:
        await instance.propose(StateVector(), step=0)
    assert "401" in str(caught.value)
    # An escalation would report a rate limit as a considered refusal to decide.
    assert "confidence" not in str(caught.value).lower()


async def test_a_response_without_choices_is_refused():
    async def transport(payload):
        return {"model": MODEL, "usage": {}}

    instance = OpenRouterDecider(config(), TOOLS, transport=transport,
                                 environ={"OPENROUTER_API_KEY": KEY})
    with pytest.raises(LlmError) as caught:
        await instance.propose(StateVector(), step=0)
    assert "choices[0]" in str(caught.value)


async def test_a_truncated_answer_is_refused_as_truncated_not_as_broken_json():
    """Observed live: this model spends ~300 tokens reasoning, then answers, and
    a 1024-token ceiling cut the object off mid-brace with finish_reason
    "length". Read as a parse error it would send the reader looking for a
    malformed response format instead of at the token ceiling."""
    async def transport(payload):
        return {"model": MODEL,
                "choices": [{"finish_reason": "length", "message": {"content": '{\n  "{\n'}}],
                "usage": {"completion_tokens": 1024}}

    instance = OpenRouterDecider(config(), TOOLS, transport=transport,
                                 environ={"OPENROUTER_API_KEY": KEY}, max_tokens=1024)
    with pytest.raises(LlmError) as caught:
        await instance.propose(StateVector(), step=0)
    assert "finish_reason=length" in str(caught.value)
    assert "1024-token ceiling" in str(caught.value)


async def test_an_answer_that_is_only_reasoning_is_refused_by_name():
    """Empty content with reasoning beside it: the model thought and never
    answered. Reported as what happened rather than as a missing field."""
    async def transport(payload):
        return {"model": MODEL,
                "choices": [{"finish_reason": "stop",
                             "message": {"content": "", "reasoning": "I should mine dirt."}}]}

    instance = OpenRouterDecider(config(), TOOLS, transport=transport,
                                 environ={"OPENROUTER_API_KEY": KEY})
    with pytest.raises(LlmError) as caught:
        await instance.propose(StateVector(), step=0)
    assert "empty content" in str(caught.value)
    assert "I should mine dirt." in str(caught.value)


async def test_prose_where_json_was_required_is_refused():
    async def transport(payload):
        return {"model": MODEL, "choices": [{"message": {"content": "I think we should mine."}}]}

    instance = OpenRouterDecider(config(), TOOLS, transport=transport,
                                 environ={"OPENROUTER_API_KEY": KEY})
    with pytest.raises(LlmError) as caught:
        await instance.propose(StateVector(), step=0)
    assert "did not return a JSON object" in str(caught.value)


async def test_an_object_wrapped_in_a_json_string_is_accepted():
    """Observed from this model on a live call: the answer arrives as the object
    encoded as a JSON *string*, which parses cleanly and then fails every field
    lookup. Unwrapped once, on purpose - a second layer is a bug to report."""
    async def transport(payload):
        inner = answer()["choices"][0]["message"]["content"]
        return {"model": MODEL, "choices": [{"message": {"content": json.dumps(inner)}}]}

    proposal = await OpenRouterDecider(
        config(), TOOLS, transport=transport,
        environ={"OPENROUTER_API_KEY": KEY}).propose(StateVector(), step=0)

    assert proposal.tool == "collect_block"


async def test_a_json_string_wrapping_something_else_is_refused():
    async def transport(payload):
        return {"model": MODEL,
                "choices": [{"message": {"content": json.dumps("still thinking")}}]}

    instance = OpenRouterDecider(config(), TOOLS, transport=transport,
                                 environ={"OPENROUTER_API_KEY": KEY})
    with pytest.raises(LlmError) as caught:
        await instance.propose(StateVector(), step=0)
    assert "JSON string containing" in str(caught.value)


async def test_arguments_that_are_not_an_object_are_refused():
    async def transport(payload):
        return answer(arguments=["dirt"])

    instance = OpenRouterDecider(config(), TOOLS, transport=transport,
                                 environ={"OPENROUTER_API_KEY": KEY})
    with pytest.raises(LlmError) as caught:
        await instance.propose(StateVector(), step=0)
    assert "not an object" in str(caught.value)


async def test_an_unadvertised_tool_is_refused():
    """A model names tools from memory. A name that is not advertised is not a
    tool, and submitting it would fail at the server with a message about a
    missing tool rather than about the objective."""
    instance = decider(reply=answer(tool="mine_everything"))

    with pytest.raises(LlmError) as caught:
        await instance.propose(StateVector(), step=0)
    assert "mine_everything" in str(caught.value)


async def test_an_invented_argument_name_is_refused():
    """The same class of error the scripted decider guards, and the one a model
    commits most: `block_typo` is dropped by the server, not rejected, so the
    objective runs against the wrong thing and reports success."""
    instance = decider(reply=answer(arguments={"block_typo": "dirt"}))

    with pytest.raises(LlmError) as caught:
        await instance.propose(StateVector(), step=0)
    assert "block_typo" in str(caught.value)


async def test_an_omitted_required_argument_is_refused():
    instance = decider(reply=answer(arguments={}))

    with pytest.raises(LlmError) as caught:
        await instance.propose(StateVector(), step=0)
    assert "block_name" in str(caught.value)


async def test_the_model_is_never_offered_a_read_or_a_control():
    """The claim the whole baseline rests on, checked rather than asserted.

    A model given `wait_for_action` will eventually call it, and the runner
    already owns that action; a model given `view_status` spends an objective
    slot re-deriving the state it was just handed.
    """
    instance = decider()
    await instance.propose(StateVector(), step=0)

    schema = instance.seen[0]["response_format"]["json_schema"]["schema"]
    allowed = schema["properties"]["tool"]["enum"]
    assert allowed == ["collect_block", "craft_item"]
    assert "wait_for_action" not in allowed
    assert "view_status" not in allowed
    assert set(allowed) == set(instance.choices)


async def test_the_model_is_never_offered_a_submission_id():
    """D15's conflict is unrecoverable because the id identifies one submission.
    A model choosing its own would let it collide with an earlier call, and the
    refusal that follows names no action, so the running one cannot be found."""
    instance = decider()
    await instance.propose(StateVector(), step=0)

    catalogue = instance.seen[0]["messages"][1]["content"]
    assert "submission_id" not in json.loads(catalogue)["tools"][0]["arguments"]


# --- what it produces -----------------------------------------------------

async def test_a_valid_answer_becomes_a_proposal_naming_its_model():
    proposal = await decider().propose(StateVector(), step=0)

    assert isinstance(proposal, Proposal)
    assert proposal.tool == "collect_block"
    assert proposal.arguments == {"block_name": "dirt"}
    assert proposal.rationale == "no dirt yet"
    assert proposal.source == f"openrouter:{MODEL}"


async def test_confidence_is_null_and_the_gate_is_not_applied():
    """D16. A frontier model has no calibrated confidence to offer, so the
    honest value is None and the Laya gate does not run. A hard-coded 1.0 would
    unblock this path while writing a certainty no measurement supports - and
    would pass any threshold fitted later, reporting calibration that never
    happened."""
    proposal = await decider().propose(StateVector(), step=0)

    assert proposal.confidence is None


async def test_done_is_a_recordable_answer_and_not_an_absence_of_one():
    """`None` means a spent script; "the model says we are finished" is an answer.

    Measured on 2026-10-02: a run collected its logs correctly, the model then
    reported nothing left to do, and the loop stopped cleanly - with a ledger
    holding one successful objective and *no record of the decision that ended
    it*. The report said the bot held no pickaxe and the ledger could not say why
    anyone believed otherwise.
    """
    instance = decider(reply=answer(done=True, tool="collect_block", rationale="nothing left"))

    claim = await instance.propose(StateVector(), step=0)
    assert isinstance(claim, Completion)
    assert claim.reason == "nothing left"
    assert len(instance.calls) == 1
    assert instance.calls[0].cost_usd == pytest.approx(0.00012)


async def test_a_completion_names_the_goal_it_believes_is_met():
    """Which goal the model thinks is done is the whole content of the claim."""
    instance = decider(
        reply=answer(done=True, goal="furnace", rationale="the furnace is crafted")
    )
    instance.goals = GoalBoard.of(GoalSet(("furnace",)), max_attempts=3)
    claim = await instance.propose(StateVector(), step=0)
    assert isinstance(claim, Completion)
    assert claim.goal == "furnace"
    assert claim.reason == "the furnace is crafted"


async def test_a_goal_set_accepts_the_obvious_construction():
    """`GoalSet(("furnace",))` is what a call site naturally writes.

    Without coercion it built a set whose `items` raised `AttributeError: 'str'
    object has no attribute 'item'`. A type annotation nobody checks is a comment.
    """
    assert GoalSet(("furnace", "white_bed")).items == ("furnace", "white_bed")
    assert GoalSet((Goal("furnace", 2),)).items == ("furnace",)
    with pytest.raises(GoalError):
        GoalSet((3,))


async def test_a_spent_script_still_returns_none():
    """The two must stay distinguishable, or the reason for the change is lost."""
    from harness.decide import ScriptedDecider

    spent = ScriptedDecider([], [{"name": "collect_block"}])
    assert await spent.propose(StateVector(), step=0) is None


async def test_cost_is_measured_per_call():
    instance = decider()
    await instance.propose(StateVector(), step=0)
    await instance.propose(StateVector(), step=1)

    assert len(instance.calls) == 2
    assert sum(c.prompt_tokens for c in instance.calls) == 200
    assert sum(c.cost_usd for c in instance.calls) == pytest.approx(0.00024)


async def test_a_provider_that_reports_no_cost_says_so_rather_than_guessing_zero():
    """Zero is a number, and a run that reports $0.00 because the provider
    omitted the field reads as free."""
    instance = decider(reply={"model": MODEL, "choices": answer()["choices"]})

    await instance.propose(StateVector(), step=0)
    assert instance.calls[0].cost_usd is None
    assert instance.calls[0].prompt_tokens == 0


async def test_a_free_model_reporting_zero_cost_is_recorded_as_zero():
    """The counterpart: this model does report `cost: 0`, and that is a
    measurement. It is recorded, not discarded as missing."""
    free = answer()
    free["usage"]["cost"] = 0
    instance = decider(reply=free)

    await instance.propose(StateVector(), step=0)
    assert instance.calls[0].cost_usd == 0.0


async def test_an_overloaded_provider_is_retried_and_the_attempts_are_counted():
    """Measured, not hypothetical: the free provider behind this model returned
    503 on three of four probes. Without a retry a run ends on a condition the
    next attempt would have passed."""
    attempts = {"n": 0}

    async def transport(payload):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return {"error": {"code": 503, "message": "Service temporarily overloaded",
                              "metadata": {"error_type": "provider_overloaded"}}}
        return answer()

    instance = OpenRouterDecider(config(), TOOLS, transport=transport,
                                 environ={"OPENROUTER_API_KEY": KEY},
                                 backoff_s=0)
    proposal = await instance.propose(StateVector(), step=0)

    assert proposal.tool == "collect_block"
    assert instance.calls[0].attempts == 3


async def test_a_bad_key_is_not_retried():
    """A wrong key fails identically every time, so retrying only delays the
    report of a real problem."""
    attempts = {"n": 0}

    async def transport(payload):
        attempts["n"] += 1
        return {"error": {"code": 401, "message": "No auth credentials found"}}

    instance = OpenRouterDecider(config(), TOOLS, transport=transport,
                                 environ={"OPENROUTER_API_KEY": KEY})
    with pytest.raises(LlmError) as caught:
        await instance.propose(StateVector(), step=0)

    assert attempts["n"] == 1
    assert "401" in str(caught.value)


async def test_an_overload_that_never_clears_is_reported_with_its_attempt_count():
    attempts = {"n": 0}

    async def transport(payload):
        attempts["n"] += 1
        return {"error": {"code": 503, "message": "Service temporarily overloaded"}}

    instance = OpenRouterDecider(config(), TOOLS, transport=transport,
                                 environ={"OPENROUTER_API_KEY": KEY},
                                 max_attempts=2, backoff_s=0)
    with pytest.raises(LlmError) as caught:
        await instance.propose(StateVector(), step=0)

    assert attempts["n"] == 2
    assert "after 2 attempts" in str(caught.value)


async def test_the_state_reaches_the_model_and_the_key_does_not():
    instance = decider()
    await instance.propose(StateVector(), step=0)

    request = json.loads(instance.seen[0]["messages"][1]["content"])
    assert request["step"] == 0
    assert "health" in request["state"]
    assert KEY not in json.dumps(instance.seen[0])


async def test_reads_are_classified_away_from_objectives():
    chosen = objective_tools(TOOLS)
    assert sorted(chosen) == ["collect_block", "craft_item"]


async def test_a_tool_the_model_cannot_reach_is_reported_rather_than_dropped():
    """Nothing here should fire against the live host - all 37 tools advertise
    arguments. A tool that does not is either a real objective the model cannot
    reach or an advertisement read wrongly, and both look identical to a tool
    the server does not have."""
    instance = decider(tools=TOOLS + [{"name": "brand_new_tool", "input_schema": {}}])

    assert instance.unreachable == ["brand_new_tool"]
    assert "brand_new_tool" not in instance.choices


async def test_a_read_or_control_the_server_dropped_is_reported_as_dead_weight():
    """A renamed or removed tool left in the exclusion set is harmless until the
    name comes back, at which point it hides a real read from the model."""
    healthy = decider()
    assert healthy.stale_read_controls == []
    assert healthy.unreachable == []

    shrunk = decider(tools=[t for t in TOOLS if t["name"] != "query_bot_data"])
    assert shrunk.stale_read_controls == ["query_bot_data"]


# --- the seam, end to end -------------------------------------------------

async def test_the_loop_runs_with_a_model_behind_it(ledger_config, budget, clock):
    """The seam is the deliverable: same loop, same protocol rules, one line
    changed, and the rows come back marked as a model's work."""
    from harness.loop import RunLoop
    from harness.objective import ObjectiveRunner
    from harness.state import StateReader

    client = FakeClient(script={
        "view_status": [status()],
        "collect_block": [settled("a1")],
    })

    replies = [answer(), answer(done=True, tool="collect_block", rationale="nothing left")]

    async def transport(payload):
        return replies.pop(0)

    instance = OpenRouterDecider(config(), TOOLS, transport=transport,
                                 environ={"OPENROUTER_API_KEY": KEY})

    with Ledger(ledger_config, "run-llm") as ledger:
        loop = RunLoop(
            StateReader(client),
            ObjectiveRunner(client, McpConfig(), budget, ledger),
            instance,
            ledger=ledger,
            clock=clock,
        )
        report = await loop.run(max_steps=8)

    assert report.ok, report.detail
    assert report.stop_reason == "plan_exhausted"
    rows = [row for row in ledger.read_all() if row.get("objective_tool")]
    assert len(rows) == 1
    # The source and the confidence are what tell a model row from a scripted
    # one a month from now, so both are asserted rather than assumed.
    assert rows[0]["question"]["source"].startswith("openrouter:")
    assert rows[0]["answer_confidence"] is None
    assert rows[0]["objective_tool"] == "collect_block"
    assert len(instance.calls) == 2


# --- .env -----------------------------------------------------------------

def test_dotenv_does_not_overwrite_the_environment(tmp_path, monkeypatch):
    """A key exported in a shell profile is deliberate; a stale .env silently
    replacing it produces a run against the wrong account with no explanation."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-the-shell")
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=from-the-file\n")

    load_dotenv(env)

    assert os.environ["OPENROUTER_API_KEY"] == "from-the-shell"


def test_dotenv_reads_quotes_comments_and_export(tmp_path, monkeypatch):
    monkeypatch.delenv("A", raising=False)
    monkeypatch.delenv("B", raising=False)
    monkeypatch.delenv("C", raising=False)
    env = tmp_path / ".env"
    env.write_text('# a comment\n\nA=plain\nexport B="quoted"\nC=\n')

    loaded = load_dotenv(env)

    assert loaded["A"] == "plain"
    assert loaded["B"] == "quoted"
    assert loaded["C"] == ""
    assert os.environ["B"] == "quoted"


def test_a_missing_dotenv_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("A", raising=False)
    assert load_dotenv(tmp_path / "absent") == {}