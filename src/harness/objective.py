"""The objective runner: a protocol state machine around one foreground action.

This is the component that makes the harness an application rather than a
prompt. A frontier model improvising this protocol deadlocks or double-submits,
because the rules are not in any tool description - "retrieve the previous
result before submitting the next objective" lives in the server instructions,
not in the schema.

Rules encoded here, each from the server's contract:

* Every foreground call needs a unique ``submission_id``. Retry the *same*
  arguments with the *same* id to recover a lost reply; change either and it is
  refused.
* One logical objective at a time. A refusal never queues. A second submission
  before retrieval is refused with the preceding action id.
* A timeout is not cancellation. The bot keeps working.
* Acceptance is not a physical result. ``partial``, ``failed`` and ``cancelled``
  are terminal, and a missing evidence leg is unverified, never success.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .config import BudgetConfig, McpConfig
from .errors import BudgetExceeded, ObjectiveFailed, UnverifiedRead
from .ledger import DecisionRow, Ledger
from .mcp_client import McpClient, ToolReply

#: Terminal states that are not success. All keep their available evidence.
FAILURE_STATES = frozenset({"partial", "failed", "cancelled", "runtime_failure"})

#: Refusals whose cause is another action, which waiting on can clear.
GATE_REFUSALS = frozenset({"ACTION_BUSY", "RESULT_NOT_RETRIEVED"})

#: The one body owner state that admits new work, from the server's own union in
#: ``action-runner.ts``: ``"idle" | "foreground" | "yielding" | "takeover"``.
FREE_BODY_OWNER = "idle"

#: Return values from ``_release_gate``. Only these two: a gate either opens or
#: it does not, and a refusal that cannot be drained stops the objective.
GATE_OPEN = "open"
GATE_STUCK = "stuck"


@dataclass
class ObjectiveResult:
    """What an objective settled with, in the server's vocabulary."""

    tool: str
    submission_id: str
    action_id: str | None
    state: str
    status: str | None
    output: dict[str, Any] | None
    evidence_ok: bool
    polls: int
    duration_ms: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.state == "settled" and self.status == "succeeded" and self.evidence_ok


@dataclass(frozen=True)
class SeenState:
    """What the decider saw when it chose this objective.

    Carried into the ledger row rather than recorded separately, because the
    Phase 2 baseline and the Phase 3 eval both need the state and the outcome in
    one row: a state joined to an outcome from a different step is a correlation
    the evidence does not support.
    """

    vector: dict[str, Any]
    state_hash: str


@dataclass
class Objective:
    """One thing to ask the bot to do."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    submission_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    #: A dropped submission reply is recovered by repeating this call verbatim.
    metadata: dict[str, Any] = field(default_factory=dict)


class ObjectiveRunner:
    """Drives one objective to a terminal state, and reports its evidence."""

    def __init__(
        self,
        client: McpClient,
        config: McpConfig,
        budget: BudgetConfig,
        ledger: Ledger | None = None,
        clock: Any = time.monotonic,
        max_gate_recoveries: int = 2,
    ) -> None:
        self.client = client
        self.config = config
        self.budget = budget
        self.ledger = ledger
        self.clock = clock
        #: Bounded because "wait, retry, wait, retry" can otherwise loop while
        #: each pass looks like progress. Each pass must open a gate that was
        #: shut, so a run that exhausts this is evidence the gate is jammed.
        self.max_gate_recoveries = max_gate_recoveries

    async def run(self, objective: Objective, seen: SeenState | None = None) -> ObjectiveResult:
        """Submit, wait, verify. Never raises on a failed objective.

        Returns the result and records a ledger row; the caller decides what to
        do next. A refusal the runner cannot resolve raises, because continuing
        would mean guessing at the protocol.

        ``seen`` is what the decider saw when it chose this objective. Without
        it the row still records what was submitted and what came of it, but not
        why, which is the half of the ledger that cannot be reconstructed later.
        """
        started = self.clock()
        row = DecisionRow(
            state_vector=(
                seen.vector if seen else {"objective": objective.tool, "arguments": objective.arguments}
            ),
            state_hash=seen.state_hash if seen else "",
            question={"kind": "objective", "tool": objective.tool},
            objective_tool=objective.tool,
            objective_args=objective.arguments,
            run_id=self.ledger.run_id if self.ledger else "",
        )

        if seen is not None:
            goal = objective.metadata.get("goal")
            row.question = {
                "kind": "objective",
                "tool": objective.tool,
                "rationale": objective.rationale,
                "source": objective.metadata.get("source", "unknown"),
                # Under `question` rather than as its own column because the goal
                # is the answer's *subject* - what the decider was trying to
                # achieve with this tool - and a new column would be a schema
                # change for every reader before any run has produced one.
                **({"goal": goal} if isinstance(goal, str) else {}),
            }
            row.options = list(objective.metadata.get("options", ()))
            row.answer = objective.tool
            row.answer_confidence = objective.metadata.get("confidence")

        reply = await self._submit(objective)

        # A gate refusal means an *earlier* action owes its result. Retrieving it
        # releases the gate; it does not satisfy this objective. So the recovery
        # drains the blocking action and then submits for real, rather than
        # reporting someone else's action as our own - which would claim a
        # physical outcome the harness never asked for.
        recoveries = 0
        while reply.state == "refused" and recoveries < self.max_gate_recoveries:
            outcome = await self._release_gate(objective, reply)
            if outcome is not GATE_OPEN:
                break
            recoveries += 1
            row.error = f"gate released ({reply.refusal_code}) then resubmitted"
            reply = await self._submit(objective)

        if reply.state == "refused":
            row.outcome = "refused"
            row.error = reply.data.get("error")
            if recoveries:
                row.error = f"{row.error} (after {recoveries} gate recoveries)"
            row.duration_ms = self._elapsed(started)
            self._record(row)
            raise ObjectiveFailed(objective.tool, "refused", reply.data)

        # Three states all mean "admitted, not finished": `accepted` when the
        # wait was omitted, `pending` when a bounded wait expired. Both carry an
        # action id and both owe the caller a retrieval. Treating only one of
        # them as admitted is how an objective silently ends as a handle.
        polls = 0
        if reply.state in ("accepted", "pending"):
            reply, polls = await self._poll(objective, reply.action_id)

        result = self._to_result(objective, reply, polls, started)

        row.outcome = f"{result.state}:{result.status}" if result.status else result.state
        row.evidence_ok = result.evidence_ok
        row.duration_ms = result.duration_ms
        row.error = result.error
        self._record(row)
        return result

    async def _submit(self, objective: Objective) -> ToolReply:
        """Submit with a bounded initial wait.

        The bounded wait is preferred over omitting it: completion returns the
        full settled result in this call and releases the gate, which is one
        round trip instead of two.
        """
        arguments = dict(objective.arguments)
        arguments["submission_id"] = objective.submission_id
        arguments["wait_timeout_ms"] = self.config.initial_wait_ms

        reply = await self.client.call(
            objective.tool,
            arguments,
            rationale=objective.rationale or f"Harness objective: {objective.tool}",
        )
        # Submitting a foreground objective and getting a direct payload back
        # means the tool or the server changed shape. Interpreting that as a
        # protocol state would report the world as refusing or admitting work
        # that was never submitted.
        state = reply.require_state()
        # A lost submission reply is recovered by repeating the identical call:
        # the server returns the original identity rather than starting again.
        if state in ("accepted", "pending") and reply.action_id is None:
            raise UnverifiedRead(objective.tool, f"{state} without an actionId")
        return reply

    async def _release_gate(self, objective: Objective, reply: ToolReply) -> object:
        """Drain whatever is blocking the gate, so the caller can submit for real.

        Returns :data:`GATE_OPEN` when the gate is now open and the objective
        should be submitted again, and :data:`GATE_STUCK` for every refusal that
        cannot be recovered by waiting. There is deliberately no third outcome:
        a gate either opens or it does not, and a refusal that cannot be drained
        should stop the objective rather than consume the remaining retries.

        Each recovery drains the blocking action to a terminal state rather than
        taking one look at it. Half-recovering leaves the gate shut, and every
        subsequent submission is then refused for a reason that has nothing to
        do with the objective being submitted.
        """
        code = reply.refusal_code

        if code == "RESULT_NOT_RETRIEVED":
            owed = reply.data.get("unretrievedActionId")
            if not isinstance(owed, str):
                raise UnverifiedRead(objective.tool, "RESULT_NOT_RETRIEVED without an id")
            await self._poll(objective, owed)
            return GATE_OPEN

        if code == "ACTION_BUSY":
            active = reply.data.get("activeActionId")
            if isinstance(active, str):
                await self._poll(objective, active)
                return GATE_OPEN
            # ACTION_BUSY with no action id is the survival reflex holding the
            # body: `hostile_reflex` fighting something the model never asked
            # about. There is no action to drain, but the ownership is finite -
            # the reflex ends and returns the body - so waiting is the correct
            # response. Treating it as terminal instead means a zombie at night
            # aborts every objective for as long as it lives, which is most of
            # the night.
            return GATE_OPEN if await self._await_free_body(objective) else GATE_STUCK

        if code == "RUNTIME_UNAVAILABLE":
            # The bot connection ended. The world effects of whatever was
            # interrupted are unknown, and retrying would produce the same
            # refusal every time while spending the objective budget.
            return GATE_STUCK

        return GATE_STUCK

    async def _poll(
        self, objective: Objective, action_id: str | None
    ) -> tuple[ToolReply, int]:
        """Wait on an admitted action until it settles or a bound fires.

        Bounded twice: by wall clock, because a stack smelt legitimately runs
        past ten minutes, and by wait count, so a runtime that answers
        ``pending`` forever terminates instead of spinning.
        """
        if action_id is None:
            raise UnverifiedRead(objective.tool, "cannot poll without an actionId")

        deadline = self.clock() + self.budget.objective_ms / 1000
        polls = 0

        while True:
            if self.clock() > deadline:
                raise BudgetExceeded(
                    "objective_ms",
                    f"{objective.tool} did not settle within {self.budget.objective_ms} ms",
                )
            if polls >= self.config.max_polls:
                raise BudgetExceeded(
                    "max_polls", f"{objective.tool} still pending after {polls} waits"
                )

            reply = await self._wait_once(objective, action_id)
            polls += 1

            if reply.state != "pending":
                return reply, polls

    async def _await_free_body(self, objective: Objective) -> bool:
        """Wait for the survival reflex to return the body.

        Read from ``view_status`` rather than retried blindly, because the server
        already reports who owns the body and re-submitting into a reflex is what
        produced the refusal in the first place. This is a deterministic wait (D7):
        the server does the observing, and no model is consulted about whether a
        fight is over.

        Bounded twice - by wall clock, because a fight is legitimately slow, and
        by wait count, because a reflex that never releases must not spin here
        forever. Returns whether the body came free.
        """
        deadline = self.clock() + self.budget.gate_wait_ms / 1000
        polls = 0

        while True:
            if self.clock() > deadline:
                return False
            if polls >= self.config.max_polls:
                return False
            if await self._body_is_free():
                return True
            polls += 1
            await asyncio.sleep(self.config.poll_ms / 1000)

    async def _body_is_free(self) -> bool:
        """Is the body unowned right now?"""
        reply = await self.client.call(
            "view_status", {}, rationale="Check whether the body is free to accept work"
        )
        return _is_free_body(reply)

    async def _wait_once(self, objective: Objective, action_id: str) -> ToolReply:
        reply = await self.client.call(
            "wait_for_action",
            {"action_id": action_id, "timeout_ms": self.config.poll_ms},
            rationale=objective.rationale or f"Retrieve result for {objective.tool}",
        )
        # `wait_for_action` is a control tool rather than a foreground one, but
        # it is the one control tool that speaks the submission envelope, so the
        # envelope is required here for the same reason it is on submit.
        reply.require_state()
        return reply

    def _to_result(
        self,
        objective: Objective,
        reply: ToolReply,
        polls: int,
        started: float,
    ) -> ObjectiveResult:
        """Turn a protocol envelope into a verdict.

        Verification is deliberately conservative: an output with no evidence is
        unverified, not successful.
        """
        status = reply.result_status
        evidence_ok = self._evidence_ok(reply)
        error = None

        if reply.state == "settled":
            if status in (None, "succeeded"):
                state = "settled" if status == "succeeded" else "unverified"
            else:
                state = status
                if status in FAILURE_STATES:
                    error = _error_text(reply)
            if status is None:
                error = "settled output carried no result status"
        elif reply.state == "storage_failed":
            state = "storage_failed"
            error = reply.data.get("error")
            evidence_ok = False
        else:
            state = reply.state
            error = reply.data.get("error")
            evidence_ok = False

        return ObjectiveResult(
            tool=objective.tool,
            submission_id=objective.submission_id,
            action_id=reply.action_id,
            state=state,
            status=status,
            output=reply.output,
            evidence_ok=evidence_ok,
            polls=polls,
            duration_ms=self._elapsed(started),
            error=error,
        )

    def _evidence_ok(self, reply: ToolReply) -> bool:
        """Does a settled output carry evidence rather than just a status?

        A dig is a request, not a receipt. Acceptance and a bare status are not
        proof of a physical outcome.
        """
        output = reply.output
        if output is None:
            return False
        result = output.get("result")
        if not isinstance(result, dict):
            return False
        if result.get("kind") == "runtime_failure":
            return False
        # The server's request block carries baseline, checkpoint and the
        # completion condition - the observation it based the result on.
        request = output.get("request")
        if not isinstance(request, dict):
            return False
        return request.get("evidence") is not None or "evidence" in result

    def _elapsed(self, started: float) -> int:
        return round((self.clock() - started) * 1000)

    def _record(self, row: DecisionRow) -> None:
        if self.ledger is not None:
            self.ledger.record(row)


def _error_text(reply: ToolReply) -> str | None:
    result = reply.result
    if result is None:
        return None
    error = result.get("error")
    return error if isinstance(error, str) else None


def _is_free_body(reply: ToolReply) -> bool:
    """Does this status read report an unowned body?

    ``owner: "idle"`` with no active action, which is exactly what the server
    reports as ``busy: false`` - bound from the owner union in
    ``action-runner.ts`` rather than assumed.

    A read that cannot be parsed is **not** freedom. Guessing that the body is
    free submits into a reflex that still owns it and then reports the resulting
    refusal as an answer, which is the one outcome this wait exists to prevent.
    """
    result = reply.result
    if result is None:
        return False
    situation = result.get("situation")
    if not isinstance(situation, dict):
        return False
    activity = situation.get("activity")
    if not isinstance(activity, dict):
        return False
    return (
        activity.get("owner") == FREE_BODY_OWNER
        and activity.get("activeAction") is None
    )