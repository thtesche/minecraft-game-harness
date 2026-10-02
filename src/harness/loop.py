"""The deterministic decision loop.

Everything about *order* is decided here and nothing about *content*. Each step:
read the world, ask the decider what to do, run it, record what came of it. That
separation is the whole point of D7 - the protocol rules are in code rather than
in a prompt, so a frontier model improvising them cannot deadlock or
double-submit - and it is also what makes the loop testable without a model.

Three conditions stop the loop, and each is recorded with the state that caused
it:

* the decider has nothing left to propose - a finished plan, not a failure
* the decider escalates - refusing to guess is the correct outcome of an
  uncalibrated decider, and there is nowhere to escalate to yet
* the world cannot be read - deciding from an invented state is worse than not
  deciding, so the state is re-read a bounded number of times and then the run
  ends

A death is deliberately not one of them. The bot dying is a fact about the
objective that was running, and the loop absorbs it, records it against that
objective, and continues.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .decide import Completion, Decider, Escalation, Proposal
from .errors import BudgetExceeded, HarnessError, ObjectiveFailed
from .goals import GoalBoard
from .ledger import DecisionRow, Ledger
from .objective import Objective, ObjectiveResult, ObjectiveRunner, SeenState
from .state import StateReader, StateVector

#: Why the loop stopped. Named so a caller branches on a cause rather than on a
#: message, and so a report is comparable between runs.
STOP_PLAN_EXHAUSTED = "plan_exhausted"
STOP_STEP_LIMIT = "step_limit"
STOP_ESCALATED = "escalated"
STOP_UNVERIFIED_STATE = "unverified_state"
STOP_REFUSED = "refused"
STOP_BUDGET = "budget"
STOP_ERROR = "error"
#: The decider asked for a goal past ``budget.max_attempts_per_goal``. Named
#: because the alternative - running it anyway - spends a real objective, and on
#: the model path a real decision, on something the harness has already declared
#: it will stop doing.
STOP_GOAL_ATTEMPTS = "goal_attempts_exhausted"
#: The decider named a goal that is not in the run's goal set. A refusal rather
#: than a re-read: it means the decider is answering a different question than
#: the scenario asked, and counting that as progress would corrupt the very
#: measurement the scenario exists to make.
STOP_GOAL_UNKNOWN = "goal_unknown"
#: The bot's Minecraft connection is closed. Named, and checked *before* the
#: unverified-read retries, because retrying a socket the server has already said
#: is closed cannot succeed and each attempt costs a round trip: mine-ai-mcp
#: answers ``/health`` 503 with ``minecraft.connected: false``, and
#: ``src/server/runtime-host.ts:143`` states that "a new Minecraft connection
#: requires an explicit service restart". So there is nothing for the harness to
#: wait for, and the report says who has to act rather than reporting a parse
#: failure three attempts later.
STOP_RUNTIME_UNAVAILABLE = "runtime_unavailable"

#: Stop reasons that mean "the run did not finish its work", as opposed to
#: finishing it. A plan that ran out is a success; a refusal is not.
INCOMPLETE_STOPS = frozenset(
    {
        STOP_STEP_LIMIT,
        STOP_ESCALATED,
        STOP_UNVERIFIED_STATE,
        STOP_REFUSED,
        STOP_BUDGET,
        STOP_ERROR,
        STOP_GOAL_ATTEMPTS,
        STOP_GOAL_UNKNOWN,
        STOP_RUNTIME_UNAVAILABLE,
    }
)


@dataclass
class StepReport:
    """One pass of the loop, successful or not."""

    index: int
    tool: str
    state_hash: str
    result: ObjectiveResult | None = None
    death_cause: str | None = None
    error: str | None = None
    #: The goal this objective served, when the run had goals.
    goal: str | None = None


@dataclass
class LoopReport:
    """The whole run, in enough detail to judge it without the logs."""

    steps: list[StepReport] = field(default_factory=list)
    stop_reason: str = STOP_PLAN_EXHAUSTED
    detail: str = ""
    #: How many times a death was absorbed, so a run that lost the bot four times
    #: is visibly different from one that did not.
    deaths_absorbed: int = 0
    #: Re-reads spent on a state that would not verify.
    unverified_reads: int = 0
    #: Attempts made at each goal, when the run had goals. The per-goal
    #: counterpart of ``steps``, and what makes "calls per goal" readable off a
    #: report instead of inferred from arguments.
    goal_attempts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.stop_reason == STOP_PLAN_EXHAUSTED

    @property
    def objectives_succeeded(self) -> int:
        return sum(1 for step in self.steps if step.result is not None and step.result.ok)

    @property
    def verified_evidence(self) -> int:
        return sum(1 for step in self.steps if step.result is not None and step.result.evidence_ok)

    def summary(self) -> dict[str, Any]:
        summary = {
            "stopReason": self.stop_reason,
            "detail": self.detail,
            "steps": len(self.steps),
            "succeeded": self.objectives_succeeded,
            "verifiedEvidence": self.verified_evidence,
            "deathsAbsorbed": self.deaths_absorbed,
            "unverifiedReads": self.unverified_reads,
        }
        if self.goal_attempts:
            summary["goalAttempts"] = dict(self.goal_attempts)
        return summary


class RunLoop:
    """Runs objectives in sequence, one trusted state read before each.

    ``on_step`` is called after every step with the report so far. It exists for
    progress output: a real run takes minutes per objective, and a loop that
    reports only at exit is indistinguishable from a hung process.
    """

    def __init__(
        self,
        reader: StateReader,
        runner: ObjectiveRunner,
        decider: Decider,
        *,
        ledger: Ledger | None = None,
        clock: Callable[[], float] = time.monotonic,
        submission_prefix: str = "",
        #: Re-reads allowed on a state that will not verify before the run ends.
        #: One is enough to ride out a dropped connection and low enough that a
        #: contract change is reported on the second failure instead of being
        #: retried through a long run.
        max_unverified_reads: int = 1,
        on_step: Callable[[StepReport], None] | None = None,
        #: The run's goal board, or ``None`` for a run with no goals. Owned here
        #: rather than by the decider because the loop is what learns the outcome:
        #: progress is recorded from what the server returned, never from what the
        #: model intended.
        goals: GoalBoard | None = None,
    ) -> None:
        self.reader = reader
        self.runner = runner
        self.decider = decider
        self.ledger = ledger
        self.clock = clock
        self.submission_prefix = submission_prefix
        self.max_unverified_reads = max_unverified_reads
        self.on_step = on_step
        self.goals = goals
        #: Handed over rather than read by the decider from somewhere else, so a
        #: decider that ignores it is visibly ignoring it.
        self.decider.goals = goals

    async def run(self, *, max_steps: int = 32) -> LoopReport:
        report = LoopReport()
        started = self.clock()

        for step in range(max_steps):
            vector, unavailable = await self._trusted_state(report)
            if vector is None:
                if unavailable is not None:
                    report.stop_reason = STOP_RUNTIME_UNAVAILABLE
                    report.detail = unavailable
                    return report
                report.stop_reason = STOP_UNVERIFIED_STATE
                report.detail = (
                    f"the world could not be read after "
                    f"{report.unverified_reads} attempt(s)"
                )
                return report

            answer = await self.decider.propose(vector, step=step)

            if answer is None:
                report.stop_reason = STOP_PLAN_EXHAUSTED
                return report

            if isinstance(answer, Escalation):
                self._record_escalation(answer, vector, step)
                report.stop_reason = STOP_ESCALATED
                report.detail = answer.reason
                return report

            if isinstance(answer, Completion):
                # A clean stop, because the decider answered: it believes there is
                # nothing left to do. Whether that belief was *true* is the
                # checkers' question and they are what say so - a loop that judged
                # the claim itself would be grading the thing it is measuring.
                self._record_completion(answer, vector, step)
                report.stop_reason = STOP_PLAN_EXHAUSTED
                report.detail = answer.reason or "the decider reported nothing left to do"
                return report

            assert isinstance(answer, Proposal)
            refused = self._refuse_off_goal(answer)
            if refused is not None:
                report.stop_reason, report.detail = refused
                return report
            outcome = await self._run_one(answer, vector, step, report)
            if outcome is not None:
                # A non-None return means the run must stop; the reason is already
                # set on the report.
                return report

        report.stop_reason = STOP_STEP_LIMIT
        report.detail = f"reached the {max_steps}-step ceiling without exhausting the plan"
        return report

    async def _runtime_unavailable(self) -> str | None:
        """Why the world is unreadable, when the host will say so.

        Returns a message when ``/health`` reports the bot's Minecraft connection
        closed, and ``None`` otherwise - including when ``/health`` itself cannot
        be read, because an unreadable diagnosis is not a diagnosis. That
        asymmetry is the point: naming the cause is strictly better than reporting
        a parse failure, and *guessing* the cause would be strictly worse than
        either.
        """
        connected = await self.reader.client.bot_connected()
        if connected is not False:
            return None
        return (
            "the bot's Minecraft connection is closed; /health reports "
            "minecraft.connected=false. mine-ai-mcp does not reconnect - "
            "src/server/runtime-host.ts says a new connection requires an "
            "explicit service restart, so this run cannot be recovered in place."
        )

    async def _trusted_state(
        self, report: LoopReport
    ) -> tuple[StateVector | None, str | None]:
        """A state the decider may be shown, and why there is none.

        Re-reads rather than proceeding with a degraded vector: a fresh read is a
        fresh attempt, whereas carrying invented numbers forward is the exact
        failure this codebase refuses elsewhere. ``last_death_at`` is captured
        here as well because absorbing a death is a comparison across the step.

        Returns ``(vector, None)`` on a usable state, ``(None, reason)`` when the
        bot's connection is known to be closed, and ``(None, None)`` when the read
        simply could not be made - which the caller must not confuse with the two.
        """
        for attempt in range(self.max_unverified_reads + 1):
            vector = await self.reader.read()
            if not vector.unverified:
                return vector, None
            report.unverified_reads += 1
            # Named on the *first* failure only. A closed socket cannot be read
            # better by asking again, so once the cause is establishable there is
            # nothing to gain from the remaining retries; but one bad read is not
            # a dead bot, and calling it one would be a confident wrong answer.
            if attempt == 0:
                unavailable = await self._runtime_unavailable()
                if unavailable is not None:
                    return None, unavailable
            if attempt < self.max_unverified_reads:
                self._record_unverified(vector, attempt)
        return None, None

    def _refuse_off_goal(self, proposal: Proposal) -> tuple[str, str] | None:
        """Refuse a proposal that names a goal this run cannot work on.

        Two refusals, both named, both stopping the run rather than quietly
        dropping the objective:

        * a goal outside the set - the decider is answering a different question
          than the scenario asked, and recording that as progress would corrupt
          the measurement the scenario exists to make;
        * a goal past ``max_attempts_per_goal`` - the harness has already declared
          it will stop trying, and running it anyway spends a real objective on
          the model path, which is 20-37 s per decision.

        Both return ``None`` when the run has no goals, which is the ``run-loop``
        path and must be unaffected by any of this.
        """
        if self.goals is None or not self.goals.enabled:
            return None
        goal = proposal.goal
        if goal is None:
            # A run with goals wants to know what each objective was for. A
            # proposal that does not say cannot be counted, and silently counting
            # it as unattributed work is how "calls per goal" becomes a guess.
            return (
                STOP_GOAL_UNKNOWN,
                f"step {proposal.tool!r} named no goal, and this run has "
                f"{len(self.goals.items)} goal(s); calls per goal cannot be measured "
                "without knowing which goal each objective served",
            )
        if not self.goals.known(goal):
            return (
                STOP_GOAL_UNKNOWN,
                f"{goal!r} is not in this run's goal set {list(self.goals.items)}",
            )
        if self.goals.exhausted(goal):
            return (
                STOP_GOAL_ATTEMPTS,
                f"{goal!r} has had {self.goals.attempts.get(goal, 0)} attempt(s) against "
                f"budget.max_attempts_per_goal={self.goals.max_attempts}",
            )
        return None

    async def _run_one(
        self,
        proposal: Proposal,
        vector: StateVector,
        step: int,
        report: LoopReport,
    ) -> str | None:
        """Run one proposal. Returns a stop reason, or ``None`` to keep going."""
        death_before = vector.last_death_at
        objective = Objective(
            tool=proposal.tool,
            arguments=proposal.arguments,
            rationale=proposal.rationale,
            submission_id=f"{self.submission_prefix}{uuid.uuid4().hex}",
            metadata={
                "source": proposal.source,
                "confidence": proposal.confidence,
                "goal": proposal.goal,
            },
        )
        seen = SeenState(vector=vector.to_dict(), state_hash=vector.state_hash)
        entry = StepReport(index=step, tool=proposal.tool, state_hash=vector.state_hash)

        try:
            result = await self.runner.run(objective, seen=seen)
        except ObjectiveFailed as error:
            entry.error = str(error)
            # Credited on the way out too: a refused submission still spent an
            # objective, and a report that counted only the successes would
            # understate the cost of the run by however many were refused.
            self._credit_abandoned(proposal.goal, entry, report, f"refused: {error}")
            report.steps.append(entry)
            self._emit(entry)
            report.stop_reason = STOP_REFUSED
            report.detail = str(error)
            return report.stop_reason
        except BudgetExceeded as error:
            entry.error = str(error)
            report.steps.append(entry)
            self._emit(entry)
            report.stop_reason = STOP_BUDGET
            report.detail = str(error)
            return report.stop_reason
        except HarnessError as error:
            entry.error = str(error)
            report.steps.append(entry)
            self._emit(entry)
            report.stop_reason = STOP_ERROR
            report.detail = str(error)
            return report.stop_reason

        entry.result = result
        entry.error = result.error

        death = await self._absorb_death(death_before, result)
        if death is not None:
            entry.death_cause = death
            report.deaths_absorbed += 1

        self._credit_goal(proposal.goal, result, entry, report)

        report.steps.append(entry)
        self._emit(entry)
        return None

    def _credit_goal(
        self,
        goal: str | None,
        result: ObjectiveResult,
        entry: StepReport,
        report: LoopReport,
    ) -> None:
        """Bank one attempt against the goal this objective served.

        Recorded from the settled result, not from the proposal: an objective
        that failed is still an attempt, and only a success credits the goal. The
        cap is on *attempts*, deliberately - a recipe may legitimately need many
        steps without any of them being a search, and counting only failures would
        let a decider retry forever as long as each try failed differently.
        """
        if goal is None or self.goals is None:
            return
        outcome = f"{result.state}:{result.status}" if result.status else result.state
        self.goals.record(goal, ok=result.ok, outcome=outcome)
        report.goal_attempts[goal] = self.goals.attempts.get(goal, 0)
        entry.goal = goal

    def _credit_abandoned(
        self,
        goal: str | None,
        entry: StepReport,
        report: LoopReport,
        outcome: str,
    ) -> None:
        """Bank an attempt for an objective that raised instead of settling."""
        if goal is None or self.goals is None:
            return
        self.goals.record(goal, ok=False, outcome=outcome)
        report.goal_attempts[goal] = self.goals.attempts.get(goal, 0)
        entry.goal = goal

    async def _absorb_death(
        self, death_before: str | None, result: ObjectiveResult
    ) -> str | None:
        """Notice a death during an objective, and keep going.

        Only re-reads after an objective that did *not* succeed. A death cannot
        make a succeeded objective not have succeeded, so a successful step needs
        no extra call, and the next step's read would find the death anyway - it
        would just attribute it to the following objective instead of this one.

        Returns the cause, or ``None`` if the bot did not die. A death is not a
        stop condition: the bot respawns, and the run's remaining objectives are
        still answerable. It is recorded because "the objective failed" and "the
        objective failed because the bot died" are different facts and only one of
        them says anything about the objective.
        """
        if result.ok:
            return None
        after = await self.reader.read()
        if after.unverified or after.last_death_at is None:
            return None
        if death_before is not None and after.last_death_at == death_before:
            return None
        return after.last_death_cause or "unknown"

    def _record_escalation(
        self, escalation: Escalation, vector: StateVector, step: int
    ) -> None:
        """Write the refusal to decide.

        Recorded as a full row rather than a log line because an escalation is a
        result: it is the decider's answer, and the Phase 3 eval needs to see how
        often the honest answer was "I will not".
        """
        self._record(
            DecisionRow(
                state_vector=vector.to_dict(),
                state_hash=vector.state_hash,
                question=escalation.question or {"kind": "decision", "step": step},
                escalated=True,
                escalation_reason=escalation.reason,
                outcome="escalated",
            )
        )

    def _record_completion(
        self, completion: Completion, vector: StateVector, step: int
    ) -> None:
        """Write the decider's claim that the work is finished.

        Recorded as a full row for the same reason an escalation is (D12): it is an
        answer, not an absence of one. And specifically because this is the answer
        that ends a run - measured on 2026-10-02, a run collected its logs correctly
        and then the model reported nothing left to do, which stopped the loop
        cleanly with a ledger holding one successful objective and no record of the
        decision that ended it. The report said ``holds_item: holds 0x
        wooden_pickaxe`` and the ledger could not say why anyone believed otherwise.

        ``outcome`` is ``completed``, deliberately distinct from ``escalated``: one is
        an answer with nothing to run and one is a refusal to answer at all.
        """
        question: dict[str, Any] = {"kind": "completion", "step": step}
        if completion.goal is not None:
            question["goal"] = completion.goal
        self._record(
            DecisionRow(
                state_vector=vector.to_dict(),
                state_hash=vector.state_hash,
                question=question,
                answer="done",
                outcome="completed",
                error=completion.reason or None,
            )
        )

    def _record_unverified(self, vector: StateVector, attempt: int) -> None:
        self._record(
            DecisionRow(
                state_vector=vector.to_dict(),
                question={"kind": "state_read", "attempt": attempt},
                outcome="unverified",
                error="; ".join(vector.unverified),
            )
        )

    def _record(self, row: DecisionRow) -> None:
        if self.ledger is not None:
            self.ledger.record(row)

    def _emit(self, entry: StepReport) -> None:
        if self.on_step is not None:
            self.on_step(entry)