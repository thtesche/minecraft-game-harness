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

from .decide import Decider, Escalation, Proposal
from .errors import BudgetExceeded, HarnessError, ObjectiveFailed
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

#: Stop reasons that mean "the run did not finish its work", as opposed to
#: finishing it. A plan that ran out is a success; a refusal is not.
INCOMPLETE_STOPS = frozenset(
    {STOP_STEP_LIMIT, STOP_ESCALATED, STOP_UNVERIFIED_STATE, STOP_REFUSED, STOP_BUDGET, STOP_ERROR}
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
        return {
            "stopReason": self.stop_reason,
            "detail": self.detail,
            "steps": len(self.steps),
            "succeeded": self.objectives_succeeded,
            "verifiedEvidence": self.verified_evidence,
            "deathsAbsorbed": self.deaths_absorbed,
            "unverifiedReads": self.unverified_reads,
        }


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
    ) -> None:
        self.reader = reader
        self.runner = runner
        self.decider = decider
        self.ledger = ledger
        self.clock = clock
        self.submission_prefix = submission_prefix
        self.max_unverified_reads = max_unverified_reads
        self.on_step = on_step

    async def run(self, *, max_steps: int = 32) -> LoopReport:
        report = LoopReport()
        started = self.clock()

        for step in range(max_steps):
            vector = await self._trusted_state(report)
            if vector is None:
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

            assert isinstance(answer, Proposal)
            outcome = await self._run_one(answer, vector, step, report)
            if outcome is not None:
                # A non-None return means the run must stop; the reason is already
                # set on the report.
                return report

        report.stop_reason = STOP_STEP_LIMIT
        report.detail = f"reached the {max_steps}-step ceiling without exhausting the plan"
        return report

    async def _trusted_state(self, report: LoopReport) -> StateVector | None:
        """A state the decider may be shown, or ``None`` to end the run.

        Re-reads rather than proceeding with a degraded vector: a fresh read is a
        fresh attempt, whereas carrying invented numbers forward is the exact
        failure this codebase refuses elsewhere. ``last_death_at`` is captured
        here as well because absorbing a death is a comparison across the step.
        """
        for attempt in range(self.max_unverified_reads + 1):
            vector = await self.reader.read()
            if not vector.unverified:
                return vector
            report.unverified_reads += 1
            if attempt < self.max_unverified_reads:
                self._record_unverified(vector, attempt)
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
            metadata={"source": proposal.source, "confidence": proposal.confidence},
        )
        seen = SeenState(vector=vector.to_dict(), state_hash=vector.state_hash)
        entry = StepReport(index=step, tool=proposal.tool, state_hash=vector.state_hash)

        try:
            result = await self.runner.run(objective, seen=seen)
        except ObjectiveFailed as error:
            entry.error = str(error)
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

        report.steps.append(entry)
        self._emit(entry)
        return None

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