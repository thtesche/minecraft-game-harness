"""The decision seam: what the loop asks, and what a decider may answer.

The loop is deterministic by design (D7). Everything that depends on judgement
about the world lives behind :class:`Decider`, so that swapping a frontier model
in changes this module and nothing else. Phase 1 ships a scripted stand-in; it
is a stand-in and says so in the ledger, because a run whose decisions came from
a script must never be mistaken for a run whose decisions came from a model.

Three answers are possible, and the third is the one that gets forgotten:

* :class:`Proposal` - here is the objective, here is why
* :class:`Escalation` - this is a question I will not answer on my own
* ``None`` - there is nothing left to decide

An escalation is not a soft no. Per :class:`~harness.config.LayaConfig`, an unset
``min_confidence`` means escalate rather than assume, so the loop's job on an
escalation is to *stop*, because stopping is the only honest option while no
model exists to escalate to. A loop that downgraded an escalation into a guess
would be the exact failure the threshold exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from .goals import GoalBoard
from .mcp_client import argument_names
from .state import StateVector


@dataclass(frozen=True)
class Proposal:
    """A concrete objective the loop should submit."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Why this, in a form worth reading in a ledger a month from now.
    rationale: str = ""
    #: ``None`` when the decider has no calibrated confidence to offer, which is
    #: the honest value for a script and a reason to distrust a model that omits it.
    confidence: float | None = None
    #: Recorded so a later reader can tell a scripted row from a model row without
    #: consulting the code that produced it.
    source: str = "unknown"
    #: Which goal this objective is pursuing, when the run has goals at all. A
    #: registry item name from the goal set (D18). It is recorded on the ledger
    #: row and it is what ``budget.max_attempts_per_goal`` is counted against, so
    #: "calls per goal" becomes a number rather than something a reader infers
    #: from the arguments.
    goal: str | None = None


@dataclass(frozen=True)
class Escalation:
    """A question the decider declines to answer."""

    reason: str
    #: What was asked, kept so the escalation is diagnosable after the fact.
    question: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Decider(Protocol):
    """Chooses the next objective, or declines to.

    ``step`` is the 0-based index of this decision within the run, passed so a
    model that needs to know how much of its plan is left can know it, rather
    than reading the remaining budget out of a vector field that does not exist.

    ``goals`` is the run's :class:`~harness.goals.GoalBoard`, or ``None`` when
    the run has no goals. It is passed rather than read from the vector because
    a goal set is a fact about the *run*, not about the world: the same world
    read twice belongs to two different runs aiming at two different things, and
    folding the goal into the state would make those two runs share a cache key.
    """
    goals: GoalBoard | None

    async def propose(
        self,
        vector: StateVector,
        *,
        step: int,
    ) -> Proposal | Escalation | None: ...


def confidence_gate(
    confidence: float | None,
    min_confidence: float | None,
) -> Escalation | None:
    """Apply the calibration policy.

    ``min_confidence is None`` means *escalate*, not *accept*. A threshold chosen
    from measured accuracy at a chosen coverage does not exist yet, so every
    answer is currently an uncalibrated guess; the policy says so and refuses it
    rather than proceeding on the assumption that a plausible number is a good
    one.

    Kept as a helper rather than inlined into a decider so the policy lives in
    one place and a future Laya-backed decider cannot forget it.
    """
    if min_confidence is None:
        return Escalation(
            reason=(
                "laya.min_confidence is unset, so no answer can be treated as "
                "calibrated; set it from a measured eval or leave it unset and "
                "expect every decision to escalate"
            ),
            question={"min_confidence": None, "confidence": confidence},
        )
    if confidence is None:
        return Escalation(
            reason="the decider returned no confidence, so none can be checked",
            question={"min_confidence": min_confidence, "confidence": None},
        )
    if confidence < min_confidence:
        return Escalation(
            reason=f"confidence {confidence:.3f} is below the {min_confidence:.3f} threshold",
            question={"min_confidence": min_confidence, "confidence": confidence},
        )
    return None


class ScriptedDecider:
    """Replays a fixed objective list. The Phase 1 stand-in for a model.

    Not a mock in the sense of being disposable: it is the thing that proves the
    loop sequences objectives, survives death, and writes a ledger, and it stays
    as the regression fixture once a real model is wired in. What it is not is a
    decision-maker, and every row it produces carries ``source="scripted"`` and a
    ``None`` confidence so that the Phase 2 baseline can exclude it.

    Arguments are checked against the advertised input schemas. The harness
    otherwise binds arguments from ``tools/list`` and never from prose, and a
    hand-written argument name is exactly the kind of guess that produces a
    plausible answer about the wrong thing.
    """

    def __init__(
        self,
        script: Sequence[tuple[str, dict[str, Any]]],
        tools: Sequence[dict[str, Any]] = (),
        rationales: Sequence[str] = (),
    ) -> None:
        self._script = list(script)
        self._tools = {str(tool.get("name")): tool for tool in tools}
        self._rationales = list(rationales)
        self._index = 0
        #: Set by the loop before each decision. A script names no goal, so this
        #: stays ``None`` unless a caller wires it deliberately.
        self.goals: GoalBoard | None = None

    @property
    def remaining(self) -> int:
        return max(0, len(self._script) - self._index)

    def reset(self) -> None:
        """Rewind, so a retried run replays the same script in the same order."""
        self._index = 0

    async def propose(self, vector: StateVector, *, step: int) -> Proposal | None:
        if self._index >= len(self._script):
            return None
        tool, arguments = self._script[self._index]
        self._index += 1
        self._reject_unknown_arguments(tool, arguments)
        rationale = (
            self._rationales[self._index - 1]
            if self._index - 1 < len(self._rationales)
            else ""
        )
        return Proposal(
            tool=tool,
            arguments=dict(arguments),
            rationale=rationale,
            confidence=None,
            source="scripted",
        )

    def _reject_unknown_arguments(self, tool: str, arguments: dict[str, Any]) -> None:
        """Refuse an argument the advertised schema does not accept.

        Silent on an unknown *tool*: the loop gets that refusal from the server
        and records it honestly. This only catches the class of error the server
        cannot see, because a wrong argument name is dropped or coerced rather
        than refused.

        Reads the schema through :func:`~harness.mcp_client.argument_names`,
        which accepts both ``input_schema`` and ``inputSchema``. This method used
        to read ``inputSchema`` alone, which mine-ai-mcp does not publish - so
        against the live host the check never ran and any argument name passed.
        """
        properties = argument_names(self._tools.get(tool))
        if not properties:
            return
        unknown = sorted(set(arguments) - properties)
        if unknown:
            raise ScriptedArgumentError(
                f"{tool} accepts {sorted(properties)}, not {unknown}; a wrong "
                "argument name is dropped by the server rather than refused, so "
                "the objective would run against the wrong thing and say nothing"
            )


class ScriptedArgumentError(RuntimeError):
    """The script names an argument the tool does not advertise."""