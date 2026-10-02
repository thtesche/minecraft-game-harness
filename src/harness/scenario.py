"""Scenarios: a prompt, a goal set, and a checker.

A scenario that is only a prompt is a demo. What makes it a measurement is the
third part - the checker, written *before* the run (D19). "The run looked
reasonable" is not a result; "the inventory held a wooden pickaxe after at most
six tool calls" is, and it is the kind of result a later, cheaper decider can be
held to.

Two rules keep the checkers honest:

**They read the world, not the run.** Every check is answered from a fresh
``view_status`` taken after the loop finishes, plus the loop's own report for
things only the run knows (how many calls it took, whether the bot died). A
checker that graded the loop's beliefs would grade its own homework, and the
most tempting version of that mistake - reading the inventory out of the last
step's *evidence* - would inherit the loop's truncation: ``StateReader`` bounds
the carry list to 12 stacks to keep the prompt small, so an item held in slot 20
is invisible to a check that reads the vector. The full inventory is read here
for exactly that reason.

**An unknown check is an error, not a pass.** A misspelled ``kind`` must not
evaluate to true, because a scenario that cannot fail is indistinguishable from
one that passed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from .goals import GoalSet
from .loop import LoopReport
from .state import TOOL_TIER_ORDER, best_equipment, situation_of


class ScenarioError(ValueError):
    """A scenario that cannot be run, or a check that cannot be evaluated."""


@dataclass(frozen=True)
class Verdict:
    """One check's answer, in enough detail to argue with."""

    kind: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.kind, "passed": self.passed, "detail": self.detail}


@dataclass
class WorldFacts:
    """What the world actually looks like, read fresh after the run.

    Deliberately independent of :class:`~harness.state.StateVector`. The vector is
    shaped for a decision prompt and is deliberately lossy - twelve stacks, no
    item counts below the carry limit, best-tool collapsed to one name. A checker
    reading a deliberately lossy view is how "the bot had the pickaxe" becomes a
    false negative.
    """

    #: Every stack in the inventory, counted, not truncated.
    inventory: dict[str, int] = field(default_factory=dict)
    #: Best equipment per table, as the server's one-row-per-class tables report
    #: it. The *name* is the item held, or ``None`` when nothing is held in any
    #: class - the rows exist either way, with ``tier: "none"``.
    best_tool: str | None = None
    best_tool_tier: str | None = None
    best_armour: str | None = None
    best_armour_tier: str | None = None
    time_phase: str | None = None
    last_death_at: str | None = None
    #: A section the harness could not interpret. Its presence fails every
    #: world-dependent check, because a check answered on an unread world is a
    #: guess wearing a verdict's clothes.
    unverified: list[str] = field(default_factory=list)
    #: The read was complete but something in it could not be placed. Kept apart
    #: from ``unverified`` because an unrankable tier does not make the answer
    #: unknowable - it only means the harness cannot say which item was best.
    #: Folding the two together would turn "the bot holds a stone pickaxe and a
    #: trident" into a failed scenario with a misleading reason attached.
    warnings: list[str] = field(default_factory=list)

    def holds(self, item: str, count: int = 1) -> bool:
        return self.inventory.get(item, 0) >= count

    def tool_tier_rank(self) -> int:
        """Rank of the best tool held, or -1 when nothing is held.

        The tier, not the name: the tier is what "stone tier" means, and a name
        comparison would put ``wooden_pickaxe`` below ``diamond_shovel`` while
        both rows say their own tier.
        """
        if self.best_tool_tier in TOOL_TIER_ORDER:
            return TOOL_TIER_ORDER.index(self.best_tool_tier)
        return -1


def world_facts(situation: dict[str, Any]) -> WorldFacts:
    """Build :class:`WorldFacts` from a raw ``view_status`` situation.

    Every field here is read from the server's own schema
    (``src/actions/view-status/contract.ts``), so a shape change on the server
    shows up here as a missing section and a red check rather than as a green
    one over a field that quietly stopped arriving.
    """
    facts = WorldFacts()

    inventory = situation.get("inventory")
    # Presence is checked as well as shape. A reply with no `inventory` key at all
    # is not an empty inventory - it is a read the harness cannot interpret, and
    # reading it as "the bot holds nothing" turns a broken read into a red run
    # that looks like a bot that failed.
    if not isinstance(inventory, dict):
        facts.unverified.append("inventory section was missing or not an object")
    elif not isinstance(inventory.get("stacks"), list):
        facts.unverified.append("inventory carried no stacks")
    else:
        for stack in inventory["stacks"]:
            if not isinstance(stack, dict):
                continue
            name = stack.get("name")
            if not isinstance(name, str) or not name:
                continue
            count = stack.get("count")
            facts.inventory[name] = (
                count if isinstance(count, int) and not isinstance(count, bool) else 1
            )

    tools = situation.get("tools")
    if not isinstance(tools, dict):
        facts.unverified.append("tools section was missing or not an object")
    else:
        for key in ("tools", "armour"):
            if not isinstance(tools.get(key), list):
                facts.unverified.append(f"tools.{key} was not a list")

    # Ranked by the state reader's own routine, not by a second one written here.
    facts.best_tool, facts.best_tool_tier, unranked_tools = best_equipment(situation, "tools")
    facts.best_armour, facts.best_armour_tier, unranked_armour = best_equipment(
        situation, "armour"
    )
    facts.warnings.extend(unranked_tools)
    facts.warnings.extend(unranked_armour)

    clock = situation.get("clock")
    if not isinstance(clock, dict):
        facts.unverified.append("clock section was missing or not an object")
    else:
        facts.time_phase = clock.get("phase")

    death = situation.get("lastDeath")
    if isinstance(death, dict):
        observed = death.get("observedAt")
        facts.last_death_at = observed if isinstance(observed, str) else None
    return facts


#: A check takes the facts it is allowed to see and returns a verdict.
#:
#: ``facts`` is ``None`` when the world could not be read. A check that depends on
#: the world must then *fail*, naming the read failure - never pass on absent
#: evidence. A check that only reads the run (how many calls it took) is still
#: answerable, and says so.
CheckFn = Callable[["CheckContext"], Verdict]


@dataclass(frozen=True)
class CheckContext:
    """Everything a check may look at."""

    facts: WorldFacts | None
    loop: LoopReport
    params: dict[str, Any]


@dataclass(frozen=True)
class Check:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.params}


@dataclass
class ScenarioReport:
    """The run and what the checkers made of it."""

    name: str
    loop: LoopReport
    verdicts: list[Verdict]
    usage: dict[str, Any] | None = None
    run_id: str = ""
    reference: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True only if every check passed *and* the run finished cleanly.

        Both, deliberately: a loop that stopped on a refusal did not satisfy its
        goals, and a checker that ignored the stop reason would call that a pass.
        """
        return bool(self.verdicts) and all(v.passed for v in self.verdicts) and self.loop.ok

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "scenario": self.name,
            "passed": self.passed,
            "runId": self.run_id,
            **self.loop.summary(),
            "verdicts": [v.to_dict() for v in self.verdicts],
        }
        if self.usage:
            out["modelUsage"] = self.usage
        if self.reference:
            out["reference"] = self.reference
        return out


@dataclass(frozen=True)
class Scenario:
    """A named, reproducible thing to measure."""

    name: str
    goals: GoalSet
    checks: tuple[Check, ...]
    prompt: str = ""
    max_steps: int = 24
    #: A published run this scenario is meant to be read against, carried through
    #: to the report rather than interpreted. Recorded rather than checked
    #: because the comparison is not always apples to apples: the reference spent
    #: 4 tool calls on this objective, 2 of them reads, and this harness reads
    #: state deterministically without spending a submission. The numbers are
    #: reported side by side and a reader draws the conclusion.
    reference: dict[str, Any] = field(default_factory=dict)
    #: The world state this scenario's number is only valid *from*, checked
    #: against a live read before the loop starts. ``inventory`` is the exact set
    #: of stacks the bot is expected to hold, name to count - exact, because the
    #: premise is the measurement: a bot that already holds the logs makes
    #: "reach a pickaxe from nothing" a different and easier question, and a
    #: report that says ``within_calls: 3`` without saying the bot started with
    #: logs is a number with a hidden term in it. Recorded here because the world
    #: is not the scenario's to choose; a mismatch is a refusal naming both sides
    #: so the scenario is corrected against reality rather than reality quietly
    #: edited to fit.
    starts_from: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validated here rather than only in ``load``, because a check that only
        # one construction path applies is not a check: a ``Scenario`` built in a
        # test or a future caller would carry a premise nothing ever enforced, and
        # the run against it would quietly measure the wrong thing.
        _check_starts_from(self.starts_from, self.name or "<unnamed scenario>")

    @classmethod
    def load(cls, path: Path | str) -> "Scenario":
        scenario_path = Path(path)
        try:
            raw_text = scenario_path.read_text()
        except OSError as error:
            # Named rather than a bare FileNotFoundError: a scenario path that
            # does not exist is a typo, and the reader needs to be told which
            # file rather than handed a traceback out of pathlib.
            raise ScenarioError(f"{scenario_path} could not be read: {error}") from error
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as error:
            raise ScenarioError(f"{scenario_path} is not valid JSON: {error}") from error
        if not isinstance(raw, dict):
            raise ScenarioError(f"{scenario_path}: expected an object, got {type(raw).__name__}")

        known = {"name", "prompt", "goals", "checks", "maxSteps", "reference", "startsFrom"}
        unknown = set(raw) - known
        if unknown:
            raise ScenarioError(
                f"{scenario_path}: unknown keys {sorted(unknown)}; expected {sorted(known)}. "
                "A typo'd key here would otherwise be a scenario that silently does less "
                "than it appears to."
            )
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ScenarioError(f"{scenario_path}: no name; a scenario nobody can identify is not reported")
        goals = GoalSet.from_any(raw.get("goals"), source=str(scenario_path))
        checks = _checks(raw.get("checks"), str(scenario_path))
        prompt = raw.get("prompt", "")
        if not isinstance(prompt, str):
            raise ScenarioError(f"{scenario_path}: prompt must be a string")
        max_steps = raw.get("maxSteps", 24)
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ScenarioError(f"{scenario_path}: maxSteps must be an integer of at least 1, got {max_steps!r}")
        reference = raw.get("reference", {})
        if not isinstance(reference, dict):
            raise ScenarioError(
                f"{scenario_path}: reference must be an object, got {type(reference).__name__}"
            )
        starts_from = raw.get("startsFrom", {})
        if not isinstance(starts_from, dict):
            raise ScenarioError(
                f"{scenario_path}: startsFrom must be an object, got {type(starts_from).__name__}"
            )
        _check_starts_from(starts_from, str(scenario_path))
        return cls(  # __post_init__ checks starts_from again, by design: see there.
            name=name.strip(),
            goals=goals,
            checks=checks,
            prompt=prompt,
            max_steps=max_steps,
            reference=reference,
            starts_from=starts_from,
        )


#: Keys ``startsFrom`` accepts. ``inventory`` is the one that is enforced; ``note``
#: is prose for the reader and is carried into the report unparsed.
STARTS_FROM_KEYS = frozenset({"inventory", "note"})


def _check_starts_from(starts_from: dict[str, Any], source: str) -> None:
    """Refuse a starting condition the harness cannot enforce.

    A permissive loader here would mean a typo'd ``inventry`` silently checks
    nothing, which is the premise disappearing rather than failing - so unknown
    keys are refused, and so is an inventory that is not a flat name-to-count map.
    """
    unknown = set(starts_from) - STARTS_FROM_KEYS
    if unknown:
        raise ScenarioError(
            f"{source}: startsFrom unknown keys {sorted(unknown)}; expected "
            f"{sorted(STARTS_FROM_KEYS)}. A typo'd key here would check nothing at all, "
            "which is the premise vanishing rather than failing."
        )
    if "inventory" not in starts_from:
        return
    inventory = starts_from["inventory"]
    if not isinstance(inventory, dict):
        raise ScenarioError(
            f"{source}: startsFrom.inventory must be an object mapping item name to count"
        )
    for item, count in inventory.items():
        if not isinstance(item, str) or not item:
            raise ScenarioError(f"{source}: startsFrom.inventory has an empty item name")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ScenarioError(
                f"{source}: startsFrom.inventory[{item!r}] must be a count of at least 1, "
                f"got {count!r}; a stack the bot must NOT hold is expressed by omitting it, "
                "not by a zero or a negative"
            )


def start_mismatch(scenario: Scenario, facts: WorldFacts | None) -> str | None:
    """Why the world is not the one this scenario's number is valid from.

    ``None`` when it matches, when the scenario declares no precondition, or when
    the world could not be read - the last of which is *not* a match. An
    unreadable world is reported as such rather than passed, so the caller can
    refuse; saying "the precondition holds" about a world it never read is the
    one answer that would be a lie.
    """
    declared = scenario.starts_from.get("inventory")
    if not isinstance(declared, dict):
        return None
    if facts is None:
        return "the world could not be read, so the scenario's starting condition is unchecked"
    if facts.unverified:
        return (
            "the starting world read was incomplete ("
            + "; ".join(facts.unverified)
            + "), so the scenario's starting condition is unchecked"
        )
    held = dict(sorted(facts.inventory.items()))
    expected = dict(sorted(declared.items()))
    if held == expected:
        return None
    wanted = ", ".join(f"{count}x {item}" for item, count in expected.items()) or "nothing"
    actual = ", ".join(f"{count}x {item}" for item, count in held.items()) or "nothing"
    return (
        f"the world does not match this scenario's starting condition. "
        f"scenario {scenario.name!r} declares: {wanted}. world holds: {actual}. "
        "A scenario that cannot measure what it claims must not produce a number, "
        "so either empty the bot's inventory or correct startsFrom.inventory to the "
        "world as it actually is."
    )


def _checks(raw: Any, source: str) -> tuple[Check, ...]:
    if not isinstance(raw, list) or not raw:
        raise ScenarioError(f"{source}: no checks; a scenario with no checker is a demo (D19)")
    out: list[Check] = []
    for index, entry in enumerate(raw):
        where = f"{source}: check {index}"
        if not isinstance(entry, dict) or "kind" not in entry:
            raise ScenarioError(f"{where}: expected {{\"kind\": ...}}")
        kind = entry["kind"]
        if kind not in CHECKS:
            raise ScenarioError(
                f"{where}: unknown check {kind!r}; known checks are {sorted(CHECKS)}. "
                "An unknown check must not evaluate to true - a scenario that "
                "cannot fail looks exactly like one that passed."
            )
        params = {key: value for key, value in entry.items() if key != "kind"}
        out.append(Check(kind=kind, params=params))
    return tuple(out)


def evaluate(check: Check, context: CheckContext) -> Verdict:
    """Run one check, with its own parameters.

    The check's ``params`` win over whatever the context carried, so a caller
    cannot accidentally grade ``holds_item`` against an empty parameter set and
    get a verdict about nothing. The context supplies the facts and the run; the
    check supplies the question.
    """
    return CHECKS[check.kind](replace(context, params=check.params))


# --------------------------------------------------------------------------
# The checks.
# --------------------------------------------------------------------------


def _needs_world(context: CheckContext) -> Verdict | None:
    """A verdict for when the world could not be read.

    Fails, and says why. A check that passed here would be reporting that the
    bot achieved something on the strength of no evidence at all.

    Only ``unverified`` counts - a section the harness could not interpret. A
    *warning* is a complete read the harness cannot fully rank, which is a
    different thing and does not make the answer unknowable.
    """
    if context.facts is None:
        return Verdict(kind="world", passed=False, detail="the world could not be read after the run")
    if context.facts.unverified:
        return Verdict(
            kind="world",
            passed=False,
            detail=f"the post-run read was incomplete: {'; '.join(context.facts.unverified)}",
        )
    return None


def _note(context: CheckContext, detail: str) -> str:
    """The detail, plus any ranking the harness could not do."""
    if not context.facts or not context.facts.warnings:
        return detail
    return f"{detail} (could not rank: {'; '.join(context.facts.warnings)})"


def _needs_run(context: CheckContext, kind: str) -> Verdict | None:
    """A verdict for a run-shaped check on a run that did nothing.

    Every run-shaped check is a statement about a set of steps - "every settled
    objective carried evidence", "no step died", "the run used at most N calls".
    On a run with no steps all three are vacuously true, and a check that passes
    on nothing is the same failure as a check that stopped running: the report
    shows three green lines for a run in which the bot never moved. So a
    universal over the empty set is reported as unmeasured, not satisfied.

    ``holds_item`` and friends do not need this. They are statements about the
    world, and an empty world genuinely fails them.
    """
    if not context.loop.steps:
        return Verdict(
            kind=kind,
            passed=False,
            detail=f"the run made no tool call, so {kind} has nothing to "
            "measure; a check that passes on nothing has not passed",
        )
    return None


def check_holds_item(context: CheckContext) -> Verdict:
    unavailable = _needs_world(context)
    if unavailable is not None:
        return unavailable
    item = context.params.get("item")
    if not isinstance(item, str) or not item:
        return Verdict("holds_item", False, "holds_item needs an `item`")
    want = context.params.get("count", 1)
    if isinstance(want, bool) or not isinstance(want, int) or want < 1:
        return Verdict("holds_item", False, f"holds_item needs a positive integer `count`, got {want!r}")
    have = context.facts.inventory.get(item, 0)
    return Verdict(
        "holds_item",
        have >= want,
        f"holds {have}x {item}, needed {want}x",
    )


def check_best_tool_at_least(context: CheckContext) -> Verdict:
    unavailable = _needs_world(context)
    if unavailable is not None:
        return unavailable
    tier = context.params.get("tier")
    if tier not in TOOL_TIER_ORDER:
        return Verdict(
            "best_tool_at_least",
            False,
            f"tier {tier!r} is not in the harness tool tier order {list(TOOL_TIER_ORDER)}",
        )
    rank = context.facts.tool_tier_rank()
    want = TOOL_TIER_ORDER.index(tier)
    return Verdict(
        "best_tool_at_least",
        rank >= want,
        _note(
            context,
            f"best tool is {context.facts.best_tool!r} at tier "
            f"{context.facts.best_tool_tier!r} (rank {rank}), needed at least "
            f"{tier} (rank {want})",
        ),
    )


def check_time_phase_is(context: CheckContext) -> Verdict:
    unavailable = _needs_world(context)
    if unavailable is not None:
        return unavailable
    want = context.params.get("phase")
    have = context.facts.time_phase
    return Verdict("time_phase_is", have == want, f"clock reads {have!r}, needed {want!r}")


def check_no_death(context: CheckContext) -> Verdict:
    """No death during the run.

    Answered from the run, not from ``lastDeath``: a death absorbed ten steps ago
    still leaves a ``lastDeath`` in the world, and grading this run on a death
    from a previous one would fail a run for something it did not do.
    """
    unused = _needs_run(context, "no_death")
    if unused is not None:
        return unused
    absorbed = context.loop.deaths_absorbed
    return Verdict(
        "no_death",
        absorbed == 0,
        f"{absorbed} death(s) absorbed during the run",
    )


def check_within_calls(context: CheckContext) -> Verdict:
    """The run finished inside a call budget.

    The comparable number. The reference run reached a wooden pickaxe from an
    empty inventory in four tool calls, two of them reads; a budget is how this
    baseline gets read against that rather than merely admired.
    """
    unused = _needs_run(context, "within_calls")
    if unused is not None:
        return unused
    budget = context.params.get("calls")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        return Verdict("within_calls", False, f"within_calls needs a positive integer `calls`, got {budget!r}")
    used = len(context.loop.steps)
    return Verdict(
        "within_calls",
        used <= budget,
        f"used {used} tool call(s), budget {budget}",
    )


def check_evidence_verified(context: CheckContext) -> Verdict:
    """Every objective that settled also carried evidence.

    A harness check rather than a world check: acceptance is not a physical
    result, and a run that "succeeded" without evidence has measured nothing.
    """
    unused = _needs_run(context, "evidence_verified")
    if unused is not None:
        return unused
    steps = context.loop.steps
    settled = [s for s in steps if s.result is not None and s.result.state == "settled"]
    missing = [s.index for s in settled if not s.result.evidence_ok]
    return Verdict(
        "evidence_verified",
        not missing,
        f"{len(settled) - len(missing)}/{len(settled)} settled objective(s) carried evidence"
        + (f"; missing at steps {missing}" if missing else ""),
    )


def check_goal_succeeded(context: CheckContext) -> Verdict:
    """Every goal the run attempted got at least one successful objective.

    Answers from the step reports rather than from the world, because a goal can
    be satisfied by an objective whose result was later spent - the question is
    whether progress was ever made, not what is in the bag at the end.
    """
    if not context.loop.goal_attempts:
        return Verdict("goal_succeeded", False, "the run recorded no goal attempts")
    succeeded = {
        step.goal
        for step in context.loop.steps
        if step.goal is not None and step.result is not None and step.result.ok
    }
    uncredited = sorted(set(context.loop.goal_attempts) - succeeded)
    return Verdict(
        "goal_succeeded",
        not uncredited,
        (f"goals with no successful objective: {uncredited}" if uncredited else
         f"all {len(succeeded)} goal(s) with an attempt had a successful objective"),
    )


CHECKS: dict[str, CheckFn] = {
    "holds_item": check_holds_item,
    "best_tool_at_least": check_best_tool_at_least,
    "time_phase_is": check_time_phase_is,
    "no_death": check_no_death,
    "within_calls": check_within_calls,
    "evidence_verified": check_evidence_verified,
    "goal_succeeded": check_goal_succeeded,
}


def verdicts_for(scenario: Scenario, loop: LoopReport, facts: WorldFacts | None) -> list[Verdict]:
    """Evaluate every check, in the order the scenario lists them."""
    return [
        evaluate(check, CheckContext(facts=facts, loop=loop, params=check.params))
        for check in scenario.checks
    ]


def format_report(report: ScenarioReport) -> str:
    """The human-facing line set, so a run's result is readable at a glance."""
    lines = [f"scenario: {report.name}  ->  {'PASS' if report.passed else 'FAIL'}"]
    for verdict in report.verdicts:
        mark = "ok  " if verdict.passed else "FAIL"
        lines.append(f"  [{mark}] {verdict.kind}: {verdict.detail}")
    if not report.loop.ok:
        lines.append(f"  loop stopped: {report.loop.stop_reason}: {report.loop.detail}")
    if report.loop.goal_attempts:
        per_goal = ", ".join(f"{k}={v}" for k, v in sorted(report.loop.goal_attempts.items()))
        lines.append(f"  attempts per goal: {per_goal}")
    if report.usage:
        lines.append(
            f"  model: {report.usage.get('calls')} call(s), "
            f"{report.usage.get('promptTokens')} prompt + "
            f"{report.usage.get('completionTokens')} completion tokens, "
            f"${report.usage.get('costUsd')}"
        )
    return "\n".join(lines)


__all__ = [
    "CHECKS",
    "Check",
    "CheckContext",
    "Scenario",
    "ScenarioError",
    "ScenarioReport",
    "Verdict",
    "WorldFacts",
    "evaluate",
    "format_report",
    "situation_of",
    "verdicts_for",
    "world_facts",
]