"""Goals: what a run is trying to achieve, and what has been tried at it.

A goal is an **item name**, not a procedure (D18). This was measured rather than
assumed: the reference run called ``view_crafting_requirements`` zero times in
388 calls, and reached a wooden pickaxe by naming items and letting
``craft_item`` walk the recipe tree. 23 craft calls produced 27 distinct items.
A goal expressed as "craft a stone pickaxe" therefore carries no recipe
knowledge for the harness to hold - it names a target and the server does the
decomposition.

What the harness *does* have to hold is the history. A decider that cannot see
that it has already tried to find logs sixteen times will try a seventeenth,
and the loop's step ceiling is a blunt instrument for stopping it. So the board
below is fed back into every decision, and
``budget.max_attempts_per_goal`` is applied against the goal the decider
actually named.

That config key was dead until this module: declared, defaulted, asserted
``> 0`` by a test, documented as a "per-goal ceiling", and read by nothing. A
key that a test checks the shape of is not a key that does anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


class GoalError(ValueError):
    """A goal set that cannot be used, named rather than guessed at."""


@dataclass(frozen=True)
class Goal:
    """One thing this run is trying to end up holding."""

    #: A Minecraft registry item name, exactly as ``craft_item`` wants it.
    #: ``bed`` is not one - the registry has ``white_bed`` and the server refuses
    #: the generic name with ``UNKNOWN_CRAFT_ITEMS``, measured.
    item: str
    count: int = 1
    #: Optional prose for the model. Never parsed, so it cannot become a hidden
    #: instruction channel; it is the same weight as a comment.
    note: str = ""

    def label(self) -> str:
        return f"{self.item}x{self.count}" if self.count != 1 else self.item


@dataclass(frozen=True)
class GoalSet:
    """An unordered set of goals.

    Unordered on purpose. Pre-ordering the goals would hand the decider a script
    and make the run a dressed-up :class:`~harness.decide.ScriptedDecider`: the
    interesting part of a baseline is the ordering mistakes, and a future
    cheaper decider would be handed the same unordered list and would miss the
    same dependencies, so the comparison stays fair.
    """

    goals: tuple[Goal, ...]

    @property
    def items(self) -> tuple[str, ...]:
        return tuple(goal.item for goal in self.goals)

    def __len__(self) -> int:
        return len(self.goals)

    def __iter__(self):
        return iter(self.goals)

    @classmethod
    def load(cls, path: Path | str) -> "GoalSet":
        """Read a goal file.

        Accepts either shape, because both read naturally at the call site::

            {"goals": ["wooden_pickaxe", {"item": "furnace", "count": 2}]}
            ["wooden_pickaxe", "furnace"]

        Unknown keys raise. A goal file with a key nobody reads is a goal file
        whose author believes something the run will not do.
        """
        goal_path = Path(path)
        try:
            raw_text = goal_path.read_text()
        except OSError as error:
            # Named rather than a bare FileNotFoundError: a scenario pointing at a
            # path that does not exist is a typo, and the reader needs to be told
            # which file rather than handed a stack trace from pathlib.
            raise GoalError(f"{goal_path} could not be read: {error}") from error
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as error:
            raise GoalError(f"{goal_path} is not valid JSON: {error}") from error
        return cls.from_any(raw, source=str(goal_path))

    @classmethod
    def from_any(cls, raw: Any, *, source: str = "goal set") -> "GoalSet":
        if isinstance(raw, dict):
            unknown = set(raw) - {"goals"}
            if unknown:
                raise GoalError(
                    f"{source}: unknown keys {sorted(unknown)}; expected ['goals']"
                )
            entries = raw.get("goals")
        elif isinstance(raw, list):
            entries = raw
        else:
            raise GoalError(
                f"{source}: expected a list of goals or {{\"goals\": [...]}}, got "
                f"{type(raw).__name__}"
            )
        if not isinstance(entries, list) or not entries:
            raise GoalError(f"{source}: no goals; a run with nothing to achieve is not a run")
        return cls(goals=tuple(_goal(entry, source, index) for index, entry in enumerate(entries)))


def _goal(entry: Any, source: str, index: int) -> Goal:
    where = f"{source}: goal {index}"
    if isinstance(entry, str):
        item, extra = entry, {}
    elif isinstance(entry, dict):
        unknown = set(entry) - {"item", "count", "note"}
        if unknown:
            raise GoalError(f"{where}: unknown keys {sorted(unknown)}; expected item/count/note")
        item = entry.get("item")
        extra = entry
    else:
        raise GoalError(f"{where}: expected a string or an object, got {type(entry).__name__}")

    if not isinstance(item, str) or not item.strip():
        raise GoalError(f"{where}: no item name; a goal is a registry item name (D18)")
    item = item.strip()
    count = extra.get("count", 1)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise GoalError(f"{where}: count must be an integer of at least 1, got {count!r}")
    note = extra.get("note", "")
    if not isinstance(note, str):
        raise GoalError(f"{where}: note must be a string, got {type(note).__name__}")
    return Goal(item=item, count=count, note=note)


@dataclass
class GoalBoard:
    """What has been attempted at each goal, for the decider to see.

    Owned by the loop and handed to the decider by reference, because the loop
    is what learns the outcome and the decider is what needs to know it. The
    decider still cannot write to it: progress is recorded from what the server
    returned, never from what the model hoped would happen.
    """

    goals: tuple[Goal, ...]
    max_attempts: int
    attempts: dict[str, int] = field(default_factory=dict)
    successes: dict[str, int] = field(default_factory=dict)
    last_outcome: dict[str, str] = field(default_factory=dict)

    @classmethod
    def of(cls, goals: GoalSet | None, *, max_attempts: int) -> "GoalBoard":
        return cls(goals=tuple(goals or ()), max_attempts=max_attempts)

    @property
    def enabled(self) -> bool:
        return bool(self.goals)

    def __len__(self) -> int:
        return len(self.goals)

    @property
    def items(self) -> tuple[str, ...]:
        """The goal item names, in declared order."""
        return tuple(goal.item for goal in self.goals)

    def known(self, item: str) -> bool:
        return any(goal.item == item for goal in self.goals)

    def exhausted(self, item: str) -> bool:
        return self.attempts.get(item, 0) >= self.max_attempts

    def open(self) -> tuple[Goal, ...]:
        """Goals still under the cap, in the order they were declared."""
        return tuple(goal for goal in self.goals if not self.exhausted(goal.item))

    def record(self, item: str, *, ok: bool, outcome: str) -> None:
        self.attempts[item] = self.attempts.get(item, 0) + 1
        if ok:
            self.successes[item] = self.successes.get(item, 0) + 1
        self.last_outcome[item] = outcome

    def view(self) -> list[dict[str, Any]]:
        """The goal list as the decider is shown it.

        Everything here is a fact about the run. ``attempts`` and
        ``exhausted`` are included because a decider that cannot see that a goal
        is finished will keep proposing it, and a decider that cannot see that
        one is exhausted will keep rediscovering that it fails.
        """
        return [
            {
                "item": goal.item,
                "count": goal.count,
                "note": goal.note,
                "attempts": self.attempts.get(goal.item, 0),
                "max_attempts": self.max_attempts,
                "exhausted": self.exhausted(goal.item),
                "succeeded": self.successes.get(goal.item, 0) > 0,
                "last_outcome": self.last_outcome.get(goal.item, ""),
            }
            for goal in self.goals
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "goals": [
                {
                    "item": goal.item,
                    "attempts": self.attempts.get(goal.item, 0),
                    "succeeded": self.successes.get(goal.item, 0) > 0,
                    "exhausted": self.exhausted(goal.item),
                }
                for goal in self.goals
            ],
            "maxAttempts": self.max_attempts,
        }


def count_carried(carried: Iterable[str], item: str) -> int:
    """How many of ``item`` the bounded carry list shows.

    ``StateReader`` renders stacks as ``name`` or ``namexN``, so the count is
    recovered by splitting on the last ``x`` that is followed by digits. Used by
    the checker, which reads the *full* inventory rather than this list - see
    :mod:`harness.scenario` - but the parsing is shared so the two cannot
    disagree about what a stack name means.
    """
    total = 0
    for stack in carried:
        name, _, count = stack.rpartition("x")
        if not name or not count.isdigit():
            name, count = stack, ""
        if name == item:
            total += int(count) if count else 1
    return total