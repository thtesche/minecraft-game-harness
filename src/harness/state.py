"""State reading.

Produces a compact decision vector from the server's own records rather than a
raw inventory dump. Two reasons:

* A 4,000-token state costs ~1.7 s on an Apple GPU, and accuracy falls to
  8-17/20 past ~4,000 tokens. Shrinking the state is a design fix, not a bigger
  ``max_len``.
* Derived features are cacheable and comparable across scenarios, which a
  positional JSON dump of the whole inventory is not.

Deterministic server-side analyses run before any model is consulted (D7): the
server already walks recipe trees and maps chunks, and asking a frontier model
to re-derive those spends a call on arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

from .errors import UnverifiedRead
from .mcp_client import McpClient

#: Bounded so the state stays inside the English checkpoint's window. This is a
#: deliberate ceiling, not an aspiration.
DEFAULT_MAX_INVENTORY_ITEMS = 12
DEFAULT_MAX_NEARBY = 4

#: Material order for tools and for armour, best last.
#:
#: The server publishes one row per class and states its own ``tier``; what it
#: does not publish is a *ranking* of those tiers, and its enum order is not one
#: - it lists the armour materials after netherite. So the order is stated here
#: as Minecraft's own material progression, which is a fact about the game rather
#: than an inference about the server. A tier absent from these lists is not
#: ranked at all: it is skipped and reported, never guessed into a position.
#:
#: Bound from ``mine-ai-mcp/src/world/tool-tiers.ts``, whose enum is
#: ``wooden stone iron golden diamond netherite leather chainmail turtle other none``.
TOOL_TIER_ORDER = ("wooden", "stone", "iron", "golden", "diamond", "netherite")
ARMOUR_TIER_ORDER = ("leather", "chainmail", "turtle", "golden", "iron", "diamond", "netherite")

#: Rows the server emits for every class whether or not anything is held. These
#: say "nothing here", which is a fact worth stating rather than omitting.
EMPTY_TIER = "none"


@dataclass
class StateVector:
    """The compact state a decision is made from."""

    health: float | None = None
    food: float | None = None
    time_phase: str | None = None
    position: dict[str, float] = field(default_factory=dict)
    on_ground: bool | None = None
    in_water: bool | None = None

    best_tool: str | None = None
    best_armour: str | None = None
    used_slots: int | None = None
    free_slots: int | None = None
    carried: list[str] = field(default_factory=list)

    nearest_hostile: str | None = None
    nearest_hostile_distance: float | None = None
    hostiles_within: int = 0
    nearest_drop: str | None = None

    #: Deterministic verdicts from the server, not model output.
    max_unassisted_drop: int | None = None
    has_water_bucket: bool | None = None
    has_scaffold: bool | None = None

    #: The most recent death the server has observed. A loop compares this across
    #: a step to tell "the objective failed" from "the bot died mid-objective",
    #: which are different events with different recoveries.
    last_death_at: str | None = None
    last_death_cause: str | None = None

    #: Why the state is not trustworthy, if it is not. Non-empty makes
    #: :attr:`trustworthy` false, which is why an unrankable equipment tier is
    #: recorded here rather than merely noted: this vector is what a decision is
    #: made from, and a decider that cannot rank what it is holding should be
    #: told so. Note that
    #: :class:`~harness.scenario.WorldFacts` splits the same idea in two - its
    #: ``unverified`` is only a section the harness could not parse, and an
    #: unrankable tier is a *warning* there, because a checker asked "is this at
    #: least stone" can still answer yes. Deliberate: a prompt is better served
    #: by pessimism than a verdict by pessimism.
    unverified: list[str] = field(default_factory=list)

    @property
    def state_hash(self) -> str:
        """A stable identity for "the state the model saw".

        This is the key the Memo cache is built on and the one field in
        ``DecisionRow`` that cannot be recovered after the fact, so it is written
        at read time or not at all.

        Quantised, and that is the whole design question here. Hashing the vector
        verbatim would produce a hash that never repeats: two reads a second apart
        differ in position, and ``observedAt`` differs every time. A cache keyed on
        that records one hit in every thousand rows and appears to do nothing,
        which is indistinguishable from a model that ignores the cache. So the bins
        are chosen at the granularity a decision could actually turn on:

        * health and food to the nearest whole point - the server reports half
          hearts, and the next decision does not turn on half a heart
        * position to the nearest 8 blocks, half a chunk
        * distances to the nearest block

        Discrete facts - what is carried, which tier is best, how many hostiles -
        are hashed exactly. Two states with the same hash are not guaranteed to be
        identical; they are guaranteed to be identical *in every respect the
        decision could turn on*, which is the property a cache needs and a stronger
        claim than this can honestly make.
        """
        return hashlib.sha256(
            json.dumps(self._hashable(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]

    def _hashable(self) -> dict[str, Any]:
        """The vector as the cache keys on it: binned, and no reading timestamp."""
        return {
            "health": _bin(self.health),
            "food": _bin(self.food),
            "time": self.time_phase,
            "position": {axis: _bin(value, 8) for axis, value in sorted(self.position.items())},
            "stance": {"on_ground": self.on_ground, "in_water": self.in_water},
            "gear": {"best_tool": self.best_tool, "best_armour": self.best_armour},
            "inventory": {
                "used": self.used_slots,
                "free": self.free_slots,
                "carried": self.carried,
            },
            "nearby": {
                "hostiles": self.hostiles_within,
                "hostile": self.nearest_hostile,
                "distance": _bin(self.nearest_hostile_distance),
                "drop": self.nearest_drop,
            },
            "mobility": {
                "drop": self.max_unassisted_drop,
                "bucket": self.has_water_bucket,
                "scaffold": self.has_scaffold,
            },
            # A state that failed to verify is a different state, and the whole
            # point of hashing it separately is that the loop must not treat an
            # unread world as the world it read a moment ago.
            "unverified": self.unverified,
            # Hashed exactly, not binned: it only changes when the bot dies, and a
            # bot that just died is in a materially different situation from one
            # that has not died this session.
            "last_death_at": self.last_death_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "health": self.health,
            "food": self.food,
            "time": self.time_phase,
            "position": self.position,
            "stance": {
                "on_ground": self.on_ground,
                "in_water": self.in_water,
            },
            "gear": {
                "best_tool": self.best_tool,
                "best_armour": self.best_armour,
            },
            "inventory": {
                "used": self.used_slots,
                "free": self.free_slots,
                "carried": self.carried,
            },
            "nearby": {
                "hostiles_within": self.hostiles_within,
                "nearest_hostile": self.nearest_hostile,
                "nearest_hostile_distance": self.nearest_hostile_distance,
                "nearest_drop": self.nearest_drop,
            },
            "mobility": {
                "max_unassisted_drop": self.max_unassisted_drop,
                "water_bucket": self.has_water_bucket,
                "scaffold": self.has_scaffold,
            },
            # Surfaced rather than left internal because a bot that just died is a
            # different situation, and a decider that cannot see that will keep
            # proposing whatever it would have proposed anyway.
            "survival": {
                "last_death_at": self.last_death_at,
                "last_death_cause": self.last_death_cause,
            },
            "unverified": self.unverified,
        }

    @property
    def trustworthy(self) -> bool:
        return not self.unverified


class StateReader:
    """Reads the live situation and derives a decision vector."""

    def __init__(
        self,
        client: McpClient,
        *,
        max_inventory_items: int = DEFAULT_MAX_INVENTORY_ITEMS,
        max_nearby: int = DEFAULT_MAX_NEARBY,
    ) -> None:
        self.client = client
        self.max_inventory_items = max_inventory_items
        self.max_nearby = max_nearby

    async def read(self) -> StateVector:
        """Read ``view_status`` and derive the vector.

        A parse failure is recorded in ``unverified`` and returned, never
        replaced by a default. A vector with invented numbers in it is worse
        than one that says it does not know.
        """
        reply = await self.client.call(
            "view_status", {}, rationale="Read live situation for the decision vector"
        )
        if reply.is_error:
            return StateVector(
                unverified=[f"view_status failed: {reply.data.get('error')}"]
            )

        situation = situation_of(reply)
        if situation is None:
            return StateVector(
                unverified=[
                    "view_status carried no situation in either reply shape: "
                    f"{sorted(reply.data)}"
                ]
            )

        return self._derive(situation)

    def _derive(self, situation: dict[str, Any]) -> StateVector:
        vector = StateVector()
        vector.unverified.extend(self._missing(situation))

        vitals = _section(situation, "vitals")
        if isinstance(vitals, dict):
            vector.health = _number(vitals.get("health"))
            vector.food = _number(vitals.get("food"))

        clock = _section(situation, "clock")
        if isinstance(clock, dict):
            vector.time_phase = clock.get("phase") or clock.get("timeOfDay")

        position = _section(situation, "position")
        if isinstance(position, dict):
            vector.position = {
                "x": _round(position.get("x")),
                "y": _round(position.get("y")),
                "z": _round(position.get("z")),
            }
            vector.on_ground = position.get("onGround")
            vector.in_water = position.get("inWater")

        tools = _section(situation, "tools")
        if isinstance(tools, dict):
            vector.best_tool, _, unranked = _best_item(tools.get("tools"), TOOL_TIER_ORDER)
            vector.best_armour, _, unranked_armour = _best_item(
                tools.get("armour"), ARMOUR_TIER_ORDER
            )
            vector.unverified.extend(unranked + unranked_armour)

        inventory = _section(situation, "inventory")
        if isinstance(inventory, dict):
            vector.used_slots = _integer(inventory.get("usedSlots"))
            vector.free_slots = _integer(inventory.get("freeSlots"))
            vector.carried = self._carried(inventory)

        nearby = _section(situation, "nearby")
        if isinstance(nearby, dict):
            hostiles = nearby.get("hostiles")
            if isinstance(hostiles, list):
                vector.hostiles_within = len(hostiles)
                if hostiles:
                    nearest = hostiles[0]
                    if isinstance(nearest, dict):
                        vector.nearest_hostile = nearest.get("name")
                        vector.nearest_hostile_distance = _round(nearest.get("distance"))
            drops = nearby.get("droppedItems")
            if isinstance(drops, list) and drops and isinstance(drops[0], dict):
                vector.nearest_drop = drops[0].get("name")

        mobility = _section(situation, "mobility")
        if isinstance(mobility, dict):
            vector.max_unassisted_drop = _integer(mobility.get("maximumDrop"))
            vector.has_water_bucket = bool(_integer(mobility.get("waterBuckets")))
            scaffold = mobility.get("scaffold")
            if isinstance(scaffold, dict):
                vector.has_scaffold = bool(scaffold.get("available"))

        death = _section(situation, "lastDeath")
        if isinstance(death, dict):
            vector.last_death_at = death.get("observedAt")
            vector.last_death_cause = death.get("cause")

        return vector

    def _carried(self, inventory: dict[str, Any]) -> list[str]:
        """Bounded carry list, most relevant first.

        Bounded because an unbounded inventory dump is both the token problem
        and the reason two states never hash equal.
        """
        stacks = inventory.get("stacks")
        if not isinstance(stacks, list):
            return []
        names: list[str] = []
        for stack in stacks:
            if isinstance(stack, dict):
                name = stack.get("name")
                count = _integer(stack.get("count"))
                if isinstance(name, str):
                    names.append(f"{name}x{count}" if count else name)
        return names[: self.max_inventory_items]

    def _missing(self, situation: dict[str, Any]) -> list[str]:
        """Sections the contract promises but this reading did not find.

        A shape change must fail loudly here rather than quietly produce a zero.
        """
        required = (
            "vitals",
            "clock",
            "position",
            "tools",
            "inventory",
            "nearby",
            "mobility",
        )
        return [f"missing section: {key}" for key in required if key not in situation]

    async def crafting_gap(self, items: dict[str, int]) -> dict[str, Any]:
        """Ask the server which leaves a recipe tree is missing.

        The server already analyses the tree. Re-deriving it with a frontier
        model spends a call on arithmetic (D7).
        """
        reply = await self.client.call(
            "view_crafting_requirements",
            {"items": items},
            rationale="Determine missing recipe leaves before committing to craft",
        )
        if reply.is_error:
            return {"error": reply.data.get("error"), "unverified": True}
        result = reply.result
        if result is None:
            return {"unverified": True, "reply": reply.data}
        return {"unverified": False, "output": result}


def situation_of(reply: ToolReply) -> dict[str, Any] | None:
    """The situation object, from whichever reply shape the server used.

    Public because the scenario checker needs the *same* parse and not a second
    one. A checker that walked the payload itself would be free to disagree with
    the state reader about where the situation lives, and a disagreement there is
    indistinguishable from "the bot did not get a pickaxe".

    ``view_status`` answers directly - ``data.result.situation`` - while an
    enveloped tool of the same name would nest it at
    ``data.output.result.situation``. Both spellings are accepted, and neither is
    assumed, because the split between direct and foreground tools belongs to
    the server and changes without notice: ``foreground`` is not published on the
    wire, and only the presence of ``submission_id`` in an advertised schema
    hints at it.

    Returning ``None`` for both is the important case. It becomes ``unverified``
    rather than an empty world.
    """
    for path in (("result", "situation"), ("output", "result", "situation")):
        node: Any = reply.data
        for key in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if isinstance(node, dict):
            return node
    return None


def _bin(value: float | None, width: int = 1) -> int | None:
    """Round to the nearest ``width``, half away from zero, or stay ``None``.

    ``None`` is preserved rather than folded into a number: an unread field and a
    field reading zero are different facts, and the cache that cannot tell them
    apart will serve the wrong answer while looking correct.
    """
    if value is None:
        return None
    return math.floor(value / width + 0.5) * width if value >= 0 else -math.floor(-value / width + 0.5) * width


def _section(situation: dict[str, Any], key: str) -> Any:
    return situation.get(key)


def _number(value: Any, digits: int = 2) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), digits)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _round(value: Any, digits: int = 1) -> float | None:
    """Round once.

    Rounding to an intermediate precision and then again compounds two
    half-way errors into the wrong answer: 71.25 to 1 decimal must not go via
    71.2.
    """
    return _number(value, digits)


def _best_item(
    entries: Any, order: tuple[str, ...]
) -> tuple[str | None, str | None, list[str]]:
    """Best item held, from the server's one-row-per-class tool table.

    ``situation.tools`` is ``{"tools": [...], "armour": [...]}`` with one entry
    per class - twelve tool classes and four armour pieces, all present whether
    or not anything is held, and an entry that holds nothing carries
    ``tier: "none"`` and ``item: null``. So an absent answer is a fact about the
    world (nothing held) and not a missing read.

    Returns ``(item, tier, unranked)``. A row whose tier this function does not
    know is reported rather than assumed, because ranking it wrongly would report
    the wrong best tool with total confidence.
    """
    if not isinstance(entries, list):
        return None, None, []

    best: tuple[int, str, str] | None = None
    unranked: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item = entry.get("item")
        tier = entry.get("tier")
        if not isinstance(item, str):
            continue  # nothing held in this class
        if tier == EMPTY_TIER:
            continue
        if tier not in order:
            unranked.append(f"{item} (tier {tier!r} not in the harness tier order)")
            continue
        rank = order.index(tier)
        if best is None or rank > best[0]:
            best = (rank, item, tier)
    return (best[1] if best else None, best[2] if best else None, unranked)


def best_equipment(situation: dict[str, Any], key: str) -> tuple[str | None, str | None, list[str]]:
    """Best item and tier held in one of the two tables, from a situation.

    Public because the scenario checker ranks equipment the same way the prompt
    does. A checker with its own ranking could disagree with the state reader
    about which tool is best, and that disagreement would be indistinguishable
    from the bot not holding the tool at all. Keyed by ``"tools"`` or
    ``"armour"``, each with its own tier order.
    """
    tools = _section(situation, "tools")
    if not isinstance(tools, dict):
        return None, None, []
    order = ARMOUR_TIER_ORDER if key == "armour" else TOOL_TIER_ORDER
    return _best_item(tools.get(key), order)