"""Tests for state derivation.

The rule under test throughout: a section the contract promises but the reading
did not find is reported, never replaced with a zero.
"""

from __future__ import annotations

from conftest import FakeClient

from harness.state import StateReader, StateVector


def situation(**overrides):
    base = {
        "vitals": {"health": 18.0, "food": 14.0, "saturation": 5.0, "airSupplyTicks": 300, "burning": False},
        "mobility": {
            "maximumDrop": 3,
            "waterBuckets": 0,
            "bucketDrop": {"available": False, "maximumBlocks": 0, "blockedBy": ["no_water_bucket"]},
            "fallSave": {"available": False, "blockedBy": ["no_water_bucket"]},
            "scaffold": {"available": False, "item": None, "blocks": 0, "blockedBy": []},
        },
        "activity": {"owner": "idle", "activeAction": None},
        "clock": {"timeOfDay": 6000, "phase": "day", "ticksUntilChange": 4000, "minutesUntilChange": 200},
        "position": {"x": -57.551, "y": 71.0, "z": -0.632, "chunkX": -4, "chunkZ": -1,
                     "headingDegrees": 90.0, "onGround": True, "inWater": False, "inLava": False},
        "inventory": {"usedSlots": 4, "freeSlots": 32,
                      "stacks": [{"name": "dirt", "count": 10}, {"name": "cobblestone", "count": 5}]},
        "tools": {
            "tools": [
                {"class": "pickaxe", "tier": "none", "item": None, "slot": None,
                 "durabilityLeft": None, "maximumDurability": None},
                {"class": "axe", "tier": "none", "item": None, "slot": None,
                 "durabilityLeft": None, "maximumDurability": None},
            ],
            "armour": [
                {"class": "helmet", "tier": "none", "item": None, "slot": None,
                 "durabilityLeft": None, "maximumDurability": None},
            ],
        },
        "nearby": {
            "rangeBlocks": 16.0,
            "players": [],
            "hostiles": [{"name": "zombie", "distance": 3.2,
                          "position": {"x": -55.0, "y": 72.0, "z": -3.0}}],
            "mobs": [],
            "droppedItems": [{"name": "iron_ore", "count": 3, "distance": 2.1,
                              "position": {"x": -58.0, "y": 71.0, "z": -1.0}}],
        },
    }
    base.update(overrides)
    return base


def reply_with(sit, *, is_error=False, state="settled"):
    from harness.mcp_client import ToolReply

    return ToolReply(
        is_error=is_error,
        data={
            "state": state,
            "actionId": "a1",
            "output": {"action": "view_status", "result": {"kind": "read", "status": "succeeded", "situation": sit}},
        },
        notifications={},
    )


async def test_derives_a_compact_vector():
    client = FakeClient(script={"view_status": [reply_with(situation())]})
    vector = await StateReader(client).read()

    assert vector.trustworthy
    assert vector.health == 18.0
    assert vector.food == 14.0
    assert vector.time_phase == "day"
    assert vector.on_ground is True
    assert vector.used_slots == 4
    assert vector.carried == ["dirtx10", "cobblestonex5"]
    assert vector.nearest_hostile == "zombie"
    assert vector.nearest_hostile_distance == 3.2
    assert vector.hostiles_within == 1
    assert vector.max_unassisted_drop == 3


async def test_tier_none_reports_no_tool_rather_than_a_placeholder():
    """Every class tier='none' is a fact worth stating, not an omission."""
    client = FakeClient(script={"view_status": [reply_with(situation())]})
    vector = await StateReader(client).read()

    assert vector.best_tool is None
    assert vector.best_armour is None
    assert vector.has_water_bucket is False


def _row(kind: str, tier: str, item: str | None) -> dict:
    return {"class": kind, "tier": tier, "item": item, "slot": 36,
            "durabilityLeft": 100, "maximumDurability": 250}


async def test_the_best_tool_is_the_highest_tier_actually_held():
    """Rows come one per class; the answer is the best one that holds something."""
    sit = situation(tools={
        "tools": [_row("pickaxe", "stone", "stone_pickaxe"),
                  _row("axe", "iron", "iron_axe"),
                  _row("sword", "none", None)],
        "armour": [_row("helmet", "iron", "iron_helmet"),
                   _row("boots", "leather", "leather_boots")],
    })
    vector = await StateReader(FakeClient(script={"view_status": [reply_with(sit)]})).read()

    assert vector.best_tool == "iron_axe"
    # Armour ranks separately: leather really is below iron, and the server's
    # enum order (which puts armour materials after netherite) is not a ranking.
    assert vector.best_armour == "iron_helmet"
    assert vector.trustworthy


async def test_a_tier_outside_the_declared_order_is_reported_not_ranked():
    """An unknown tier must not be guessed into a position and reported as best."""
    sit = situation(tools={
        "tools": [_row("pickaxe", "turtle", "turtle_pickaxe")],
        "armour": [_row("helmet", "none", None)],
    })
    vector = await StateReader(FakeClient(script={"view_status": [reply_with(sit)]})).read()

    assert vector.best_tool is None
    assert not vector.trustworthy
    assert "not in the harness tier order" in vector.unverified[0]


async def test_a_null_tools_section_does_not_crash_the_reading():
    """The live host publishes `tools: null` when the table is unavailable."""
    sit = situation(tools=None)
    vector = await StateReader(FakeClient(script={"view_status": [reply_with(sit)]})).read()

    assert vector.best_tool is None
    assert vector.health == 18.0, "the rest of the vector is still read"


async def test_missing_section_is_reported_not_defaulted():
    """A shape change must fail loudly rather than produce a zero."""
    partial = situation()
    del partial["mobility"]

    client = FakeClient(script={"view_status": [reply_with(partial)]})
    vector = await StateReader(client).read()

    assert not vector.trustworthy
    assert vector.unverified == ["missing section: mobility"]


async def test_unparseable_reply_is_unverified():
    from harness.mcp_client import ToolReply

    client = FakeClient(
        script={"view_status": [ToolReply(is_error=True, data={"state": "refused", "error": "no bot"},
                                          notifications={})]}
    )
    vector = await StateReader(client).read()

    assert not vector.trustworthy
    assert "no bot" in vector.unverified[0]


async def test_output_without_situation_is_unverified():
    from harness.mcp_client import ToolReply

    client = FakeClient(
        script={"view_status": [ToolReply(is_error=False, data={"state": "settled", "output": {"result": {}}},
                                          notifications={})]}
    )
    vector = await StateReader(client).read()

    assert not vector.trustworthy


async def test_inventory_is_bounded():
    stacks = [{"name": f"item_{i}", "count": i} for i in range(40)]
    client = FakeClient(script={"view_status": [reply_with(situation(inventory={"usedSlots": 40,
                                                                             "freeSlots": 0,
                                                                             "stacks": stacks}))]})
    vector = await StateReader(client, max_inventory_items=8).read()

    assert len(vector.carried) == 8


async def test_vector_is_serialisable_and_small():
    client = FakeClient(script={"view_status": [reply_with(situation())]})
    vector = await StateReader(client).read()
    payload = vector.to_dict()

    assert set(payload) == {
        "health", "food", "time", "position", "stance",
        "gear", "inventory", "nearby", "mobility", "survival", "unverified",
    }
    assert payload["position"]["x"] == -57.6


async def test_rounding_happens_once():
    """Rounding twice compounds two half-way errors into the wrong value.

    71.25 to one decimal must not be reached by way of 71.2.
    """
    client = FakeClient(script={"view_status": [reply_with(situation(position={"x": 0.0, "y": 71.25, "z": 0.0}))]})
    vector = await StateReader(client).read()

    assert vector.position["y"] == 71.2


# --- state_hash -------------------------------------------------------------
#
# The hash is the Memo cache key, and the one ledger field that cannot be
# recovered after the fact. Its whole design question is quantisation: hashed
# verbatim it never repeats, because two reads a second apart differ in position,
# and a cache that never hits is indistinguishable from a model that ignores it.


async def read_hash(**overrides) -> str:
    client = FakeClient(script={"view_status": [reply_with(situation(**overrides))]})
    return (await StateReader(client).read()).state_hash


def standing(**overrides):
    return situation(position={"x": -57.551, "y": 71.0, "z": -0.632,
                               "onGround": True, "inWater": False}, **overrides)


async def test_two_reads_a_second_apart_share_a_hash():
    """The property the cache depends on: drift inside a bin is not a new state."""
    drifted = situation(position={"x": -57.480, "y": 71.0, "z": -0.601,
                                  "onGround": True, "inWater": False})
    assert await read_hash(**standing()) == await read_hash(**drifted)


async def test_moving_a_whole_chunk_moves_the_hash():
    """The bin is half a chunk wide, so a chunk of travel is a different place."""
    far = situation(position={"x": -40.0, "y": 71.0, "z": -0.632,
                              "onGround": True, "inWater": False})
    assert await read_hash(**standing()) != await read_hash(**far)


async def test_a_decisive_change_in_vitals_moves_the_hash():
    """Half a heart of drift is noise; three hearts is a different situation."""
    hurt = await read_hash(vitals={"health": 17.0, "food": 18.0, "saturation": 5.0, "burning": False})
    worse = await read_hash(vitals={"health": 14.0, "food": 18.0, "saturation": 5.0, "burning": False})
    assert hurt != worse


async def test_a_carried_item_moving_moves_the_hash():
    """Discrete facts are hashed exactly, not binned: one seed matters."""
    holding = await read_hash(inventory={"usedSlots": 1, "freeSlots": 35,
                                         "stacks": [{"name": "wheat_seeds", "count": 1}]})
    empty = await read_hash(inventory={"usedSlots": 0, "freeSlots": 36, "stacks": []})
    assert holding != empty


async def test_a_new_death_moves_the_hash():
    """A bot that just died is materially not the bot that has not."""
    alive = await read_hash()
    died = await read_hash(lastDeath={"observedAt": "2026-10-02T14:00:00.000Z",
                                      "cause": "MineAI was slain by Zombie"})
    assert alive != died


async def test_the_last_death_is_read_from_the_situation():
    client = FakeClient(script={"view_status": [reply_with(situation(lastDeath={
        "observedAt": "2026-10-02T12:30:36.841Z",
        "cause": "MineAI was slain by Zombie",
        "dimension": "overworld",
    }))]})
    vector = await StateReader(client).read()

    assert vector.last_death_at == "2026-10-02T12:30:36.841Z"
    assert vector.last_death_cause == "MineAI was slain by Zombie"
    # A world with no death yet has no lastDeath, which is not a missing section:
    # requiring it would report an unverified world on every fresh session.
    assert vector.unverified == []
    assert vector.to_dict()["survival"] == {
        "last_death_at": "2026-10-02T12:30:36.841Z",
        "last_death_cause": "MineAI was slain by Zombie",
    }


def test_an_unread_field_is_not_folded_into_a_zero():
    """Otherwise the cache serves a missing reading as though it were a real one."""
    assert StateVector(health=None).state_hash != StateVector(health=0).state_hash


def test_an_unverified_state_hashes_apart_from_the_same_numbers_trusted():
    """Deciding from an unread world is a different decision and must look like one."""
    assert (
        StateVector(unverified=["missing section: mobility"]).state_hash
        != StateVector().state_hash
    )


def test_the_hash_is_short_and_stable():
    """Short so a ledger row stays readable; stable so a row means the same thing
    when it is read back weeks later, which is what the Phase 3 eval needs."""
    digest = StateVector(health=20.0).state_hash
    assert digest == StateVector(health=20.0).state_hash
    assert len(digest) == 16
    assert set(digest) <= set("0123456789abcdef")
