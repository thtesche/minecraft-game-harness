"""Tests for state derivation.

The rule under test throughout: a section the contract promises but the reading
did not find is reported, never replaced with a zero.
"""

from __future__ import annotations

from conftest import FakeClient

from harness.state import StateReader


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
        "tools": {"best": {"name": None, "tier": "none"}},
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
    assert vector.has_water_bucket is False


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
        "gear", "inventory", "nearby", "mobility", "unverified",
    }
    assert payload["position"]["x"] == -57.6


async def test_rounding_happens_once():
    """Rounding twice compounds two half-way errors into the wrong value.

    71.25 to one decimal must not be reached by way of 71.2.
    """
    client = FakeClient(script={"view_status": [reply_with(situation(position={"x": 0.0, "y": 71.25, "z": 0.0}))]})
    vector = await StateReader(client).read()

    assert vector.position["y"] == 71.2