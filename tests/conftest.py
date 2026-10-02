"""Shared test fixtures.

The runner's protocol rules are tested against a fake client, not a live
Minecraft server. That is deliberate: the rules are the part with no excuse for
being wrong, and a test that needs a running world is a test that does not run.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness.config import BudgetConfig, LedgerConfig, McpConfig  # noqa: E402
from harness.mcp_client import ToolReply  # noqa: E402

RUN_ID = "test-run"


def settled(action_id: str, status: str = "succeeded", **output: Any) -> ToolReply:
    """A settled reply carrying evidence, as the contract shapes it."""
    body: dict[str, Any] = {
        "action": "collect_block",
        "durationMs": 1200,
        "request": {"evidence": {"baseline": {"dirt": 3}, "completion": "inventory rose"}},
        "result": {"kind": "task", "status": status},
    }
    body.update(output)
    return ToolReply(
        is_error=status not in ("succeeded",),
        data={"state": "settled", "wakeReason": "settled", "actionId": action_id, "output": body},
        notifications={},
    )


def pending(action_id: str) -> ToolReply:
    return ToolReply(
        is_error=False,
        data={"state": "pending", "wakeReason": "timeout", "actionId": action_id, "progress": {}},
        notifications={},
    )


def accepted(action_id: str) -> ToolReply:
    return ToolReply(
        is_error=False,
        data={"state": "accepted", "actionId": action_id, "action": "collect_block", "admittedAt": "now"},
        notifications={},
    )


def refused(code: str, **extra: Any) -> ToolReply:
    return ToolReply(
        is_error=True,
        data={"state": "refused", "code": code, "error": f"refused {code}", **extra},
        notifications={},
    )


def body_owned(action: str = "hostile_reflex") -> ToolReply:
    """A direct ``view_status`` reporting the body is held by a survival reflex."""
    return ToolReply(
        is_error=False,
        data={
            "action": "view_status",
            "durationMs": 1,
            "result": {
                "kind": "read",
                "status": "succeeded",
                "situation": {
                    "activity": {"owner": "takeover",
                                 "activeAction": {"action": action, "startedAt": "now"}},
                },
            },
        },
        notifications={},
    )


def body_free() -> ToolReply:
    """A direct ``view_status`` reporting ``owner: "idle"`` and no active action."""
    return ToolReply(
        is_error=False,
        data={
            "action": "view_status",
            "durationMs": 1,
            "result": {
                "kind": "read",
                "status": "succeeded",
                "situation": {"activity": {"owner": "idle", "activeAction": None}},
            },
        },
        notifications={},
    )


def situation(**overrides: Any) -> dict[str, Any]:
    """A complete, trustworthy situation: every section the contract promises.

    Every section present and no unranked tier, so ``StateReader`` returns a
    vector with an empty ``unverified``. Individual tests delete or corrupt the one
    section they are about.
    """
    base: dict[str, Any] = {
        "vitals": {"health": 20.0, "food": 18.0, "saturation": 5.0, "burning": False},
        "mobility": {
            "maximumDrop": 3,
            "waterBuckets": 0,
            "scaffold": {"available": False, "item": None, "blocks": 0},
        },
        "activity": {"owner": "idle", "activeAction": None},
        "clock": {"timeOfDay": 6000, "phase": "day"},
        "position": {"x": -57.551, "y": 71.0, "z": -0.632, "onGround": True, "inWater": False},
        "inventory": {"usedSlots": 4, "freeSlots": 32,
                      "stacks": [{"name": "dirt", "count": 10}]},
        "tools": {
            "tools": [{"class": "pickaxe", "tier": "wooden", "item": "wooden_pickaxe", "slot": 0,
                       "durabilityLeft": 50, "maximumDurability": 59}],
            "armour": [{"class": "helmet", "tier": "leather", "item": "leather_helmet", "slot": 1,
                        "durabilityLeft": 40, "maximumDurability": 55}],
        },
        "nearby": {"hostiles": [], "droppedItems": []},
        "botId": "probe",
        "dimension": "overworld",
        "gameMode": "survival",
        "observedAt": "2026-10-02T13:00:00.000Z",
    }
    base.update(overrides)
    return base


def status(state: dict[str, Any] | None = None) -> ToolReply:
    """A direct ``view_status`` reply carrying a trustworthy situation."""
    return ToolReply(
        is_error=False,
        data={
            "action": "view_status",
            "durationMs": 1,
            "result": {"kind": "read", "status": "succeeded", "situation": state or situation()},
        },
        notifications={},
    )


@dataclass
class FakeClient:
    """Scripted MCP client.

    ``script`` maps a tool name to the replies it returns, in order, repeating
    the last one once exhausted. ``calls`` records every ``call`` so tests can
    assert on protocol behaviour rather than only on outcomes.
    """

    script: dict[str, list[ToolReply]] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    health_payload: dict[str, Any] = field(default_factory=dict)

    async def call(
        self,
        tool: str,
        arguments: dict[str, Any] | None = None,
        *,
        rationale: str,
        read_timeout_ms: int | None = None,
    ) -> ToolReply:
        self.calls.append((tool, dict(arguments or {})))
        replies = self.script.get(tool)
        if not replies:
            raise AssertionError(f"no scripted reply for {tool}")
        if len(replies) > 1:
            return replies.pop(0)
        return replies[0]

    async def health(self) -> dict[str, Any]:
        return self.health_payload

    async def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": tool} for tool in self.script]

    def args_for(self, tool: str) -> list[dict[str, Any]]:
        return [args for name, args in self.calls if name == tool]


@pytest.fixture
def mcp_config() -> McpConfig:
    return McpConfig(initial_wait_ms=10, poll_ms=10, max_polls=5)


@pytest.fixture
def budget() -> BudgetConfig:
    # gate_wait_ms=0 by default: a test that means to wait for the body says so
    # explicitly, so an accidental wait cannot hide behind a passing bound.
    return BudgetConfig(objective_ms=5_000, max_attempts_per_goal=3,
                        max_consecutive_failures=2, gate_wait_ms=0)


@pytest.fixture
def ledger_config(tmp_path: Path) -> LedgerConfig:
    return LedgerConfig(path=tmp_path / "ledger.jsonl", sqlite_path=tmp_path / "ledger.sqlite")


@pytest.fixture
def clock() -> Any:
    """Monotonic clock that advances one millisecond per call."""

    class Clock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            self.t += 0.001
            return self.t

    return Clock()