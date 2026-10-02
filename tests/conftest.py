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
    return BudgetConfig(objective_ms=5_000, max_attempts_per_goal=3, max_consecutive_failures=2)


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