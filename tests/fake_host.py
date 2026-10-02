"""A faithful stand-in for the mine-ai-mcp host.

The unit tests drive a scripted fake client, which proves the runner's logic but
not the transport: the mcp SDK call signature, the ``structuredContent`` shape,
and the ``/health`` read are all unverified until something real is on the
other end. This module runs an actual MCP server over Streamable HTTP and
reimplements the protocol rules from the server's contract:

* every foreground call needs a unique ``submission_id``
* retrying the identical call returns the original action rather than starting
  a second one; changing either the arguments or the id is a SUBMISSION_CONFLICT
* one objective is admitted at a time
* RESULT_NOT_RETRIEVED refuses the next submission until the owed result is
  collected
* an initial wait that expires answers ``pending``, not ``accepted``

It is a stand-in for the transport and the protocol, not for Minecraft. Nothing
here reports a physical world outcome that a test then believes.

Two reply shapes
----------------

The server answers in two shapes, in the same session, and the difference is not
cosmetic:

* **foreground** tools get the submission envelope -
  ``{state, actionId, output}``
* **information and control** tools answer directly -
  ``{action, durationMs, result, survival, survivalPolicy}``, with no ``state``

``view_status`` is a direct tool. This file once wrapped it in the envelope, and
because the client was written against this file, every test agreed with the
client and both were wrong: ``harness state`` reported an empty world against
the live host while the suite stayed green. A stand-in has to be pinned to the
*server's* contract, never to the client's current assumptions - otherwise it
amplifies exactly the bug it exists to catch.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import socket
from dataclasses import dataclass, field
from typing import Any

import uvicorn

def _tool_row(kind: str, tier: str = "none", item: str | None = None) -> dict[str, Any]:
    """One row of the server's tool table.

    ``situation.tools`` carries one row per class - twelve tool classes, four
    armour pieces - whether or not anything is held, with ``tier: "none"`` and a
    null ``item`` when the slot is empty. This fixture once published
    ``{"best": {"name": ..., "tier": ...}}``, a shape the server does not use,
    which is how a null ``tools.best`` reached ``_best_tier`` and crashed
    ``harness state`` against a live host while every test stayed green.
    """
    return {
        "class": kind,
        "tier": tier,
        "item": item,
        "slot": None if item is None else 36,
        "durabilityLeft": None if item is None else 100,
        "maximumDurability": None if item is None else 250,
    }


#: The twelve tool classes and four armour pieces the server always publishes.
TOOL_CLASSES = ("pickaxe", "shovel", "axe", "sword", "hoe", "shears", "bow",
                "shield", "bucket", "water_bucket", "lava_bucket", "powder_snow_bucket")
ARMOUR_CLASSES = ("helmet", "chestplate", "leggings", "boots")

SITUATION: dict[str, Any] = {
    "vitals": {"health": 20.0, "food": 18.0, "saturation": 5.0, "airSupplyTicks": 300, "burning": False},
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
        "tools": [_tool_row(kind) for kind in TOOL_CLASSES],
        "armour": [_tool_row(kind) for kind in ARMOUR_CLASSES],
    },
    "nearby": {
        "rangeBlocks": 16.0, "players": [], "hostiles": [], "mobs": [], "droppedItems": [],
    },
}


@dataclass
class _Action:
    action_id: str
    tool: str
    arguments: dict[str, Any]
    # How many wait_for_action calls report pending before settling.
    pending_polls: int = 0
    status: str = "succeeded"


@dataclass
class FakeHost:
    """Protocol state, held the way the real session service holds it."""

    actions: dict[str, _Action] = field(default_factory=dict)
    by_submission: dict[str, _Action] = field(default_factory=dict)
    admitted: str | None = None
    retrieved: set[str] = field(default_factory=set)
    calls: list[dict[str, Any]] = field(default_factory=list)
    health_calls: int = 0
    counter: int = 0

    #: A survival reflex holding the body. While set, foreground submissions are
    #: refused with ACTION_BUSY and *no* action id, which is what the real
    #: `hostile_reflex` produces - there is no action to retrieve, only ownership
    #: to wait out. Not drainable, but finite: `reflex_reads_left` counts the
    #: view_status calls until it lets go.
    reflex: str | None = None
    reflex_reads_left: int = 0

    def new_id(self) -> str:
        self.counter += 1
        return f"act-{self.counter}"

    def activity(self) -> dict[str, Any]:
        """What view_status reports about ownership."""
        if self.reflex:
            return {"owner": "takeover",
                    "activeAction": {"action": self.reflex, "startedAt": "2026-01-01T00:00:00.000Z"}}
        active = self.admitted if self.admitted and self.admitted not in self.retrieved else None
        if active:
            return {"owner": "foreground",
                    "activeAction": {"action": self.actions[active].tool, "startedAt": None}}
        return {"owner": "idle", "activeAction": None}


def build_app(host: FakeHost):
    """An MCP server that speaks the mine-ai-mcp protocol subset."""
    import mcp.types as types
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name="fake-mine-ai-mcp", version="0.0.0", instructions="fake")

    def settled_output(tool: str, action: _Action) -> dict[str, Any]:
        status = action.status
        result: dict[str, Any] = {"kind": "task", "status": status}
        if status != "succeeded":
            result["error"] = f"{tool} ended {status}"
        return {
            "action": tool,
            "durationMs": 1200,
            # The observation the verdict rested on. Absent means unverified.
            "request": {"evidence": {"baseline": {"dirt": 3}, "completion": "inventory rose"}},
            "result": result,
        }

    def is_error_for(data: dict[str, Any]) -> bool:
        """Does this reply represent a failure, in either shape?

        Direct replies carry no protocol ``state``, so the result object's own
        ``status`` is the only thing left to judge them by.
        """
        output = data.get("output")
        result = output.get("result") if isinstance(output, dict) else None
        if not isinstance(result, dict):
            direct = data.get("result")
            result = direct if isinstance(direct, dict) else None
        status = result.get("status") if isinstance(result, dict) else None
        if data.get("error"):
            return True
        return data.get("state") in ("refused", "storage_failed") or status in ("failed", "cancelled")

    def envelope(data: dict[str, Any]):
        """The exact reply shape the real server builds for format=json.

        A real ``CallToolResult``, not a plain dict: a dict return value makes the
        SDK invent a text body from the return annotation, which would hide the
        envelope this file exists to pin down.
        """
        error = is_error_for(data)
        output = data.get("output")
        result = output.get("result") if isinstance(output, dict) else None
        text = str(data.get("error") or (result.get("error") if isinstance(result, dict) else None) or "")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)] if error and text else [],
            structuredContent={"response": {"format": "json", "data": data}, "notifications": {}},
            isError=error,
        )

    @server.tool(name="view_status", description="Read the live situation.")
    async def view_status(rationale: str = "", response_format: str = "markdown"):
        # A *direct* reply, not the submission envelope. view_status is an
        # information tool, and the server answers those immediately with
        # {action, durationMs, result, survival, survivalPolicy} and no `state`
        # at all. This stand-in used to wrap it in the envelope, which is how a
        # client could agree with this file and still fail against the real host.
        sit = copy.deepcopy(SITUATION)
        if host.reflex_reads_left > 0:
            host.reflex_reads_left -= 1
            if host.reflex_reads_left == 0:
                host.reflex = None  # the reflex ends and returns the body
        sit["activity"] = host.activity()
        data = {
            "action": "view_status",
            "durationMs": 1,
            "result": {"kind": "read", "status": "succeeded", "situation": sit},
            "survival": {"summary": "none", "dangers": []},
            "survivalPolicy": {"revision": "fake:0", "effective": {}},
        }
        return envelope(data)

    @server.tool(name="read_recent_events", description="Read recent events.")
    async def read_recent_events(pad_bytes: int = 0, rationale: str = "",
                                 response_format: str = "markdown"):
        """A direct read whose reply can be made arbitrarily large.

        ``pad_bytes`` is a test affordance, not a server feature. It exists
        because httpx2 refuses any server-sent event above 1 MiB and the real
        host's `tools/list` is 2.91 MiB - a ceiling nothing in this suite would
        otherwise ever approach, which is precisely how that bug shipped.
        """
        data = {
            "action": "read_recent_events",
            "durationMs": 1,
            "result": {
                "kind": "read",
                "status": "succeeded",
                "events": [{"cursor": i, "text": "x" * 64} for i in range(8)],
                "pad": "p" * max(0, pad_bytes),
            },
            "survival": {"summary": "none", "dangers": []},
        }
        return envelope(data)

    @server.tool(name="collect_block", description="Collect one block type.")
    async def collect_block(submission_id: str, block_name: str, wait_timeout_ms: int | None = None,
                            response_format: str = "markdown", rationale: str = ""):
        host.calls.append({"tool": "collect_block", "arguments": dict(block_name=block_name,
                                                                        submission_id=submission_id,
                                                                        wait_timeout_ms=wait_timeout_ms)})

        # A retry of the identical call is a recovery, not a new objective.
        existing = host.by_submission.get(submission_id)
        if existing is not None:
            if existing.arguments.get("block_name") != block_name:
                return envelope({"state": "refused", "code": "SUBMISSION_CONFLICT",
                                 "error": "submission_id reused with different arguments"})
            if wait_timeout_ms is None:
                return envelope({"state": "accepted", "actionId": existing.action_id,
                                 "action": existing.tool, "progress": {}, "request": None,
                                 "awaitingResult": None, "storageError": None})
            return _wait(host, existing, wait_timeout_ms)

        if host.reflex:
            # The survival reflex owns the body, so there is no action id to
            # drain. This is the refusal the real host returns while its
            # hostile_reflex runs, and the only thing that clears it is waiting.
            return envelope({"state": "refused", "code": "ACTION_BUSY",
                             "error": f"No new action started; body owner: {host.reflex}. "
                                      "Wait for physical ownership to become available."})

        if host.admitted is not None and host.admitted not in host.retrieved:
            active = host.actions[host.admitted]
            return envelope({"state": "refused", "code": "RESULT_NOT_RETRIEVED",
                             "error": "No new action started. Call wait_for_action with "
                                      "unretrievedActionId to retrieve the preceding full result.",
                             "unretrievedActionId": host.admitted, "action": active.tool})

        action = _Action(action_id=host.new_id(), tool="collect_block", arguments={"block_name": block_name})
        host.actions[action.action_id] = action
        host.by_submission[submission_id] = action
        host.admitted = action.action_id

        if wait_timeout_ms is None:
            return envelope({"state": "accepted", "actionId": action.action_id, "action": action.tool,
                             "progress": {}, "request": None, "awaitingResult": None, "storageError": None})
        return _wait(host, action, wait_timeout_ms)

    def _wait(host: FakeHost, action: _Action, timeout_ms: int):
        """Model of ``AsyncActions.wait``.

        Note what is absent: there is no "already retrieved" refusal. Waiting on
        an action whose result was already collected returns the terminal output
        again. That is the retry-recovery path - a client whose reply was lost
        repeats the identical call and gets the original answer back rather than
        a second action.
        """
        if action.pending_polls > 0:
            action.pending_polls -= 1
            return envelope({"state": "pending", "wakeReason": "timeout", "actionId": action.action_id,
                             "progress": {"action": action.tool}, "request": None})
        host.retrieved.add(action.action_id)
        return envelope({"state": "settled", "wakeReason": "settled", "actionId": action.action_id,
                         "output": settled_output(action.tool, action)})

    @server.tool(name="wait_for_action", description="Retrieve an admitted action's result.")
    async def wait_for_action(action_id: str, timeout_ms: int, response_format: str = "markdown",
                              rationale: str = ""):
        host.calls.append({"tool": "wait_for_action", "arguments": {"action_id": action_id,
                                                                   "timeout_ms": timeout_ms}})
        if not isinstance(timeout_ms, int) or timeout_ms < 0 or timeout_ms > 120_000:
            return envelope({"state": "refused", "code": "INVALID_ARGUMENTS",
                             "error": "timeout_ms must be an integer from 0 to 120000."})
        action = host.actions.get(action_id)
        if action is None:
            return envelope({"state": "refused", "code": "ACTION_NOT_FOUND",
                             "error": f"No execution with this action ID exists: {action_id}"})
        return _wait(host, action, timeout_ms)

    @server.custom_route("/health", methods=["GET"])
    async def health(request):  # noqa: ANN001 - Starlette signature
        from starlette.responses import JSONResponse

        host.health_calls += 1
        active = host.admitted if host.admitted and host.admitted not in host.retrieved else None
        return JSONResponse({
            "status": "ok" if active else "idle",
            "foreground": {
                "active": ({"actionId": active, "action": host.actions[active].tool,
                            "progress": {}, "request": None} if active else None),
                "awaitingResult": None,
                "storageError": None,
            },
        })

    return server


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@contextlib.asynccontextmanager
async def run_host(host: FakeHost | None = None):
    """Serve a fake host on a loopback port and yield its URLs."""
    host = host or FakeHost()
    server = build_app(host)
    port = _free_port()
    config = uvicorn.Config(server.streamable_http_app(), host="127.0.0.1", port=port,
                            log_level="warning", lifespan="on")
    uvicorn_server = uvicorn.Server(config)
    task = asyncio.create_task(uvicorn_server.serve())
    try:
        for _ in range(200):
            if uvicorn_server.started:
                break
            await asyncio.sleep(0.02)
        else:  # pragma: no cover - only on a machine that cannot bind
            raise RuntimeError("fake host did not start")
        yield host, f"http://127.0.0.1:{port}/mcp", f"http://127.0.0.1:{port}/health"
    finally:
        uvicorn_server.should_exit = True
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=5)