"""Command line entry points.

``health``, ``state``, ``run-one`` and ``ledger`` are the Phase 0 surface, kept
because they are how a skeleton is checked against a live world without running a
whole plan. ``run-loop`` is Phase 1: a sequence of objectives, one trusted state
read before each, which is where the harness stops being a smoke test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from .config import Config
from .decide import ScriptedDecider
from .errors import BudgetExceeded, HarnessError, ObjectiveFailed
from .ledger import Ledger
from .loop import INCOMPLETE_STOPS, RunLoop, StepReport
from .mcp_client import McpClient
from .objective import Objective, ObjectiveRunner
from .state import StateReader
from . import sse


def _config(args: argparse.Namespace) -> Config:
    config = Config.load(args.config)
    run_id = config.run_id or f"run-{uuid.uuid4().hex[:8]}"
    return Config(
        mcp=config.mcp,
        ledger=config.ledger,
        budget=config.budget,
        laya=config.laya,
        llm=config.llm,
        run_id=run_id,
    )


async def cmd_health(args: argparse.Namespace) -> int:
    config = _config(args)
    async with McpClient(config.mcp) as client:
        health = await client.health()
        tools = await client.list_tools()
    print(json.dumps({
        "health": health,
        "toolCount": len(tools),
        # Reported because it is the one thing that can stop every reply from
        # arriving at all, and its failure mode names the network rather than
        # itself. `installed: true` with `event_sourcesBuilt: 0` means the patch
        # no longer matches the SDK's call site.
        "sse": sse.describe(),
    }, indent=2))
    return 0


async def cmd_state(args: argparse.Namespace) -> int:
    config = _config(args)
    async with McpClient(config.mcp) as client:
        vector = await StateReader(client).read()
    print(json.dumps(vector.to_dict(), indent=2))
    return 0 if vector.trustworthy else 1


async def cmd_run_one(args: argparse.Namespace) -> int:
    """Submit one objective and report its settled evidence.

    This is the Phase 0 exit criterion: one objective completes unattended and
    one ledger row exists.
    """
    config = _config(args)
    arguments: dict[str, Any] = json.loads(args.arguments) if args.arguments else {}

    with Ledger(config.ledger, config.run_id) as ledger:
        async with McpClient(config.mcp) as client:
            runner = ObjectiveRunner(client, config.mcp, config.budget, ledger)
            objective = Objective(
                tool=args.tool,
                arguments=arguments,
                rationale=args.rationale or f"Phase 0 smoke run: {args.tool}",
            )
            try:
                result = await runner.run(objective)
            except BudgetExceeded as error:
                print(f"budget exhausted: {error}", file=sys.stderr)
                return 2
            except ObjectiveFailed as error:
                print(f"objective failed: {error}", file=sys.stderr)
                return 3

    print(
        json.dumps(
            {
                "tool": result.tool,
                "state": result.state,
                "status": result.status,
                "evidenceOk": result.evidence_ok,
                "polls": result.polls,
                "durationMs": result.duration_ms,
                "error": result.error,
                "ledgerRow": str(Path(config.ledger.path)),
            },
            indent=2,
        )
    )
    return 0 if result.ok else 1


async def cmd_run_loop(args: argparse.Namespace) -> int:
    """Run a sequence of objectives, one trusted state read before each.

    Phase 1's exit criterion: a multi-objective run where every ledger row
    carries the state the decision was made against.

    The objectives come from ``--script``, a JSON list of ``{"tool", "arguments"}``
    entries, which is the :class:`~harness.decide.ScriptedDecider` standing in for
    a model. The seam is the deliverable; the script is the stand-in, and every row
    it writes is marked ``source: "scripted"`` so a scripted run is never mistaken
    for a model run.
    """
    config = _config(args)
    script = json.loads(args.script) if args.script else []
    if not isinstance(script, list) or not all(isinstance(entry, dict) for entry in script):
        print('script must be a JSON list of {"tool": ..., "arguments": {...}}', file=sys.stderr)
        return 64

    entries: list[tuple[str, dict[str, Any]]] = []
    for entry in script:
        tool = entry.get("tool")
        if not isinstance(tool, str) or not tool:
            print(f"script entry has no tool: {entry}", file=sys.stderr)
            return 64
        arguments = entry.get("arguments") or {}
        if not isinstance(arguments, dict):
            print(f"script arguments must be an object: {entry}", file=sys.stderr)
            return 64
        entries.append((tool, arguments))

    with Ledger(config.ledger, config.run_id) as ledger:
        async with McpClient(config.mcp) as client:
            tools = await client.list_tools()
            decider = ScriptedDecider(entries, tools)
            loop = RunLoop(
                StateReader(client),
                ObjectiveRunner(client, config.mcp, config.budget, ledger),
                decider,
                ledger=ledger,
                submission_prefix=config.mcp.submission_prefix or f"{config.run_id}-",
                on_step=lambda step: _print_step(step),
            )
            report = await loop.run(max_steps=args.max_steps)

    print(json.dumps({**report.summary(), "runId": config.run_id}, indent=2))
    if report.stop_reason in INCOMPLETE_STOPS:
        print(f"run stopped: {report.detail or report.stop_reason}", file=sys.stderr)
    return 0 if report.ok else 1


def _print_step(step: StepReport) -> None:
    """One line per step, on stderr.

    stderr so the stdout report stays machine-readable. A real run takes minutes
    per objective, and a loop that reports only at exit is indistinguishable from
    a hung process.
    """
    if step.result is None:
        print(f"[{step.index}] {step.tool}: {step.error}", file=sys.stderr)
        return
    result = step.result
    death = f" DIED: {step.death_cause}" if step.death_cause else ""
    print(
        f"[{step.index}] {step.tool}: {result.state}:{result.status} "
        f"evidence={'ok' if result.evidence_ok else 'MISSING'} "
        f"{result.duration_ms / 1000:.1f}s{death}",
        file=sys.stderr,
    )


async def cmd_ledger(args: argparse.Namespace) -> int:
    config = _config(args)
    with Ledger(config.ledger, config.run_id) as ledger:
        rows = ledger.read_all()
    if args.counts:
        print(json.dumps(ledger_counts(rows), indent=2))
    else:
        print(json.dumps(rows[-args.limit :], indent=2))
    return 0


def ledger_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        key = row.get("outcome") or "no_outcome"
        totals[key] = totals.get(key, 0) + 1
    return totals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Laya-gated decision harness for Mine AI MCP",
    )
    parser.add_argument("--config", help="path to a JSON config file")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="read /health and count published tools").set_defaults(
        handler=cmd_health
    )
    sub.add_parser("state", help="print the derived decision state vector").set_defaults(
        handler=cmd_state
    )

    run_one = sub.add_parser("run-one", help="submit one objective end to end")
    run_one.add_argument("tool", help="an MCP tool name, e.g. view_status or collect_block")
    run_one.add_argument("--arguments", help="tool arguments as JSON")
    run_one.add_argument("--rationale", help="why this objective, in the model's own words")
    run_one.set_defaults(handler=cmd_run_one)

    run_loop = sub.add_parser(
        "run-loop",
        help="run a sequence of objectives, one state read before each",
    )
    run_loop.add_argument(
        "--script",
        help='JSON list of {"tool": ..., "arguments": {...}}, in order',
    )
    run_loop.add_argument(
        "--max-steps",
        type=int,
        default=32,
        help="ceiling on loop passes, independent of the script length",
    )
    run_loop.set_defaults(handler=cmd_run_loop)

    ledger = sub.add_parser("ledger", help="inspect recorded decisions")
    ledger.add_argument("--limit", type=int, default=20)
    ledger.add_argument("--counts", action="store_true", help="summarise by outcome")
    ledger.set_defaults(handler=cmd_ledger)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(args.handler(args))
    except HarnessError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())