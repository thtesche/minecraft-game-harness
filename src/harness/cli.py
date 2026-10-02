"""Command line entry points.

Phase 0 surface: connect, read the state, run one objective, write a ledger row.
The decision loop arrives in Phase 4 and replaces ``run-one``; until then this
is how a skeleton is checked end to end against a live world.
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
from .errors import BudgetExceeded, HarnessError, ObjectiveFailed
from .ledger import Ledger
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