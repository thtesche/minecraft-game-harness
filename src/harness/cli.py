"""Command line entry points.

``health``, ``state``, ``run-one`` and ``ledger`` are the Phase 0 surface, kept
because they are how a skeleton is checked against a live world without running a
whole plan. ``run-loop`` is Phase 1: a sequence of objectives, one trusted state
read before each, which is where the harness stops being a smoke test. With
``--decider llm`` it is Phase 2: the model chooses each objective.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from .config import Config
from .decide import ScriptedDecider
from .errors import BudgetExceeded, HarnessError, ObjectiveFailed
from .ledger import Ledger
from .llm import LlmError, LlmUsage, OpenRouterDecider, load_dotenv
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

    The objectives come from one of two deciders:

    ``--decider script`` replays ``--script``, a JSON list of
    ``{"tool", "arguments"}`` entries, through
    :class:`~harness.decide.ScriptedDecider`. Every row it writes is marked
    ``source: "scripted"`` so a scripted run is never mistaken for a model run.

    ``--decider llm`` asks the model named in ``llm.model`` on every decision,
    through :class:`~harness.llm.OpenRouterDecider`. That is the Phase 2
    baseline, and it is the run the later, cheaper deciders have to beat.
    """
    config = _config(args)
    if args.decider == "llm":
        return await _run_loop_with_model(config, args)
    if not args.script:
        print("run-loop needs --script; there is no model to choose objectives yet",
              file=sys.stderr)
        return 64
    try:
        script = json.loads(args.script)
    except json.JSONDecodeError as error:
        # Named rather than raised: a quote left out of a shell command is a typo,
        # and a traceback sends the reader looking through the parser.
        print(f"--script is not valid JSON: {error}", file=sys.stderr)
        return 64
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


async def _run_loop_with_model(config: Config, args: argparse.Namespace) -> int:
    """`run-loop --decider llm`: a model chooses every objective.

    The key is read from ``.env`` and then from the environment, with the
    environment winning, and is never written anywhere. Everything that can go
    wrong before the first decision - no key, no model named - is reported by
    name here, because the alternative is an exception from inside a context
    manager with a live ledger and a live Minecraft session attached.
    """
    load_dotenv()
    if not config.llm.model:
        print(
            "llm.model is empty; set it in the config to an OpenRouter model id. "
            "An empty model would be sent as a request with no model in it.",
            file=sys.stderr,
        )
        return 78
    if not os.environ.get(config.llm.api_key_env):
        print(
            f"{config.llm.api_key_env} is not set. Copy .env.example to .env, put "
            f"the key in it, and export it; the key is read from the environment.",
            file=sys.stderr,
        )
        return 78

    with Ledger(config.ledger, config.run_id) as ledger:
        async with McpClient(config.mcp) as client:
            tools = await client.list_tools()
            try:
                decider = OpenRouterDecider(config.llm, tools)
            except LlmError as error:
                print(str(error), file=sys.stderr)
                return 78
            if decider.unreachable or decider.stale_read_controls:
                # Both are facts about this harness, not the server, and both are
                # otherwise invisible. An unreachable tool looks exactly like a
                # tool the server does not have.
                if decider.unreachable:
                    print(
                        f"note: {len(decider.unreachable)} advertised tool(s) advertise no "
                        f"arguments and were kept from the model: {decider.unreachable}",
                        file=sys.stderr,
                    )
                if decider.stale_read_controls:
                    print(
                        f"note: these read/control tools are excluded from decisions but "
                        f"the server no longer advertises them, so the exclusion is dead "
                        f"weight: {decider.stale_read_controls}",
                        file=sys.stderr,
                    )
            loop = RunLoop(
                StateReader(client),
                ObjectiveRunner(client, config.mcp, config.budget, ledger),
                decider,
                ledger=ledger,
                submission_prefix=config.mcp.submission_prefix or f"{config.run_id}-",
                on_step=lambda step: _print_step(step),
            )
            report = await loop.run(max_steps=args.max_steps)

    summary = report.summary()
    usage = _usage_summary(decider.calls)
    if usage:
        summary["modelUsage"] = usage
    print(json.dumps({**summary, "runId": config.run_id}, indent=2))
    if report.stop_reason in INCOMPLETE_STOPS:
        print(f"run stopped: {report.detail or report.stop_reason}", file=sys.stderr)
    return 0 if report.ok else 1


def _usage_summary(calls: list[LlmUsage]) -> dict[str, Any] | None:
    """Aggregate the run's model cost.

    This is the Phase 2 baseline measurement. Printing it at the end of the run
    is what turns "we used a model" into a number a later decider can be measured
    against.
    """
    if not calls:
        return None
    cost = sum(call.cost_usd or 0.0 for call in calls)
    retried = sum(1 for call in calls if call.attempts > 1)
    return {
        "calls": len(calls),
        "models": sorted({call.model for call in calls if call.model}),
        "promptTokens": sum(call.prompt_tokens for call in calls),
        "completionTokens": sum(call.completion_tokens for call in calls),
        "reasoningTokens": sum(call.reasoning_tokens for call in calls),
        "retriedCalls": retried,
        "costUsd": round(cost, 6),
        "costReported": any(call.cost_usd is not None for call in calls),
    }


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
        "--decider",
        choices=("script", "llm"),
        default="script",
        help=(
            "script replays --script and records source=scripted; llm asks "
            "llm.model on every decision and records source=openrouter:<model>"
        ),
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