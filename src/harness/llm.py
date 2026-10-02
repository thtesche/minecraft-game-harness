"""A frontier model on every decision: the Phase 2 baseline.

The Phase 1 loop is deterministic and every judgement call is hidden behind
:class:`~harness.decide.Decider`. This module fills that seam with a real
decision-maker, so the same loop runs with a model choosing each objective and
the ledger records what it cost. That number - model calls per objective, and
what each call spent - is the baseline Phase 4 has to beat.

Three things this deliberately does not do
-----------------------------------------

**It does not apply :func:`~harness.decide.confidence_gate`.** That gate is a
Laya policy: Laya is a classifier with fitted temperatures whose confidence can
be measured against a real accuracy curve (D12). A frontier model asked to emit
a JSON objective has no such number; whatever it reports about its own
certainty is a fluent sentence, not a calibrated probability. Gating on it would
either stop the run on every decision or, once ``min_confidence`` is fitted in
Phase 3, pass everything and report that calibration had happened when nothing
was measured. So this decider reports ``confidence=None``, which means "not
calibrated", and the ledger says so rather than claiming certainty it does not
have. A hard-coded ``1.0`` would unblock the same code path while writing a
false statement into the record of every run.

**It does not trust the model's arguments.** The model is asked for a tool and
an object; both are checked against the ``tools/list`` advertisement before the
objective is submitted. A model invents argument names at least as readily as
a script does, and the server drops an unknown argument instead of refusing it,
so an unchecked model call produces an objective that ran against the wrong
thing and reported success.

**It does not put the API key anywhere but the request header.** Not in the
ledger, not in a log line, not in the error messages below.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

import httpx

from .config import LlmConfig
from .decide import Completion, Proposal
from .errors import HarnessError
from .goals import GoalBoard
from .mcp_client import argument_names, input_schema_of
from .state import StateVector

#: Tools that read or steer the harness rather than submit work.
#:
#: Declared here rather than inferred. An advertisement carries no flag saying
#: "this one is an objective", and the tier enums the server does publish are not
#: a ranking of these (D13), so inferring from names would be a guess dressed as
#: a fact. Verified against the live host: all ten exist there, and everything
#: else - 27 tools - is offered to the model. Default is open, so a tool the
#: server adds later is usable without a change here.
CONTROL_AND_READ_TOOLS = frozenset({
    # Information: a read. The loop reads state before every decision, so
    # offering these as objectives invites the model to spend an objective slot
    # re-deriving what it was just told.
    "view_status",
    "view_blocks",
    "view_crafting_requirements",
    "view_frontier",
    "read_recent_events",
    "query_bot_data",
    "note_read",
    # Control: the harness's own protocol surface. A model choosing
    # `wait_for_action` would be racing the runner for the same action.
    "wait_for_action",
    "cancel_foreground_action",
    "set_survival_policy",
})

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

#: Failures worth trying again. A free model sits behind a provider that is
#: overloaded a good fraction of the time - measured, not assumed: four probes
#: returned 503 three times, interleaved with successes - so without this a run
#: ends on a condition that would have passed on the next attempt. A 401 or a
#: 400 is not here, because the second attempt of a wrong key fails identically
#: and the wait only delays the report of a real problem.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
RETRYABLE_CODES = frozenset({"provider_overloaded", "rate_limit_exceeded",
                             "timeout", "server_error"})

#: The JSON the model is asked for. Deliberately not ``strict``: the provider's
#: strict mode is the one part of this path the harness does not control, and
#: the checks that actually matter - is the tool advertised, do the arguments
#: match its schema - are performed here regardless of what the provider
#: guarantees. ``tool`` is an enum of real tool names, so a hallucinated tool is
#: unrepresentable rather than caught afterwards.
_DECISION_SCHEMA: dict[str, Any] = {
    "name": "objective_decision",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "done": {
                "type": "boolean",
                "description": "True when there is nothing further worth doing.",
            },
            "tool": {
                "type": "string",
                "description": "The single tool to run next. Ignored when done is true.",
            },
            "arguments": {
                "type": "object",
                "description": "Arguments for that tool, exactly as its input schema names them.",
            },
            "rationale": {
                "type": "string",
                "description": "One sentence: why this objective, from this state.",
            },
        },
        "required": ["done", "tool", "arguments", "rationale"],
    },
}

_SYSTEM_PROMPT = """\
You are the decision step of an autonomous Minecraft bot. You do not play the \
game; you choose the next single objective, and a separate deterministic loop \
submits it, waits for it to finish, and reads the world again.

You are given the bot's current state as JSON and a catalogue of tools with \
their exact input schemas. Choose exactly one tool.

Rules:
- Use only argument names and values the tool's schema allows. A misspelled \
argument is dropped silently by the server, so the objective runs against the \
wrong thing and reports success anyway.
- Set "done": true when there is nothing further worth doing. Do not set it \
merely to end the run early.
- One objective at a time. You will be called again after it completes, with \
the state as it then is.
- Prefer a concrete objective you can act on over a plan. If the next step is \
not one of these tools, choose the nearest tool that makes progress toward it.
- Name the tool's arguments exactly as the schema spells them, and use the \
generic block and item names the tools accept ("logs", "stone") rather than a \
species name the tool may not recognise.

If a "goals" list is present, every objective must name the one goal it \
serves in "goal", and the run is measured by calls per goal - so an objective \
that serves no listed goal is work the run will refuse. Each goal carries how \
many attempts it has already had; a goal marked exhausted is one the harness \
will refuse, so work on another. A goal is a target item, not a procedure: \
`craft_item` resolves the recipe tree itself and tells you what leaves are \
missing, so name the item you want rather than the steps to reach it.

Answer in exactly this shape, with the reason in "rationale" and only the \
tool's own arguments in "arguments":

    {"goal": "wooden_pickaxe", "done": false, "tool": "collect_block", "arguments": {"block_name": "logs"}, "rationale": "no logs held, and logs are the first step to a pickaxe"}

A wrong answer here is worse than a slow one: an argument name the tool does \
not advertise is dropped by the server, so the objective runs against the \
wrong thing and reports success anyway.
"""


class LlmError(HarnessError):
    """The model could not be asked, or would not answer in a usable shape.

    Raised rather than escalated. An escalation means "this decider declines to
    answer", so turning a 401 or a rate limit into one would report a broken key
    as a considered refusal to decide - and, per D12, would stop the run with a
    reason pointing at the wrong thing.
    """


class _NotJson(Exception):
    """The model's answer was not JSON, or not an object."""


class _Transient(LlmError):
    """A provider failure that trying again could clear.

    Internal. Anything the operator needs to read becomes a plain
    :class:`LlmError` once the retries are exhausted, carrying the attempt count
    so a run that needed four attempts is visible as such.
    """


def _load_json_object(text: str) -> Any:
    """Parse the model's answer, tolerating one level of string wrapping.

    A model asked for a JSON object sometimes returns the object *as a JSON
    string* - ``"{\\"tool\\": ...}"`` - which parses cleanly and then fails every
    field lookup. Observed from this model on a live call, so it is handled once
    and deliberately: a second layer would be a model writing JSON inside JSON
    inside a string, and that is a bug to report rather than to keep unwrapping.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise _NotJson(str(error)) from error
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as error:
            raise _NotJson(f"answer was a JSON string containing {error}") from error
    return parsed


@dataclass(frozen=True)
class LlmUsage:
    """What one model call cost.

    Recorded whether or not anyone reads it. The Phase 2 baseline is measured in
    calls per objective, and a run whose cost was not recorded cannot be compared
    with the Phase 4 run meant to be cheaper.
    """

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: int | None = None
    #: How many HTTP attempts this decision took. Over 1 means the provider
    #: failed and was retried, which is worth knowing when a run is slow.
    attempts: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
        }


@dataclass(frozen=True)
class LlmDecision:
    """One parsed model answer, before it becomes a :class:`Proposal`."""

    tool: str | None
    arguments: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    done: bool = False
    goal: str | None = None
    usage: LlmUsage = field(default_factory=LlmUsage)


def load_dotenv(path: Path | str | None = None) -> dict[str, str]:
    """Read ``.env`` into the environment, without overwriting what is there.

    An existing environment variable wins. A key exported in a shell profile is
    a deliberate setting; a stale line in a checked-out ``.env`` silently
    replacing it is the kind of thing that produces a run against the wrong
    account with no explanation.

    Returns the values read, so a caller can report which file was used. The
    key is never logged.
    """
    env_path = Path(path) if path else Path(".env")
    if not env_path.exists():
        return {}
    loaded: dict[str, str] = {}
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        name, sep, value = line.partition("=")
        name = name.strip()
        if not sep or not name:
            continue
        value = value.strip()
        # Strip one matched pair of quotes; the rest is taken literally, which
        # is what a key is.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        loaded[name] = value
        os.environ.setdefault(name, value)
    return loaded


def objective_tools(tools: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The tools a decision may choose from, keyed by name.

    Excludes the reads and controls named above, and anything that advertises no
    arguments - there would be nothing to check a proposed argument object
    against, so offering it would mean accepting whatever the model invented.
    """
    chosen: dict[str, dict[str, Any]] = {}
    for tool in tools:
        name = str(tool.get("name") or "")
        if not name or name in CONTROL_AND_READ_TOOLS:
            continue
        if not argument_names(tool):
            continue
        chosen[name] = tool
    return chosen


def _catalogue(tools: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Name, purpose and accepted arguments for each tool, and nothing else.

    The full advertisement is 2.4 MB, almost all of it output schemas describing
    results the model never reads. The input side is 28,799 bytes across all 37
    tools, so the whole catalogue fits in a prompt comfortably and no filtering
    heuristic is needed to get there.
    """
    rows: list[dict[str, Any]] = []
    for name, tool in sorted(tools.items()):
        schema = input_schema_of(tool)
        properties = schema.get("properties", {})
        required = set(schema.get("required") or ())
        rows.append({
            "name": name,
            "description": tool.get("description") or "",
            "arguments": {
                arg: {
                    "type": spec.get("type", "any"),
                    "description": spec.get("description", ""),
                    "required": arg in required,
                    **({"enum": spec["enum"]} if isinstance(spec.get("enum"), list) else {}),
                    **({"minimum": spec["minimum"]} if "minimum" in spec else {}),
                    **({"maximum": spec["maximum"]} if "maximum" in spec else {}),
                }
                for arg, spec in sorted(properties.items())
                if arg not in _HARNESS_OWNED
            },
        })
    return rows


#: Arguments the harness fills in and the model must not choose. Naming them
#: here rather than letting the model fill them keeps a submission id out of a
#: model's reach, which is what makes D15's conflict unrecoverable rather than
#: merely likely.
_HARNESS_OWNED = frozenset({"submission_id"})


class OpenRouterDecider:
    """Chooses the next objective with a frontier model, every time.

    One HTTP round trip per decision, no retry inside the decider: a retry that
    hides a failing key looks like a slow model, and the run stops later for a
    reason that names neither. The loop's own bounds are the retry policy.
    """

    source = "openrouter"

    def __init__(
        self,
        config: LlmConfig,
        tools: Sequence[dict[str, Any]],
        *,
        transport: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
        environ: dict[str, str] | None = None,
        max_tokens: int = 4096,
        max_attempts: int = 4,
        backoff_s: float = 2.0,
    ) -> None:
        if not config.model:
            raise LlmError(
                "llm.model is empty; set it to an OpenRouter model id, for example "
                "the one in config.example.json. An empty model would otherwise be "
                "sent as a request with no model in it."
            )
        self._config = config
        self._tools = objective_tools(tools)
        if not self._tools:
            raise LlmError(
                "no tool advertises an input schema, so there is nothing a decision "
                "could choose; check that tools/list returned the real advertisements"
            )
        names = {str(tool.get("name") or "") for tool in tools}
        # Kept because both are facts about this file rather than about the
        # server, and both are otherwise invisible: a tool the model cannot reach
        # looks the same as a tool the server does not have.
        self._unreachable = sorted(names - set(self._tools) - CONTROL_AND_READ_TOOLS)
        self._stale_read_controls = sorted(CONTROL_AND_READ_TOOLS - names)
        self._catalogue = _catalogue(self._tools)
        self._environ = environ if environ is not None else os.environ
        self._transport = transport or self._http_post
        self._max_tokens = max_tokens
        self._max_attempts = max_attempts
        #: Zero in tests, so a retry test costs milliseconds instead of seconds.
        #: The policy is the same either way; only the wait is configured.
        self._backoff_s = backoff_s
        #: Every call made, so a run can be measured after the fact.
        self.calls: list[LlmUsage] = []
        #: Set by the loop. ``None`` for a run with no goals, in which case the
        #: prompt omits them and the answer is not asked for a goal.
        self.goals: GoalBoard | None = None

    @property
    def choices(self) -> list[str]:
        """Tool names this decider can choose."""
        return sorted(self._tools)

    @property
    def unreachable(self) -> list[str]:
        """Advertised tools the model may not choose, and why.

        A tool here advertises no arguments, so there is nothing to check a
        proposed argument object against. Every one of the 37 tools on the live
        host advertises arguments and so never appears here; if one does, it is
        either a genuinely argument-less objective the model cannot reach, or an
        advertisement the harness is reading wrongly. Worth saying out loud
        either way.
        """
        return list(self._unreachable)

    @property
    def stale_read_controls(self) -> list[str]:
        """Read/control tools this harness excludes that the server no longer has.

        Pure staleness: a renamed or removed tool left in the set above is dead
        weight that would hide a real read from the model if the name came back.
        """
        return list(self._stale_read_controls)

    async def propose(self, vector: StateVector, *, step: int) -> Proposal | None:
        goals = self.goals
        enabled = goals is not None and goals.enabled
        schema = dict(_DECISION_SCHEMA["schema"])
        if enabled:
            # An enum, so a hallucinated goal is unrepresentable rather than
            # caught afterwards - the same reasoning as the tool enum below. The
            # prompt carries `goals.view()` from the same board, so the enum and
            # the prompt cannot describe different sets.
            schema["properties"] = {
                **schema["properties"],
                "goal": {
                    "type": "string",
                    "enum": list(goals.items),
                    "description": "The goal this objective serves.",
                },
            }
            schema["required"] = [*schema["required"], "goal"]
        payload = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps({
                        "step": step,
                        **({"goals": goals.view()} if enabled else {}),
                        "state": vector.to_dict(),
                        "tools": self._catalogue,
                    }, indent=2),
                },
            ],
            "temperature": self._config.temperature,
            "max_tokens": self._max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": _DECISION_SCHEMA["name"],
                    # Enforced, not advisory. Measured: without `strict` this model
                    # returned chain-of-thought prose in `content` and the
                    # constraint was silently ignored.
                    "strict": True,
                    "schema": {
                        **schema,
                        "properties": {
                            **schema["properties"],
                            "tool": {
                                "type": "string",
                                "enum": sorted(self._tools),
                                "description": "The single tool to run next. Ignored when done is true.",
                            },
                        },
                    },
                },
            },
        }
        response = await self._ask(payload)
        decision = self._parse(response)
        self.calls.append(decision.usage)
        if decision.done:
            # A Completion rather than None. None means "this decider has nothing
            # left to give" - true of a script that has run out, and a fact about
            # the harness. Here it is the model's assertion that the work is
            # finished, which is an answer about the goals and the most consequential
            # thing the model says all run. Returning None made it indistinguishable
            # from a spent script and left no ledger row, so a run that ended
            # because the model believed it was finished - while holds_item said the
            # bot held nothing - looked identical to a plan that ran out correctly.
            return Completion(reason=decision.rationale, goal=decision.goal)
        return Proposal(
            tool=decision.tool or "",
            arguments=dict(decision.arguments),
            rationale=decision.rationale,
            # Not calibrated, and never will be from a self-report. See the module
            # docstring: this is the deliberate difference from a Laya decider.
            confidence=None,
            source=f"{self.source}:{decision.usage.model or self._config.model}",
            goal=decision.goal,
        )

    def _parse(self, response: dict[str, Any]) -> LlmDecision:
        """Turn a provider response into a decision, or refuse it by name."""
        if "error" in response:
            error = response["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            code = error.get("code") if isinstance(error, dict) else None
            raise LlmError(f"OpenRouter returned an error ({code}): {message}")
        try:
            choice = response["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise LlmError(
                f"OpenRouter response has no choices[0].message; keys were "
                f"{sorted(response)}"
            ) from error
        # Truncation is its own failure and would otherwise arrive as a JSON
        # parse error about a brace. Observed live: this model spends ~300
        # tokens on reasoning, then the answer, and a 1024-token ceiling cut it
        # off mid-object with finish_reason "length".
        if choice.get("finish_reason") == "length":
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            raise LlmError(
                f"model hit the {self._max_tokens}-token ceiling and was cut off "
                f"(finish_reason=length, completion_tokens="
                f"{usage.get('completion_tokens')}); raise llm max_tokens or ask for "
                f"less reasoning. A truncated answer cannot be parsed and must not "
                f"be guessed at."
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            reasoning = message.get("reasoning")
            raise LlmError(
                f"model returned empty content (message keys {sorted(message)})"
                + (f"; it reasoned instead: {reasoning[:200]!r}" if reasoning else "")
            )
        try:
            body = _load_json_object(content)
        except _NotJson as error:
            raise LlmError(
                f"model did not return a JSON object: {error}; it said {content[:200]!r}"
            ) from error
        if not isinstance(body, dict):
            raise LlmError(f"model returned {type(body).__name__}, not an object")

        tool = body.get("tool")
        arguments = body.get("arguments")
        if not isinstance(arguments, dict):
            raise LlmError(f"model returned arguments as {type(arguments).__name__}, not an object")
        if body.get("done") is True:
            # The goal is read here too. A completion says *which* goal the model
            # believes is met, and that is the entire content of the claim - a
            # `done` with no goal says only "stop", which is indistinguishable from
            # running out of steam, and those are different findings. `_checked_goal`
            # still applies, because a completion claiming an off-set goal is the
            # model answering a different question than the one asked.
            return LlmDecision(tool=None, rationale=str(body.get("rationale") or ""),
                               done=True, goal=self._checked_goal(body.get("goal")),
                               usage=self._usage(response))

        if not isinstance(tool, str) or tool not in self._tools:
            raise LlmError(
                f"model chose {tool!r}, which is not an advertised objective tool; "
                f"it may choose from {sorted(self._tools)}"
            )
        goal = self._checked_goal(body.get("goal"))
        self._check_arguments(tool, arguments)
        return LlmDecision(tool=tool, arguments=arguments,
                           rationale=str(body.get("rationale") or ""),
                           goal=goal, usage=self._usage(response))

    def _checked_goal(self, value: Any) -> str | None:
        """The goal this objective serves, or ``None`` for a run without goals.

        Refused here rather than left to the loop, because the loop's refusal
        stops the whole run: a model that names a goal nobody asked for is
        answering a different question, and the cheapest place to notice is where
        the answer arrives.
        """
        goals = self.goals
        if goals is None or not goals.enabled:
            return None
        if not isinstance(value, str) or not value:
            raise LlmError(
                f"this run has goals {list(goals.items)} and every decision must name "
                "one in `goal`; the run is measured by calls per goal, so an unnamed "
                "objective cannot be counted. It came back as "
                f"{type(value).__name__} {value!r}"
            )
        if not goals.known(value):
            raise LlmError(
                f"model named goal {value!r}, which is not in this run's goal set "
                f"{list(goals.items)}"
            )
        return value

    def _check_arguments(self, tool: str, arguments: dict[str, Any]) -> None:
        """Refuse arguments the advertised schema does not accept.

        Same reason as :class:`~harness.decide.ScriptedDecider`, and the model
        needs it more: a model fills an object from a description and invents
        plausible names, and the server drops what it does not recognise rather
        than refusing, so the objective would run against the wrong thing and
        report success.
        """
        accepted = argument_names(self._tools[tool]) - _HARNESS_OWNED
        unknown = sorted(set(arguments) - accepted)
        if unknown:
            raise LlmError(
                f"{tool} accepts {sorted(accepted)}, not {unknown}; a wrong argument "
                "name is dropped by the server rather than refused, so the objective "
                "would run against the wrong thing and say nothing"
            )
        required = set(input_schema_of(self._tools[tool]).get("required") or ()) - _HARNESS_OWNED
        missing = sorted(required - set(arguments))
        if missing:
            raise LlmError(f"{tool} requires {sorted(required)}; the model omitted {missing}")

    def _usage(self, response: dict[str, Any]) -> LlmUsage:
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        details = usage.get("completion_tokens_details")
        cost = usage.get("cost")
        return LlmUsage(
            model=str(response.get("model") or self._config.model),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            reasoning_tokens=int(details.get("reasoning_tokens") or 0)
            if isinstance(details, dict) else 0,
            cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
            latency_ms=response.get("_latency_ms"),
            attempts=int(response.get("_attempts") or 1),
        )

    def _http_post(self, payload: dict[str, Any]) -> Awaitable[dict[str, Any]]:
        key = self._environ.get(self._config.api_key_env, "")
        if not key:
            raise LlmError(
                f"{self._config.api_key_env} is not set. Put it in .env (see "
                f".env.example) and export it; the key is read from the environment, "
                f"never from the config file or the ledger."
            )
        base = (self._config.base_url or DEFAULT_BASE_URL).rstrip("/")
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # OpenRouter attributes usage by these; they carry no identity.
            "HTTP-Referer": "https://github.com/thtesche/minecraft-game-harness",
            "X-Title": "minecraft-game-harness",
        }
        client = httpx.AsyncClient(timeout=httpx.Timeout(180.0))
        return self._post(client, base, headers, payload)

    async def _post(
        self,
        client: httpx.AsyncClient,
        base: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """One HTTP attempt. Retrying is :meth:`_ask`'s business, not this one's."""
        try:
            response = await client.post(f"{base}/chat/completions", json=payload, headers=headers)
        except httpx.HTTPError as error:
            raise _Transient(f"could not reach OpenRouter at {base}: {error}") from error
        finally:
            await client.aclose()
        try:
            body = response.json()
        except ValueError as error:
            detail = (f"OpenRouter returned HTTP {response.status_code} with a non-JSON "
                      f"body: {response.text[:200]}")
            raise (_Transient(detail) if response.status_code in RETRYABLE_STATUS
                   else LlmError(detail)) from error
        if isinstance(body, dict):
            return body
        raise LlmError(f"OpenRouter returned {type(body).__name__}, not an object")

    async def _ask(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ask the provider, retrying only what is worth retrying.

        Lives here rather than inside the HTTP call so the policy is the
        decider's, and so a test's stub transport is subject to exactly the same
        retry behaviour as a real one. Attempts are counted onto the response
        instead of being swallowed, so a run that took four attempts to get one
        decision does not look like a slow model.
        """
        started = time.monotonic()
        attempts = 0
        last = "the provider was not asked"
        for attempts in range(1, self._max_attempts + 1):
            try:
                response = await self._transport(payload)
            except _Transient as error:
                last = str(error)
            else:
                detail = _response_failure(response)
                if detail is None:
                    return {
                        **response,
                        "_attempts": attempts,
                        "_latency_ms": int((time.monotonic() - started) * 1000),
                    }
                last, retryable = detail
                if not retryable:
                    raise LlmError(last)
            if attempts < self._max_attempts:
                # Short and growing. The provider recovers in seconds, and a run
                # that waits a minute per decision is its own kind of failure.
                await asyncio.sleep(min(self._backoff_s * attempts, 8.0))
        raise LlmError(f"{last} (after {attempts} attempt{'s' if attempts != 1 else ''})")


def _response_failure(response: dict[str, Any]) -> tuple[str, bool] | None:
    """``None`` if the response carries a usable answer, else a message and
    whether retrying could clear it.

    The provider's own message and code are carried through, because they are
    what names the problem; the harness adds only the retry decision.
    """
    if not isinstance(response, dict):
        return (f"provider transport returned {type(response).__name__}, not an object", False)
    error = response.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
    # OpenRouter reports a provider overload as `code: 503` with
    # `metadata.error_type: "provider_overloaded"`. Both spellings are checked
    # because a free model is overloaded far more often than it is rate limited,
    # and that is the case worth retrying.
    retryable = (
        code in RETRYABLE_CODES
        or metadata.get("error_type") in RETRYABLE_CODES
        or (isinstance(code, int) and code in RETRYABLE_STATUS)
    )
    message = error.get("message") or "(no message)"
    return (f"OpenRouter returned an error ({code}): {message}", retryable)