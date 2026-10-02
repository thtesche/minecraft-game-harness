"""Typed MCP client for the mine-ai-mcp host.

Thin by design. It speaks Streamable HTTP, sends ``response_format: "json"`` so
that replies are parseable rather than prose, and hands back the server's own
shapes. It contains no retry policy and no protocol interpretation - that lives
in :mod:`harness.objective`, where the rules can be tested without a server.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp.shared.exceptions import MCPError

from . import sse
from .config import McpConfig
from .errors import HarnessError, ProtocolError, UnverifiedRead

#: The server's schema caps a rationale at 200 characters.
MAX_RATIONALE_CHARS = 200


@dataclass(frozen=True)
class ToolReply:
    """One tool result, in the server's own vocabulary.

    The server answers in **two different shapes**, and conflating them is the
    failure this type exists to prevent. Foreground tools return the submission
    envelope::

        {"state": "settled", "actionId": ..., "output": {...}}

    while information and control tools return their result directly::

        {"action": "view_status", "durationMs": 1, "result": {...},
         "survival": {...}, "survivalPolicy": {...}}

    ``view_status`` is one of the second kind, so a client that assumes the
    envelope reads nothing at all and reports an empty world as a fact. Neither
    shape is derived from a list of tool names kept here; the server's catalog
    changes independently and a stale copy fails silently.

    ``data`` is the raw reply envelope: ``state`` is one of ``accepted``,
    ``pending``, ``settled``, ``refused`` or ``storage_failed``, plus per-state
    fields such as ``actionId``.
    """

    is_error: bool
    data: dict[str, Any]
    notifications: dict[str, Any]

    @property
    def is_protocol(self) -> bool:
        """Did the submission envelope arrive, rather than a direct payload?"""
        return isinstance(self.data.get("state"), str)

    @property
    def state(self) -> str | None:
        """The protocol state, or ``None`` for a direct call.

        ``None`` means the tool does not speak the submission protocol. Callers
        that require the envelope say so with :meth:`require_state` rather than
        reading this and comparing against a state that cannot arrive.
        """
        state = self.data.get("state")
        return state if isinstance(state, str) else None

    def require_state(self) -> str:
        """The protocol state, insisting on the envelope.

        For the two places the protocol must be in play - submitting a
        foreground objective and waiting on one. A direct payload there means
        the tool or the server changed shape, and continuing would mean
        interpreting an answer that was never about the protocol.
        """
        state = self.state
        if state is None:
            raise ProtocolError(
                f"expected a submission envelope, got a direct reply: {sorted(self.data)}"
            )
        return state

    @property
    def action_id(self) -> str | None:
        action_id = self.data.get("actionId")
        return action_id if isinstance(action_id, str) else None

    @property
    def refusal_code(self) -> str | None:
        if self.state != "refused":
            return None
        code = self.data.get("code")
        return code if isinstance(code, str) else None

    @property
    def output(self) -> dict[str, Any] | None:
        """The settled output inside an envelope. ``None`` for a direct call."""
        output = self.data.get("output")
        return output if isinstance(output, dict) else None

    @property
    def result(self) -> dict[str, Any] | None:
        """The tool's own result object, whichever shape carried it.

        A direct call puts it at ``data.result``; a settled envelope nests it at
        ``data.output.result``. Both spellings exist in the same session, so every
        consumer of "the result" goes through here instead of picking one.
        """
        for container in (self.output, self.data):
            if isinstance(container, dict):
                inner = container.get("result")
                if isinstance(inner, dict):
                    return inner
        return None

    @property
    def result_status(self) -> str | None:
        """The physical outcome inside a settled output, if present."""
        result = self.result
        if result is None:
            return None
        status = result.get("status")
        return status if isinstance(status, str) else None


#: The two spellings an MCP ``tools/list`` entry has used for its input schema.
#: The specification says ``inputSchema``; mine-ai-mcp publishes ``input_schema``.
#: Both are read because reading only one silently disables every check that
#: depends on it - the argument guard accepted ``block_typo`` against the live
#: host for exactly this reason, while a test feeding it ``inputSchema`` stayed
#: green. A check that no longer runs looks identical to a check that passes.
_INPUT_SCHEMA_KEYS = ("input_schema", "inputSchema")


def input_schema_of(tool: dict[str, Any] | None) -> dict[str, Any]:
    """The input schema a tool advertises, under whichever key it uses.

    Returns ``{}`` when there is none, so a caller can ask for ``properties``
    without first proving the schema is there.
    """
    if not isinstance(tool, dict):
        return {}
    for key in _INPUT_SCHEMA_KEYS:
        schema = tool.get(key)
        if isinstance(schema, dict):
            return schema
    return {}


def argument_names(tool: dict[str, Any] | None) -> set[str]:
    """Argument names the advertisement accepts, empty if it says nothing."""
    properties = input_schema_of(tool).get("properties")
    return set(properties) if isinstance(properties, dict) else set()


class McpClient:
    """Connection to one mine-ai-mcp host."""

    def __init__(self, config: McpConfig) -> None:
        self.config = config
        self._session: Any = None
        self._stack: Any = None

    async def __aenter__(self) -> "McpClient":
        from contextlib import AsyncExitStack

        from mcp import ClientSession
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

        # The SDK's default read timeout is 300 s, which a stack smelt exceeds on
        # its own. The transport holds the response open, so the timeout has to be
        # built explicitly and passed down: a read timeout is not cancellation,
        # and the bot keeps working after one fires.
        timeout_s = self.config.tool_timeout_ms / 1000
        self._stack = AsyncExitStack()

        # Before the transport exists, because the ceiling is read when the SDK
        # builds its event parser, which happens inside the first request.
        sse.set_event_size_limit(self.config.max_sse_event_bytes)
        sse.install()

        http_client = await self._stack.enter_async_context(
            create_mcp_http_client(timeout=_timeout(timeout_s))
        )
        read, write = await self._stack.enter_async_context(
            streamable_http_client(self.config.url, http_client=http_client)
        )
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write, read_timeout_seconds=timeout_s)
        )
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
        self._session = None

    async def list_tools(self) -> list[dict[str, Any]]:
        """Tool advertisements.

        Arguments are bound from this, never from prose or recall: a guessed
        argument name produces a plausible answer about the wrong thing and
        never raises its voice.
        """
        if self._session is None:
            raise ProtocolError("client not connected")
        try:
            result = await self._session.list_tools()
        except MCPError as error:
            raise self._explain_lost_stream("tools/list", error) from error
        return [tool.model_dump() for tool in result.tools]

    def _explain_lost_stream(self, what: str, error: MCPError) -> HarnessError:
        """Explain a dropped SSE stream, which the SDK reports as a dead socket.

        httpx2 raises ``SSEError`` when an event exceeds its ceiling, and the SDK
        catches it with a blanket ``except Exception`` and reports the stream as
        having ended without a response. Left alone that message sends the next
        hour of debugging towards the network. Two causes are distinguished:
        the ceiling is too low, or the patch in :mod:`harness.sse` stopped
        applying because the SDK's call site moved.
        """
        if "stream ended without a response" not in str(error):
            return error
        if not sse.require_live_patch():
            return ProtocolError(
                f"{what}: the server's reply never arrived, and the harness's SSE "
                "event-size patch is not in effect - mcp.client.streamable_http no "
                f"longer calls through harness.sse. {sse.describe()}"
            )
        return UnverifiedRead(
            what,
            f"the reply exceeded the {sse.current_limit()}-byte server-sent event "
            f"ceiling and the server dropped it; raise mcp.max_sse_event_bytes "
            f"(the live tools/list is {sse.MEASURED_TOOLS_LIST_BYTES} bytes, "
            f"httpx2's own default is {sse.SDK_DEFAULT_MAX_EVENT_BYTES})",
        )

    async def call(
        self,
        tool: str,
        arguments: dict[str, Any] | None = None,
        *,
        rationale: str,
        read_timeout_ms: int | None = None,
    ) -> ToolReply:
        """Invoke a tool and return the parsed protocol envelope.

        ``response_format: "json"`` is forced: the markdown rendering embeds
        progress without a time field, which is what makes a client poll in
        bundles of three to five.
        """
        if self._session is None:
            raise ProtocolError("client not connected")

        text = rationale.strip()
        if not text:
            raise UnverifiedRead(tool, "every tool call must carry a rationale")
        if len(text) > MAX_RATIONALE_CHARS:
            # The server rejects an over-long rationale as INVALID_ARGUMENTS.
            # Truncating here keeps the call admissible and the intent readable.
            text = text[: MAX_RATIONALE_CHARS - 1] + "…"

        payload = dict(arguments or {})
        payload["response_format"] = "json"
        payload["rationale"] = text

        timeout_s = (read_timeout_ms or self.config.tool_timeout_ms) / 1000
        result = await self._session.call_tool(
            tool, payload, read_timeout_seconds=timeout_s
        )
        return _parse_reply(tool, result)

    async def health(self) -> dict[str, Any]:
        """Read ``/health``.

        Not MCP: a plain GET. After a transport drop this is how we learn
        whether admitted work is still running.
        """
        import httpx2

        async with httpx2.AsyncClient(timeout=10.0) as client:
            response = await client.get(self.config.health())
            response.raise_for_status()
            return response.json()


def _timeout(read_s: float):
    """Transport timeouts.

    Connect, write and pool stay short: a loopback host either accepts now or is
    not there. Only read is stretched, because the server holds the response
    open for as long as the action takes.
    """
    import httpx2

    return httpx2.Timeout(30.0, read=read_s)


def _parse_reply(tool: str, result: Any) -> ToolReply:
    # The SDK exposes `structuredContent` as `structured_content`; the payload
    # inside it is the server's own JSON and keeps its own spelling.
    structured = getattr(result, "structured_content", None)
    if not isinstance(structured, dict):
        raise UnverifiedRead(tool, "reply carried no structured content")

    response = structured.get("response")
    if not isinstance(response, dict):
        raise UnverifiedRead(tool, "structuredContent had no response object")

    if response.get("format") != "json":
        # We asked for json and got markdown, so there is nothing to parse.
        # Treating the prose as data is how a shape change becomes a silent
        # default rather than a loud failure.
        raise UnverifiedRead(tool, f"expected json format, got {response.get('format')!r}")

    data = response.get("data")
    if not isinstance(data, dict):
        raise UnverifiedRead(tool, "response had no data object")

    notifications = structured.get("notifications")
    if not isinstance(data.get("state"), str):
        # Accept the direct-call shape - {action, durationMs, result, survival,
        # survivalPolicy} - but only if it really looks like one. A reply that is
        # neither envelope nor direct is a third shape nobody has modelled, and
        # accepting it silently is how every reader downstream starts reporting
        # a world that does not exist.
        direct = isinstance(data.get("result"), dict) and isinstance(data.get("action"), str)
        if not direct:
            raise UnverifiedRead(
                tool,
                "reply matched neither the submission envelope nor a direct "
                f"result: {sorted(data)}",
            )

    return ToolReply(
        is_error=bool(getattr(result, "is_error", False)),
        data=data,
        notifications=notifications if isinstance(notifications, dict) else {},
    )


def json_size(payload: Any) -> int:
    """Byte size of a payload, for state-vector budget checks."""
    return len(json.dumps(payload, default=str).encode())