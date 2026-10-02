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

from .config import McpConfig
from .errors import ProtocolError, UnverifiedRead

#: The server's schema caps a rationale at 200 characters.
MAX_RATIONALE_CHARS = 200


@dataclass(frozen=True)
class ToolReply:
    """One tool result, in the server's own vocabulary.

    ``data`` is the protocol envelope from ``structuredContent.response.data``:
    ``state`` is one of ``accepted``, ``pending``, ``settled``, ``refused`` or
    ``storage_failed``, plus per-state fields such as ``actionId``.
    """

    is_error: bool
    data: dict[str, Any]
    notifications: dict[str, Any]

    @property
    def state(self) -> str:
        state = self.data.get("state")
        if not isinstance(state, str):
            raise ProtocolError(f"reply has no string state: {sorted(self.data)}")
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
        output = self.data.get("output")
        return output if isinstance(output, dict) else None

    @property
    def result_status(self) -> str | None:
        """The physical outcome inside a settled output, if present."""
        output = self.output
        if output is None:
            return None
        result = output.get("result")
        if not isinstance(result, dict):
            return None
        status = result.get("status")
        return status if isinstance(status, str) else None


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
        result = await self._session.list_tools()
        return [tool.model_dump() for tool in result.tools]

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
    return ToolReply(
        is_error=bool(getattr(result, "is_error", False)),
        data=data,
        notifications=notifications if isinstance(notifications, dict) else {},
    )


def json_size(payload: Any) -> int:
    """Byte size of a payload, for state-vector budget checks."""
    return len(json.dumps(payload, default=str).encode())