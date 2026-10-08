"""The one place the harness reaches into the MCP SDK.

Why this module exists
----------------------

``tools/list`` from the live host is a single server-sent event of 2.91 MiB:
37 tools, of which ``wait_for_action`` is 913 KB because mine-ai-mcp inlines
every ``$ref`` into ``properties`` (its ``definitions`` are 28 KB; the expansion
is 32x that). httpx2 refuses any SSE event above 1 MiB and the SDK constructs its
parser with no way to configure it::

    event_source = EventSource(response)   # mcp/client/streamable_http.py

The ``SSEError`` that httpx2 raises is swallowed by the SDK's own ``except
Exception`` around the event loop, which then reports the far less useful
``SSE stream ended without a response``. So the one number standing between the
harness and every advertised tool is a default nobody can reach.

Why patching rather than working around it
------------------------------------------

The alternative - reading ``tools/list`` out of band over a plain httpx client -
fixes that one call and leaves the transport unable to carry any other oversized
reply. ``wait_for_action`` publishes the union of all 27 foreground output
schemas, so a single settled wait is the same hazard wearing a different hat.
Raising the ceiling fixes the class.

Why the failure has to stay loud
--------------------------------

A silent no-op here is the dangerous outcome: the patch stops matching the SDK's
call site, ``list_tools`` fails again, and the error points at the network rather
than at this module. So :func:`require_live_patch` distinguishes "the limit was
exceeded and needs raising" from "the patch never took effect", and the client
turns the first into a named error instead of the SDK's misleading text.
"""

from __future__ import annotations

import httpx2
from mcp.client import streamable_http as _sdk

from .errors import ProtocolError

#: httpx2's own default, which the SDK therefore inherits. Named so the error
#: message can say what was exceeded rather than just that something was.
SDK_DEFAULT_MAX_EVENT_BYTES = 1_048_576

#: Size of the live ``tools/list`` event, measured 2026-10-02. Kept as a constant
#: because it is the number the configured default is justified against, and a
#: future default chosen below it would be chosen below a known workload.
MEASURED_TOOLS_LIST_BYTES = 3_054_268

_installed = False
_limit = SDK_DEFAULT_MAX_EVENT_BYTES
_event_sources_built = 0


def set_event_size_limit(max_bytes: int) -> None:
    """Set the ceiling used by every server-sent event the harness reads.

    Read at construction time rather than bound once, so a process that opens
    more than one client with different configuration still gets the ceiling its
    own configuration asked for.
    """
    global _limit
    if max_bytes <= 0:
        raise ValueError(f"max_sse_event_bytes must be positive, got {max_bytes}")
    _limit = int(max_bytes)


def current_limit() -> int:
    return _limit


def install() -> None:
    """Put the bounded parser where the SDK will construct it.

    Idempotent. Raises if the SDK no longer looks the way it did when this was
    written, because a patch that cannot apply must say so rather than leave the
    harness failing later with an error that blames something else.
    """
    global _installed, _event_sources_built

    if not hasattr(_sdk, "EventSource"):
        raise ProtocolError(
            "mcp.client.streamable_http has no EventSource attribute; the SDK's "
            "SSE call site has changed and harness.sse no longer applies. "
            "Re-measure tools/list against the live host before raising this."
        )
    if not issubclass(httpx2.EventSource, object):  # pragma: no cover - sanity
        raise ProtocolError("httpx2.EventSource is not a class; httpx2 has changed shape")

    if _installed:
        return

    class _BoundedEventSource(httpx2.EventSource):
        """``httpx2.EventSource`` with a ceiling the harness controls."""

        def __init__(self, response, max_event_size: int | None = None) -> None:
            global _event_sources_built
            _event_sources_built += 1
            super().__init__(
                response,
                max_event_size=_limit,
            )

    _sdk.EventSource = _BoundedEventSource
    _installed = True
    _event_sources_built = 0


def is_installed() -> bool:
    return _installed


def event_sources_built() -> int:
    """How many event sources the patched class has constructed.

    Zero after a connection that opened a stream means the SDK is no longer
    calling through this module, which is the difference between a limit that
    needs raising and a patch that has stopped applying.
    """
    return _event_sources_built


def require_live_patch() -> bool:
    """Confirm the patch is installed and has actually been used."""
    if not _installed:
        return False
    return _event_sources_built > 0


def describe() -> dict[str, int | bool]:
    """State of the patch, for `harness health`."""
    return {
        "installed": _installed,
        "event_sources_built": _event_sources_built,
        "limit": _limit,
        "sdk_default": SDK_DEFAULT_MAX_EVENT_BYTES,
    }


def reset_for_tests() -> None:
    """Restore the SDK's own parser. Test-only; not part of the client API."""
    global _installed, _limit, _event_sources_built
    if _installed:
        _sdk.EventSource = httpx2.EventSource
    _installed = False
    _limit = SDK_DEFAULT_MAX_EVENT_BYTES
    _event_sources_built = 0