"""The SSE event ceiling, and the two reply shapes.

Both exist because a live host refused to be talked to by the code under test:
``tools/list`` arrived as one 2.91 MiB event against httpx2's 1 MiB cap, and
``view_status`` answered in a shape nothing in the harness modelled. The
integration tests in ``test_live_client.py`` prove both against a real
transport; this file pins the reasoning so the fix cannot be quietly undone.
"""

from __future__ import annotations

import httpx2
import mcp.client.streamable_http as sdk
import pytest

from harness import sse
from harness.config import McpConfig
from harness.errors import ProtocolError, UnverifiedRead
from harness.mcp_client import ToolReply


@pytest.fixture(autouse=True)
def _restore_sdk():
    """The patch is process-wide, so every test puts the SDK back."""
    sse.reset_for_tests()
    yield
    sse.reset_for_tests()


def test_the_measurement_beats_the_sdk_default():
    """The configured default is justified against a real number, not a hunch."""
    assert sse.MEASURED_TOOLS_LIST_BYTES > sse.SDK_DEFAULT_MAX_EVENT_BYTES
    assert McpConfig().max_sse_event_bytes > sse.MEASURED_TOOLS_LIST_BYTES


def test_install_replaces_the_sdk_parser():
    original = sdk.EventSource
    sse.set_event_size_limit(4 * 1024 * 1024)
    sse.install()
    assert sdk.EventSource is not original
    assert issubclass(sdk.EventSource, httpx2.EventSource)
    assert sse.is_installed()


def test_install_is_idempotent():
    sse.install()
    first = sdk.EventSource
    sse.install()
    assert sdk.EventSource is first


def test_install_fails_loudly_when_the_sdk_call_site_moves(monkeypatch):
    """A patch that cannot apply must say so, not fail later as a dead socket."""
    monkeypatch.delattr(sdk, "EventSource")
    with pytest.raises(ProtocolError) as error:
        sse.install()
    assert "EventSource" in str(error.value)


def test_a_non_positive_limit_is_rejected():
    with pytest.raises(ValueError):
        sse.set_event_size_limit(0)


def test_the_limit_is_read_per_event_source_not_captured_once():
    """Two clients with different configuration each get their own ceiling.

    `_max_event_size` is httpx2's own attribute, read here because it is the
    effective bound: resolving it once at install time would pin every later
    client to whichever configuration happened to connect first.
    """
    sse.set_event_size_limit(4 * 1024 * 1024)
    sse.install()

    first = sdk.EventSource(_response())
    sse.set_event_size_limit(16 * 1024 * 1024)
    second = sdk.EventSource(_response())

    assert first._max_event_size == 4 * 1024 * 1024
    assert second._max_event_size == 16 * 1024 * 1024


def test_the_installed_parser_keeps_httpx2_default_when_none_is_configured():
    sse.reset_for_tests()
    sse.install()
    assert sdk.EventSource(_response())._max_event_size == sse.current_limit()


def _response() -> object:
    """The smallest thing httpx2.EventSource will accept without streaming."""
    return type("Response", (), {
        "status_code": 200,
        "headers": httpx2.Headers({"content-type": "text/event-stream"}),
        "stream": None,
        "aread": lambda self: b"",
    })()


def test_require_live_patch_distinguishes_never_used_from_too_small():
    """The difference between a limit to raise and a patch that stopped working."""
    sse.reset_for_tests()
    assert not sse.require_live_patch()

    sse.install()
    # Installed, but nothing has gone through it yet - an SDK that no longer
    # calls this class looks exactly like this.
    assert not sse.require_live_patch()

    sdk.EventSource(_response())
    assert sse.require_live_patch()


def test_describe_reports_both_numbers():
    sse.set_event_size_limit(5 * 1024 * 1024)
    sse.install()
    described = sse.describe()
    assert described["limit"] == 5 * 1024 * 1024
    assert described["sdk_default"] == sse.SDK_DEFAULT_MAX_EVENT_BYTES


# --- the two reply shapes ---------------------------------------------------


def direct_reply(**overrides) -> ToolReply:
    """What an information tool answers with: no `state` anywhere."""
    data = {
        "action": "view_status",
        "durationMs": 1,
        "result": {"kind": "read", "status": "succeeded", "situation": {"vitals": {}}},
        "survival": {"summary": "none"},
    }
    data.update(overrides)
    return ToolReply(is_error=False, data=data, notifications={})


def envelope_reply(**overrides) -> ToolReply:
    data = {
        "state": "settled",
        "actionId": "a1",
        "output": {"action": "collect_block", "result": {"status": "succeeded"}},
    }
    data.update(overrides)
    return ToolReply(is_error=False, data=data, notifications={})


def test_a_direct_reply_has_no_state_but_does_not_raise():
    reply = direct_reply()
    assert reply.state is None
    assert not reply.is_protocol
    assert reply.result_status == "succeeded"
    assert reply.action_id is None
    assert reply.refusal_code is None


def test_require_state_refuses_a_direct_reply():
    with pytest.raises(ProtocolError) as error:
        direct_reply().require_state()
    assert "direct reply" in str(error.value)


def test_require_state_returns_the_state_of_an_envelope():
    assert envelope_reply().require_state() == "settled"


def test_result_is_found_in_either_shape():
    assert direct_reply().result["kind"] == "read"
    assert envelope_reply().result["status"] == "succeeded"


def test_a_refusal_has_no_result_object():
    """Refusals must not borrow a result from anywhere, including a neighbour."""
    reply = envelope_reply(state="refused", code="ACTION_BUSY", output=None)
    reply = ToolReply(is_error=True, data={"state": "refused", "code": "ACTION_BUSY",
                                           "activeActionId": "a0"}, notifications={})
    assert reply.result is None
    assert reply.result_status is None
    assert reply.refusal_code == "ACTION_BUSY"


def test_the_state_reader_reads_a_situation_from_either_shape():
    from harness.state import _situation

    assert _situation(direct_reply()) == {"vitals": {}}
    nested = ToolReply(
        is_error=False,
        data={"state": "settled", "output": {"result": {"situation": {"vitals": {"health": 20}}}}},
        notifications={},
    )
    assert _situation(nested) == {"vitals": {"health": 20}}


def test_no_situation_in_either_place_is_none_not_an_empty_world():
    from harness.state import _situation

    assert _situation(direct_reply(result={"status": "succeeded"})) is None
    assert _situation(envelope_reply()) is None


def test_a_third_reply_shape_is_refused_at_parse_time():
    """Neither envelope nor direct means nobody modelled it. Say so."""
    from harness.mcp_client import _parse_reply

    class Result:
        is_error = False
        structured_content = {"response": {"format": "json",
                                           "data": {"unexpected": "shape"}}}

    with pytest.raises(UnverifiedRead) as error:
        _parse_reply("view_status", Result())
    assert "neither" in str(error.value)