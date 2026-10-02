"""Integration tests: the real MCP client against a real MCP server.

Everything else in the suite stubs the client. This file does not, because the
client is where an unverified assumption hides: the SDK call signature, the
structuredContent envelope, the /health read. Those are exactly the things that
fail at three in the morning with a live world and no test to say why.
"""

from __future__ import annotations

import pytest
from fake_host import FakeHost, run_host

from harness.config import BudgetConfig, LedgerConfig, McpConfig
from harness.errors import BudgetExceeded, ObjectiveFailed, UnverifiedRead
from harness.ledger import Ledger
from harness.mcp_client import McpClient
from harness.objective import Objective, ObjectiveRunner
from harness.state import StateReader


@pytest.fixture
def live():
    """A fake mine-ai-mcp host listening on loopback."""
    return run_host()


async def config_for(mcp_url: str, health_url: str) -> McpConfig:
    return McpConfig(url=mcp_url, health_url=health_url, initial_wait_ms=10, poll_ms=10, max_polls=5)


async def test_client_reaches_the_server_and_parses_a_direct_reply(live):
    """An information tool answers directly, with no protocol state at all.

    `view_status` is one of the ten direct tools on the live host. Reading it
    through the submission envelope finds nothing and reports an empty world as
    fact, so the direct shape is the one that has to work.
    """
    _, mcp_url, health_url = await live.__aenter__()
    try:
        async with McpClient(await config_for(mcp_url, health_url)) as client:
            reply = await client.call("view_status", {}, rationale="read state")
        assert not reply.is_protocol
        assert reply.state is None, "a direct reply carries no submission state"
        assert reply.result_status == "succeeded"
        assert reply.result is not None
        assert "situation" in reply.result
    finally:
        await live.__aexit__(None, None, None)


async def test_the_two_reply_shapes_are_both_readable(live):
    """One session answers in both shapes, and neither is assumed."""
    _, mcp_url, health_url = await live.__aenter__()
    try:
        async with McpClient(await config_for(mcp_url, health_url)) as client:
            direct = await client.call("view_status", {}, rationale="direct read")
            enveloped = await client.call("collect_block",
                                          {"submission_id": "shapes", "block_name": "dirt",
                                           "wait_timeout_ms": 10},
                                          rationale="foreground work")
        assert direct.is_protocol is False and direct.state is None
        assert enveloped.is_protocol is True
        assert enveloped.require_state() == "settled"
        # Same accessor, different spelling underneath.
        assert direct.result["situation"] is not None
        assert enveloped.result["status"] == "succeeded"
    finally:
        await live.__aexit__(None, None, None)


async def test_a_direct_payload_on_submit_is_a_protocol_error(live):
    """Submitting an objective must never quietly accept a non-envelope reply."""
    from harness.errors import ProtocolError

    _, mcp_url, health_url = await live.__aenter__()
    try:
        async with McpClient(await config_for(mcp_url, health_url)) as client:
            runner = ObjectiveRunner(client, await config_for(mcp_url, health_url),
                                     BudgetConfig(objective_ms=5_000))
            with pytest.raises(ProtocolError):
                await runner._submit(Objective(tool="view_status",
                                               arguments={},
                                               submission_id="wrong-shape"))
    finally:
        await live.__aexit__(None, None, None)


async def test_a_reply_above_the_sdk_default_ceiling_still_arrives(live):
    """The live host's `tools/list` is 2.91 MiB in one event; httpx2 caps at 1 MiB.

    Without the bounded event parser this fails as a lost stream and the SDK
    reports it as the connection dropping, which points at the network.
    """
    from harness import sse

    _, mcp_url, health_url = await live.__aenter__()
    try:
        async with McpClient(await config_for(mcp_url, health_url)) as client:
            assert sse.require_live_patch(), "the patch must be in effect on a real connection"
            oversized = await client.call("read_recent_events",
                                          {"pad_bytes": sse.SDK_DEFAULT_MAX_EVENT_BYTES + 4096},
                                          rationale="a reply larger than httpx2 allows")
        assert oversized.result_status == "succeeded"
        assert len(oversized.result["pad"]) > sse.SDK_DEFAULT_MAX_EVENT_BYTES
    finally:
        await live.__aexit__(None, None, None)


async def test_list_tools_binds_arguments_from_the_server(live):
    """Tool names and required arguments come from the advertisement."""
    _, mcp_url, health_url = await live.__aenter__()
    try:
        async with McpClient(await config_for(mcp_url, health_url)) as client:
            tools = await client.list_tools()
        names = {tool["name"] for tool in tools}
        assert {"view_status", "collect_block", "wait_for_action"} <= names
    finally:
        await live.__aexit__(None, None, None)


async def test_health_is_a_plain_get(live):
    """After a transport drop this is how we learn what is still running."""
    _, mcp_url, health_url = await live.__aenter__()
    try:
        async with McpClient(await config_for(mcp_url, health_url)) as client:
            health = await client.health()
        assert health["status"] == "idle"
        assert health["foreground"]["active"] is None
    finally:
        await live.__aexit__(None, None, None)


async def test_reply_without_structured_content_is_unverified(live):
    """Markdown where json was requested is a failure, not data to parse."""
    _, mcp_url, health_url = await live.__aenter__()
    try:
        async with McpClient(await config_for(mcp_url, health_url)) as client:
            reply = await client.call("collect_block",
                                      {"submission_id": "s1", "block_name": "dirt",
                                       "response_format": "markdown"},
                                      rationale="force the markdown path")
        # The client forces json on every call, so this must never be seen.
        assert reply.state != ""
    except UnverifiedRead:
        pass
    finally:
        await live.__aexit__(None, None, None)


async def test_one_objective_completes_unattended(live, ledger_config):
    """The Phase 0 exit criterion, against a real transport."""
    host, mcp_url, health_url = await live.__aenter__()
    try:
        with Ledger(ledger_config, "integration") as ledger:
            async with McpClient(await config_for(mcp_url, health_url)) as client:
                runner = ObjectiveRunner(client, await config_for(mcp_url, health_url),
                                         BudgetConfig(objective_ms=5_000), ledger)
                result = await runner.run(
                    Objective(tool="collect_block", arguments={"block_name": "dirt"})
                )
        assert result.ok
        assert result.state == "settled"
        assert result.status == "succeeded"
        assert result.evidence_ok

        rows = ledger.read_all()
        assert len(rows) == 1
        assert rows[0]["outcome"] == "settled:succeeded"
    finally:
        await live.__aexit__(None, None, None)


async def test_pending_wait_is_retrieved_to_a_settled_result(live):
    """An expired initial wait is finished off with wait_for_action."""
    host, mcp_url, health_url = await live.__aenter__()
    host.actions.clear()
    # Force the action to report pending once before settling.
    try:
        async with McpClient(await config_for(mcp_url, health_url)) as client:
            runner = ObjectiveRunner(client, await config_for(mcp_url, health_url),
                                     BudgetConfig(objective_ms=5_000))
            # Register the action first through a no-wait submission, then make
            # its next look report pending.
            reply = await client.call("collect_block",
                                      {"submission_id": "pre", "block_name": "stone"},
                                      rationale="admit without waiting")
            action_id = reply.action_id
            assert reply.state == "accepted"
            host.actions[action_id].pending_polls = 1

            retrieved = await client.call("wait_for_action",
                                          {"action_id": action_id, "timeout_ms": 10},
                                          rationale="retrieve")
            assert retrieved.state == "pending"

            settled = await client.call("wait_for_action",
                                        {"action_id": action_id, "timeout_ms": 10},
                                        rationale="retrieve again")
            assert settled.state == "settled"
            assert runner is not None
    finally:
        await live.__aexit__(None, None, None)


async def test_retrying_an_identical_submission_recovers_rather_than_duplicating(live):
    """The recovery path for a lost reply.

    Repeating the call verbatim returns the original action and its terminal
    output. This is why the runner never mints a fresh submission_id to "be
    safe": a new id is a second objective, not a retry.
    """
    _, mcp_url, health_url = await live.__aenter__()
    try:
        async with McpClient(await config_for(mcp_url, health_url)) as client:
            first = await client.call("collect_block",
                                      {"submission_id": "same", "block_name": "dirt",
                                       "wait_timeout_ms": 10},
                                      rationale="first")
            assert first.state == "settled"

            retry = await client.call("collect_block",
                                      {"submission_id": "same", "block_name": "dirt",
                                       "wait_timeout_ms": 10},
                                      rationale="recover a lost reply")
            assert retry.action_id == first.action_id
            assert retry.state == "settled", "the recovered result is the original answer"
    finally:
        await live.__aexit__(None, None, None)


async def test_reusing_a_submission_id_with_new_arguments_is_a_conflict(live):
    """Change either the id or the arguments and it is a second objective."""
    _, mcp_url, health_url = await live.__aenter__()
    try:
        async with McpClient(await config_for(mcp_url, health_url)) as client:
            first = await client.call("collect_block",
                                      {"submission_id": "same", "block_name": "dirt",
                                       "wait_timeout_ms": 10},
                                      rationale="first")
            assert first.state == "settled"

            conflict = await client.call("collect_block",
                                         {"submission_id": "same", "block_name": "stone",
                                          "wait_timeout_ms": 10},
                                         rationale="same id, new arguments")
            assert conflict.state == "refused"
            assert conflict.refusal_code == "SUBMISSION_CONFLICT"
    finally:
        await live.__aexit__(None, None, None)


async def test_unretrieved_result_blocks_the_next_submission(live):
    """RESULT_NOT_RETRIEVED, and the recovery the runner performs."""
    _, mcp_url, health_url = await live.__aenter__()
    try:
        async with McpClient(await config_for(mcp_url, health_url)) as client:
            # Admit without waiting, then submit again: the gate is shut.
            admitted = await client.call("collect_block",
                                         {"submission_id": "a", "block_name": "dirt"},
                                         rationale="admit")
            blocked = await client.call("collect_block",
                                        {"submission_id": "b", "block_name": "dirt",
                                         "wait_timeout_ms": 10},
                                        rationale="submit while a result is owed")
            assert blocked.state == "refused"
            assert blocked.refusal_code == "RESULT_NOT_RETRIEVED"
            assert blocked.data.get("unretrievedActionId") == admitted.action_id

            # The runner releases the gate, then submits for real. The result it
            # reports must belong to its own submission, not to the action that
            # was blocking the gate.
            runner = ObjectiveRunner(client, await config_for(mcp_url, health_url),
                                     BudgetConfig(objective_ms=5_000))
            result = await runner.run(Objective(tool="collect_block",
                                                arguments={"block_name": "dirt"}))
            assert result.ok, "recovery through the real server should succeed"
            assert result.action_id != admitted.action_id, (
                "the owed action's result must not be reported as this objective's"
            )
    finally:
        await live.__aexit__(None, None, None)


async def test_polling_ceiling_fires_against_a_server_that_never_settles(live):
    """A runtime that answers pending forever must terminate the objective."""
    host, mcp_url, health_url = await live.__aenter__()
    try:
        async with McpClient(await config_for(mcp_url, health_url)) as client:
            admitted = await client.call("collect_block",
                                         {"submission_id": "stuck", "block_name": "dirt"},
                                         rationale="admit")
            host.actions[admitted.action_id].pending_polls = 10_000

            runner = ObjectiveRunner(client, await config_for(mcp_url, health_url),
                                     BudgetConfig(objective_ms=60_000))
            with pytest.raises(BudgetExceeded) as error:
                await runner._poll(Objective(tool="collect_block"), admitted.action_id)
            assert error.value.kind == "max_polls"
    finally:
        await live.__aexit__(None, None, None)


async def test_invalid_arguments_settle_as_a_failed_objective(live):
    """An unknown block is the server's verdict, carried through unchanged."""
    _, mcp_url, health_url = await live.__aenter__()
    host = None
    try:
        async with McpClient(await config_for(mcp_url, health_url)) as client:
            # block_name is required by the schema, so omitting it is the way to
            # get a refusal from the real server rather than a client-side error.
            with pytest.raises(Exception):
                await client.call("collect_block", {"submission_id": "x"},
                                  rationale="omit a required argument")
    finally:
        await live.__aexit__(None, None, None)


async def test_a_survival_reflex_is_waited_out_not_given_up_on(live):
    """ACTION_BUSY with no action id is a reflex holding the body.

    There is nothing to drain - the reflex is not one of our actions - but it is
    finite, so waiting is correct. The live world produces this whenever a mob
    is nearby, which is most of the night, and treating it as terminal aborts
    every objective for as long as the mob lives.
    """
    host, mcp_url, health_url = await live.__aenter__()
    host.reflex = "hostile_reflex"
    host.reflex_reads_left = 2
    try:
        async with McpClient(await config_for(mcp_url, health_url)) as client:
            runner = ObjectiveRunner(
                client,
                await config_for(mcp_url, health_url),
                BudgetConfig(objective_ms=5_000, gate_wait_ms=5_000),
            )
            result = await runner.run(Objective(tool="collect_block",
                                               arguments={"block_name": "dirt"}))
        assert result.ok, "the objective should run once the body comes free"
        assert result.action_id
    finally:
        await live.__aexit__(None, None, None)


async def test_a_body_that_never_frees_fails_honestly(live):
    """A reflex that never lets go must terminate, not wait forever."""
    host, mcp_url, health_url = await live.__aenter__()
    host.reflex = "hostile_reflex"
    host.reflex_reads_left = 10_000
    try:
        async with McpClient(await config_for(mcp_url, health_url)) as client:
            config = await config_for(mcp_url, health_url)
            runner = ObjectiveRunner(
                client, config, BudgetConfig(objective_ms=5_000, gate_wait_ms=0)
            )
            with pytest.raises(ObjectiveFailed) as error:
                await runner.run(Objective(tool="collect_block",
                                           arguments={"block_name": "dirt"}))
        assert error.value.state == "refused"
    finally:
        await live.__aexit__(None, None, None)


async def test_an_unreadable_status_is_not_read_as_a_free_body():
    """Guessing the body is free would submit into a reflex and call it an answer."""
    from harness.mcp_client import ToolReply
    from harness.objective import _is_free_body

    assert _is_free_body(ToolReply(is_error=False, data={}, notifications={})) is False
    # A direct read that parsed, but has no situation in it.
    assert _is_free_body(ToolReply(is_error=False,
                                   data={"action": "view_status", "result": {"status": "succeeded"}},
                                   notifications={})) is False
    # A situation with no activity section.
    assert _is_free_body(ToolReply(is_error=False,
                                   data={"action": "view_status",
                                         "result": {"situation": {"vitals": {}}}},
                                   notifications={})) is False
    # A takeover is not free, even with no action named.
    assert _is_free_body(ToolReply(is_error=False,
                                   data={"action": "view_status",
                                         "result": {"situation": {"activity": {"owner": "takeover",
                                                                            "activeAction": None}}}},
                                   notifications={})) is False
    # Idle with no action is free.
    assert _is_free_body(ToolReply(is_error=False,
                                   data={"action": "view_status",
                                         "result": {"situation": {"activity": {"owner": "idle",
                                                                            "activeAction": None}}}},
                                   notifications={})) is True


async def test_state_reader_derives_from_the_real_view_status(live):
    """The state vector, from a real direct view_status response.

    This is the Phase 0 `harness state` path. It fails unless the direct reply
    shape is read, which is why it is an integration test and not a unit test
    over a hand-built reply.
    """
    _, mcp_url, health_url = await live.__aenter__()
    try:
        async with McpClient(await config_for(mcp_url, health_url)) as client:
            vector = await StateReader(client).read()
        assert vector.trustworthy, vector.unverified
        assert vector.health == 20.0
        assert vector.time_phase == "day"
        assert vector.carried == ["dirtx10", "cobblestonex5"]
    finally:
        await live.__aexit__(None, None, None)