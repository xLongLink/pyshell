import json
import httpx
import pytest
import asyncio
from types import SimpleNamespace
from web.src.shell import RemoteShell
from web.src.workspace import Workspace


@pytest.mark.parametrize(
    "events",
    [
        [{"type": "stdout_delta", "text": "partial"}],
        [{"type": "execution_finished"}],
        [{"type": "result", "stdout": "", "stderr": "", "exception": None, "done": True}],
        [{"type": "unknown"}],
    ],
)
def test_unconfirmed_stream_is_not_replayed_and_blocks_more_work(events):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, text="\n".join(json.dumps(event) for event in events))

    async def run():
        shell = RemoteShell()
        shell.client = httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url=shell.url)
        try:
            with pytest.raises(RuntimeError, match="may already have run"):
                await shell.stream("value = 42", lambda _event: None)
            assert shell.uncertain is True

            # Further submissions must fail locally instead of replaying code.
            with pytest.raises(RuntimeError, match="uncertain"):
                await shell.stream("value = 42", lambda _event: None)
            assert len(requests) == 1
        finally:
            await shell.close()

    asyncio.run(run())


def test_confirmed_stream_forwards_events_in_order_without_terminal_marker():
    events = [
        {"type": "stdout_delta", "text": "output\n"},
        {"type": "stderr_delta", "text": "warning\n"},
        {"type": "result", "stdout": "output\n", "stderr": "warning\n", "exception": None, "done": True},
        {"type": "execution_finished"},
    ]

    async def run():
        shell = RemoteShell()
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text="\n".join(json.dumps(event) for event in events)))
        shell.client = httpx.AsyncClient(transport=transport, base_url=shell.url)
        received = []
        try:
            result = await shell.stream("print('output')", received.append)
            assert received == events[:-1]
            assert result == {"stdout": "output\n", "stderr": "warning\n", "exception": None, "done": True, "ui": []}
            assert shell.uncertain is False
        finally:
            await shell.close()

    asyncio.run(run())


def test_compacted_snapshot_retains_current_widgets_and_slow_clients_resynchronize():
    workspace = Workspace(SimpleNamespace(model="test"))
    create = {"version": "v0.9", "createSurface": {"surfaceId": "sales", "catalogId": "urn:pyshell:catalog:v3"}}
    update = {"version": "v0.9", "updateComponents": {"surfaceId": "sales", "components": [{"id": "root"}]}}
    workspace.publish({"type": "a2ui", "messages": [create, update]})
    queue = asyncio.Queue(maxsize=3)
    queue.put_nowait({"type": "voice", "name": "answer", "sdp": "answer"})
    workspace.listeners.add(queue)

    # Overflow both replay history and the subscriber's tiny queue.
    for _ in range(2100):
        workspace.publish({"type": "stdout_delta", "text": "x"})
    assert len(workspace.events) <= 2000
    snapshot = workspace.snapshot()
    assert snapshot["placements"] == {"sales": {"pinned": False}}
    assert [event for event in snapshot["events"] if event["type"] == "a2ui"] == [{"type": "a2ui", "messages": [create, update]}]

    # Resynchronization must preserve the pending voice answer.
    queued = []
    while not queue.empty():
        queued.append(queue.get_nowait())
    assert any(event["type"] == "snapshot" for event in queued)
    assert any(event["type"] == "voice" for event in queued)
