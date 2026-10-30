import json
import httpx
import pytest
import asyncio
from types import SimpleNamespace
from web.src.script import RemoteShell
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
            with pytest.raises(RuntimeError, match="uncertain"):
                await shell.stream("value = 42", lambda _event: None)
            assert len(requests) == 1
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
    for _ in range(2100):
        workspace.publish({"type": "stdout_delta", "text": "x"})
    assert len(workspace.events) <= 2000
    snapshot = workspace.snapshot()
    assert snapshot["placements"] == {"sales": {"pinned": False}}
    assert [event for event in snapshot["events"] if event["type"] == "a2ui"] == [{"type": "a2ui", "messages": [create, update]}]
    queued = []
    while not queue.empty():
        queued.append(queue.get_nowait())
    assert any(event["type"] == "snapshot" for event in queued)
    assert any(event["type"] == "voice" for event in queued)
