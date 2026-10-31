import json
import httpx
import pytest
import asyncio
from types import SimpleNamespace
from sample.src.shell import RemoteShell
from sample.src.workspace import Workspace


@pytest.mark.parametrize(
    "events",
    [
        [{"type": "widget", "surface_id": "sales", "action": "focus"}],
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
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://testserver") as client:
            shell = RemoteShell(client)
            with pytest.raises(RuntimeError, match="may already have run"):
                await shell.stream("value = 42")
            assert shell.uncertain is True

            # Further submissions must fail locally instead of replaying code.
            with pytest.raises(RuntimeError, match="uncertain"):
                await shell.stream("value = 42")
            assert len(requests) == 1

    asyncio.run(run())


def test_confirmed_stream_does_not_relay_widgets_to_application():
    events = [
        {"type": "a2ui", "messages": [{"version": "v0.9", "createSurface": {"surfaceId": "sales", "catalogId": "urn:pyshell:catalog:v3"}}]},
        {"type": "widget", "surface_id": "sales", "action": "pin"},
        {"type": "result", "stdout": "output\n", "stderr": "warning\n", "exception": None, "done": True},
        {"type": "execution_finished"},
    ]

    async def run():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text="\n".join(json.dumps(event) for event in events)))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            shell = RemoteShell(client)
            result = await shell.stream("print('output')")
            assert result == {"stdout": "output\n", "stderr": "warning\n", "exception": None, "done": True}
            assert shell.uncertain is False

    asyncio.run(run())


def test_slow_application_clients_preserve_voice_controls_without_widget_state():
    workspace = Workspace(SimpleNamespace())
    queue = asyncio.Queue(maxsize=3)
    queue.put_nowait({"type": "voice", "name": "answer", "sdp": "answer"})
    workspace.listeners.add(queue)

    # Overflow the subscriber's tiny queue repeatedly.
    for _ in range(10):
        workspace.publish({"type": "error", "text": "x"})

    # Resynchronization must preserve the pending voice answer.
    queued = []
    while not queue.empty():
        queued.append(queue.get_nowait())
    assert any(event["type"] == "snapshot" for event in queued)
    assert any(event["type"] == "voice" for event in queued)
