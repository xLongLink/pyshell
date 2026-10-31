import os
import httpx
import pytest
import asyncio
from types import SimpleNamespace
from importlib import import_module
from threading import Event
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from sample.src.workspace import Command, Workspace
from starlette.websockets import WebSocketDisconnect

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
    server = import_module("sample.main")


def test_lifespan_drains_accepted_work_before_closing_transports(monkeypatch):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    provider = SimpleNamespace(client=SimpleNamespace(close=AsyncMock()))
    monkeypatch.setattr(server, "OpenAIProvider", lambda **kwargs: provider)
    monkeypatch.setattr(server, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(server, "OpenAIResponsesModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(server, "OpenAIRealtimeModel", lambda *args, **kwargs: object())

    async def run():
        started = asyncio.Event()
        release = asyncio.Event()
        exiting = asyncio.Event()

        async def execute(instruction, emit):
            started.set()
            await release.wait()
            assert not client.is_closed
            provider.client.close.assert_not_awaited()
            return "Confirmed before shutdown"

        monkeypatch.setattr(server, "Agent", lambda *args, **kwargs: SimpleNamespace(arun=execute, shell=SimpleNamespace(uncertain=False)))

        async def application():
            async with server.lifespan(server.app):
                server.app.state.workspace.start(Command(type="run", instruction="Finish accepted work"))
                await started.wait()
                exiting.set()

        owner = asyncio.create_task(application())
        try:
            await asyncio.wait_for(exiting.wait(), 1)
            workspace = server.app.state.workspace
            assert workspace.closing
            assert not owner.done()
            assert not client.is_closed
            with pytest.raises(RuntimeError, match="shutting down"):
                workspace.start(Command(type="run", instruction="Do not accept"))
        finally:
            release.set()
            await asyncio.wait_for(owner, 1)
        assert client.is_closed
        provider.client.close.assert_awaited_once_with()
        assert workspace.status()["latest_outcome"]["result"] == "Confirmed before shutdown"
        assert workspace.task.result() is None

    asyncio.run(run())


def test_lifespan_releases_transports_on_initialization_failure(monkeypatch):
    # Arrange
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    provider = SimpleNamespace(client=SimpleNamespace(close=AsyncMock()))
    monkeypatch.setattr(server, "OpenAIProvider", lambda **kwargs: provider)
    monkeypatch.setattr(server, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(server, "OpenAIResponsesModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(server, "Agent", lambda *args, **kwargs: SimpleNamespace())

    def realtime(*args, **kwargs):
        raise RuntimeError("Initialization failed")

    monkeypatch.setattr(server, "OpenAIRealtimeModel", realtime)

    async def run():
        # Act
        # Assert
        with pytest.raises(RuntimeError, match="Initialization failed"):
            async with server.lifespan(server.app):
                pytest.fail("Initialization must fail before entering the lifespan body")
        assert client.is_closed
        provider.client.close.assert_awaited_once_with()

    asyncio.run(run())


def test_lifespan_releases_transports_on_shutdown_failure(monkeypatch):
    # Arrange
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    provider = SimpleNamespace(client=SimpleNamespace(close=AsyncMock()))
    monkeypatch.setattr(server, "OpenAIProvider", lambda **kwargs: provider)
    monkeypatch.setattr(server, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(server, "OpenAIResponsesModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(server, "OpenAIRealtimeModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(server, "Agent", lambda *args, **kwargs: SimpleNamespace())

    async def run():
        # Act
        # Assert
        with pytest.raises(RuntimeError, match="Shutdown failed"):
            async with server.lifespan(server.app):
                monkeypatch.setattr(server.app.state.workspace, "close", AsyncMock(side_effect=RuntimeError("Shutdown failed")))
        assert client.is_closed
        provider.client.close.assert_awaited_once_with()

    asyncio.run(run())


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
def test_sample_renders_voice_and_isolated_container_frame_without_frontend_build(host):
    client = TestClient(server.app, base_url=f"http://{host}:8010")
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "PyShell" in response.text
    assert f'src="http://{host}:8000/"' in response.text
    assert 'sandbox="allow-scripts allow-same-origin"' in response.text
    assert 'referrerpolicy="no-referrer"' in response.text
    assert 'aria-label="Start voice"' in response.text
    assert 'import { mountVoice } from "/voice.js"' in response.text
    assert "test-key" not in response.text
    script = client.get("/voice.js")
    assert script.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")
    assert "export function mountVoice" in script.text
    assert client.get("/missing.js").status_code == 404


@pytest.mark.parametrize(
    "origin", [None, "http://localhost:8000", "http://localhost:5180", "https://example.com", "http://localhost:5180.evil.example"]
)
def test_browser_rejects_untrusted_origin(origin):
    headers = {"origin": origin} if origin else {}
    with pytest.raises(WebSocketDisconnect) as error:
        with TestClient(server.app).websocket_connect("/ws", headers=headers):
            pass
    assert error.value.code == 1008


def test_web_server_rejects_untrusted_host():
    assert TestClient(server.app).get("/", headers={"host": "evil.example"}).status_code == 400


def test_browser_executes_once_and_reconnects_without_replay(monkeypatch):
    execute = AsyncMock(return_value="Completed")
    workspace = Workspace(SimpleNamespace(arun=execute, shell=SimpleNamespace(uncertain=False)))
    completed = Event()
    workspace.results.add(lambda record: completed.set())
    monkeypatch.setattr(server.app.state, "workspace", workspace, raising=False)
    monkeypatch.setattr(server.app.state, "realtime_model", None, raising=False)
    client = TestClient(server.app)
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
        assert socket.receive_json()["type"] == "snapshot"
        socket.send_json({"type": "run", "instruction": "show sales"})
        assert socket.receive_json() == {"type": "instruction", "text": "show sales"}
        assert completed.wait(1)
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
        snapshot = socket.receive_json()
        assert snapshot["type"] == "snapshot"
    execute.assert_awaited_once_with("show sales", workspace.publish)


@pytest.mark.parametrize(
    ("command", "error"),
    [
        ({"type": "widget", "surface_id": "sales", "action": "pin"}, "validation error"),
        ({"type": "reset"}, "validation error"),
        ({"type": "run", "instruction": " "}, "Enter an instruction."),
    ],
)
def test_invalid_commands_do_not_start_work(monkeypatch, command, error):
    execute = AsyncMock()
    workspace = Workspace(SimpleNamespace(arun=execute))
    monkeypatch.setattr(server.app.state, "workspace", workspace, raising=False)
    monkeypatch.setattr(server.app.state, "realtime_model", None, raising=False)
    with TestClient(server.app).websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
        assert socket.receive_json()["type"] == "snapshot"
        socket.send_json(command)
        event = socket.receive_json()
        assert event["type"] == "error"
        assert error in event["text"]
        execute.assert_not_awaited()
        assert workspace.task is None
