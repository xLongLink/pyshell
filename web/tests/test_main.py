import os
import httpx
import pytest
import asyncio
from types import SimpleNamespace
from importlib import import_module
from threading import Event
from unittest.mock import AsyncMock, patch
from web.src.workspace import Command, Workspace
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
    server = import_module("web.main")


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


@pytest.mark.parametrize("failure", ["initialization", "shutdown"])
def test_lifespan_releases_transports_on_failure(monkeypatch, failure):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    provider = SimpleNamespace(client=SimpleNamespace(close=AsyncMock()))
    monkeypatch.setattr(server, "OpenAIProvider", lambda **kwargs: provider)
    monkeypatch.setattr(server, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(server, "OpenAIResponsesModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(server, "Agent", lambda *args, **kwargs: SimpleNamespace())

    def realtime(*args, **kwargs):
        if failure == "initialization":
            raise RuntimeError("Initialization failed")
        return object()

    monkeypatch.setattr(server, "OpenAIRealtimeModel", realtime)

    async def run():
        with pytest.raises(RuntimeError, match="failed"):
            async with server.lifespan(server.app):
                monkeypatch.setattr(server.app.state.workspace, "close", AsyncMock(side_effect=RuntimeError("Shutdown failed")))
        assert client.is_closed
        provider.client.close.assert_awaited_once_with()

    asyncio.run(run())


def test_web_server_serves_website_and_assets(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>PyShell</title>")
    (tmp_path / "app.js").write_text("console.log('pyshell')")
    monkeypatch.setattr(server.frontend, "directory", str(tmp_path))
    monkeypatch.setattr(server.frontend, "all_directories", [str(tmp_path)])
    client = TestClient(server.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "PyShell" in response.text
    assert client.get("/app.js").text == "console.log('pyshell')"
    assert client.get("/missing.js").status_code == 404


def test_geography_proxy_preserves_data_and_cache_headers_without_execution(monkeypatch):
    headers = {"cache-control": "public, max-age=3600", "etag": '"map-v1"', "last-modified": "Thu, 01 Oct 2026 00:00:00 GMT"}
    geography = AsyncMock(return_value=httpx.Response(200, content=b'{"type":"FeatureCollection","features":[]}', headers=headers))
    workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(geography=geography, uncertain=True)))
    monkeypatch.setattr(server.app.state, "workspace", workspace, raising=False)
    response = TestClient(server.app).get("/geographies/CHE")
    assert response.status_code == 200
    assert response.content == geography.return_value.content
    assert response.headers["content-type"] == "application/geo+json"
    for name, value in headers.items():
        assert response.headers[name] == value
    geography.assert_awaited_once_with("CHE")
    assert workspace.snapshot()["events"] == []
    assert workspace.agent.shell.uncertain is True


@pytest.mark.parametrize("status", [None, 404, 500])
def test_geography_proxy_reports_backend_failure_without_leaking_details(monkeypatch, status):
    request = httpx.Request("GET", "http://execution-private/geographies/CHE")
    error = httpx.ConnectError("private backend address", request=request)
    if status is not None:
        error = httpx.HTTPStatusError("private backend address", request=request, response=httpx.Response(status))
    geography = AsyncMock(side_effect=error)
    monkeypatch.setattr(
        server.app.state, "workspace", SimpleNamespace(agent=SimpleNamespace(shell=SimpleNamespace(geography=geography))), raising=False
    )
    response = TestClient(server.app).get("/geographies/CHE")
    assert response.status_code == (404 if status == 404 else 502)
    assert response.json() == {"detail": "Unsupported geography" if status == 404 else "Map data is unavailable"}


@pytest.mark.parametrize("geography", ["sources", "USA.json", "__proto__"])
def test_geography_proxy_rejects_invalid_assets(geography):
    response = TestClient(server.app).get(f"/geographies/{geography}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Unsupported geography"}


@pytest.mark.parametrize("origin", [None, "https://example.com", "http://localhost:5180.evil.example"])
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
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:5180"}) as socket:
        assert socket.receive_json()["type"] == "snapshot"
        socket.send_json({"type": "run", "instruction": "show sales"})
        assert socket.receive_json() == {"type": "instruction", "text": "show sales"}
        assert completed.wait(1)
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:5180"}) as socket:
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
    with TestClient(server.app).websocket_connect("/ws", headers={"origin": "http://localhost:5180"}) as socket:
        assert socket.receive_json()["type"] == "snapshot"
        socket.send_json(command)
        event = socket.receive_json()
        assert event["type"] == "error"
        assert error in event["text"]
        execute.assert_not_awaited()
        assert workspace.task is None
