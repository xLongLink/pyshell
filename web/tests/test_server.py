import pytest
import server
import asyncio
import threading
from script import Settings
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


@pytest.fixture
def browser_api(monkeypatch, model_for, remote_shell):
    requests = []
    model = model_for(["print('browser result')\ndone()"], ["print('next result')\ndone()"], requests=requests)
    monkeypatch.setattr(server, "Settings", lambda: Settings(_env_file=None, openai_api_key="test-key"))
    monkeypatch.setattr(server, "OpenAIResponsesModel", lambda *args, **kwargs: model)
    monkeypatch.setattr(server, "RemoteShell", lambda url: remote_shell)
    with TestClient(server.app) as client:
        yield client, requests


def receive_until_idle(socket):
    events = []
    while True:
        event = socket.receive_json()
        events.append(event)
        if event == {"type": "state", "busy": False}:
            return events


def test_web_server_serves_website_and_assets(browser_api, monkeypatch, tmp_path):
    client, _ = browser_api
    (tmp_path / "index.html").write_text("<!doctype html><title>PyShell</title>")
    (tmp_path / "app.js").write_text("console.log('pyshell')")
    monkeypatch.setattr(server.frontend, "directory", str(tmp_path))
    monkeypatch.setattr(server.frontend, "all_directories", [str(tmp_path)])
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "PyShell" in response.text
    assert client.get("/app.js").text == "console.log('pyshell')"
    assert client.get("/missing.js").status_code == 404


def test_web_server_origin_can_connect_to_agent(browser_api):
    client, _ = browser_api
    with client.websocket_connect("/ws", headers={"origin": "http://127.0.0.1:8010"}) as socket:
        assert socket.receive_json()["type"] == "snapshot"


def test_browser_executes_and_reconnects_without_replaying(browser_api):
    client, requests = browser_api
    with client.websocket_connect("/ws", headers={"origin": "http://127.0.0.1:5180"}) as socket:
        assert socket.receive_json()["type"] == "snapshot"
        socket.send_json({"type": "run", "instruction": "print a result"})
        events = receive_until_idle(socket)
        assert "print('browser result')" in "".join(event.get("text", "") for event in events if event["type"] == "code_delta")
        assert any(event["type"] == "result" and event["stdout"] == "browser result\n" for event in events)
        assert {"type": "done"} in events
    with client.websocket_connect("/ws", headers={"origin": "http://127.0.0.1:5180"}) as socket:
        snapshot = socket.receive_json()
        assert snapshot["busy"] is False
        assert any(event["type"] == "done" for event in snapshot["events"])
        assert len(requests) == 1


def test_browser_reset_clears_shared_state(browser_api, remote_shell):
    client, _ = browser_api
    remote_shell.execute("value = 42")
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:5180"}) as socket:
        socket.receive_json()
        socket.send_json({"type": "reset"})
        assert {"type": "reset"} in receive_until_idle(socket)
        assert "NameError" in remote_shell.execute("print(value)")["exception"]
        assert client.app.state.workspace.agent.message_history == []


def test_browser_reuses_the_provider_event_loop(browser_api, monkeypatch):
    client, requests = browser_api
    agent = client.app.state.workspace.agent
    run = agent.run
    threads = []

    def tracked_run(instruction, emit):
        threads.append((threading.get_ident(), asyncio.get_event_loop()))
        run(instruction, emit)

    monkeypatch.setattr(agent, "run", tracked_run)
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:5180"}) as socket:
        socket.receive_json()
        for instruction in ("first", "second"):
            socket.send_json({"type": "run", "instruction": instruction})
            assert {"type": "done"} in receive_until_idle(socket)
    assert len(requests) == 2
    assert len(threads) == 2
    assert threads[0] == threads[1]


def test_browser_reports_model_failure_without_executing(browser_api, model_for, remote_shell):
    client, _ = browser_api
    agent = client.app.state.workspace.agent
    agent.agent = server.Agent(model_for(["value = 42\n", ConnectionError("stream interrupted")]), shell=remote_shell).agent
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:5180"}) as socket:
        socket.receive_json()
        socket.send_json({"type": "run", "instruction": "set value"})
        events = receive_until_idle(socket)
        assert any(event["type"] == "error" and "stream interrupted" in event["text"] for event in events)
        assert not any(event["type"] in {"result", "done"} for event in events)
    assert "NameError" in remote_shell.execute("print(value)")["exception"]


def test_browser_rejects_untrusted_origin(browser_api):
    client, _ = browser_api
    with pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect("/ws", headers={"origin": "https://example.com"}):
            pass
    assert error.value.code == 1008


def test_invalid_command_does_not_start_work(browser_api):
    client, requests = browser_api
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:5180"}) as socket:
        socket.receive_json()
        socket.send_json({"type": "run", "instruction": " "})
        events = receive_until_idle(socket)
        assert events[0]["type"] == "error"
        assert requests == []


def test_disconnect_does_not_cancel_work_or_allow_overlap(browser_api, monkeypatch):
    client, _ = browser_api
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def run(instruction, emit):
        started.set()
        assert release.wait(timeout=5)
        emit({"type": "result", "stdout": "finished\n", "stderr": "", "exception": None, "done": True})
        finished.set()

    monkeypatch.setattr(client.app.state.workspace.agent, "run", run)
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:5180"}) as socket:
        socket.receive_json()
        socket.send_json({"type": "run", "instruction": "work"})
        assert started.wait(timeout=5)
    try:
        with client.websocket_connect("/ws", headers={"origin": "http://localhost:5180"}) as socket:
            assert socket.receive_json()["busy"] is True
            socket.send_json({"type": "reset"})
            assert socket.receive_json()["type"] == "error"
            assert socket.receive_json() == {"type": "state", "busy": True}
            release.set()
            events = receive_until_idle(socket)
            assert {"type": "done"} in events
            assert finished.is_set()
    finally:
        release.set()
