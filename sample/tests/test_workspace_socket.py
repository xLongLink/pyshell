import pytest
import asyncio
import importlib
from types import SimpleNamespace
from contextlib import contextmanager
from fastapi.testclient import TestClient
from sample.src.workspace import Workspace


@pytest.fixture
def browser(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    main = importlib.import_module("sample.main")
    executions = []

    async def arun(instruction, publish):
        executions.append(instruction)
        return "completed"

    workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(uncertain=False), arun=arun))
    monkeypatch.setattr(main.app.state, "workspace", workspace, raising=False)
    monkeypatch.setattr(main.app.state, "realtime_model", object(), raising=False)
    disconnected = asyncio.Event()

    async def application(scope, receive, send):
        try:
            await main.app(scope, receive, send)
        finally:
            disconnected.set()

    async def wait_closed():
        async with asyncio.timeout(2):
            await disconnected.wait()
            if workspace.task is not None:
                await workspace.task

    @contextmanager
    def connect():
        with TestClient(application).websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
            try:
                yield socket
            finally:
                socket.close()
                socket.portal.call(wait_closed)

    return SimpleNamespace(connect=connect, workspace=workspace, executions=executions)


@pytest.mark.parametrize("state", ["idle", "busy", "uncertain", "closing"])
@pytest.mark.parametrize(
    ("command", "validation"),
    [
        ({"type": "run", "instruction": "x" * 8001}, "8000"),
        ({"type": "run", "instruction": 123}, "valid string"),
        ({"type": "run", "instruction": None}, "valid string"),
        ({"type": "run", "instruction": " \n\t"}, None),
        ({"type": "run"}, None),
        ({"type": "run", "instruction": "calculate"}, None),
        ({"type": "run", "instruction": "calculate", "extra": True}, "Extra inputs"),
        ({"type": "other", "instruction": "calculate"}, "'run'"),
    ],
)
def test_browser_validates_shared_workspace_submission(browser, state, command, validation):
    if state == "idle" and command == {"type": "run", "instruction": "calculate"}:
        with browser.connect() as socket:
            assert socket.receive_json() == {"type": "snapshot"}
            socket.send_json(command)
            assert socket.receive_json() == {"type": "instruction", "text": "calculate"}
        assert browser.executions == ["calculate"]
        assert browser.workspace.recent_tasks[-1]["outcome"] == "completed"
        return
    workspace = browser.workspace
    workspace.current_instruction = "existing" if state == "busy" else None
    workspace.agent.shell.uncertain = state == "uncertain"
    workspace.closing = state == "closing"
    status = workspace.status()
    listener = asyncio.Queue()
    workspace.listeners.add(listener)

    with browser.connect() as socket:
        assert socket.receive_json() == {"type": "snapshot"}
        socket.send_json(command)
        error = socket.receive_json()
        assert error["type"] == "error"
        if validation:
            assert validation in error["text"]
        elif state == "closing":
            assert "shutting down" in error["text"]
        elif state == "busy":
            assert "busy" in error["text"]
        elif not command.get("instruction", "").strip():
            assert error["text"] == "Enter an instruction."
        else:
            assert "uncertain" in error["text"]

        socket.send_json({"type": "run", "instruction": 123})
        assert socket.receive_json()["type"] == "error"

    assert workspace.status() == status
    assert workspace.task is None
    assert not workspace.recent_tasks
    assert listener.empty()
    assert workspace.listeners == {listener}
    assert not browser.executions


def test_browser_accepts_trimmed_instruction_at_limit(browser):
    with browser.connect() as socket:
        assert socket.receive_json() == {"type": "snapshot"}
        socket.send_json({"type": "run", "instruction": " " + "x" * 7998 + " "})
        assert socket.receive_json() == {"type": "instruction", "text": "x" * 7998}
    assert browser.executions == ["x" * 7998]
    assert browser.workspace.recent_tasks[-1]["outcome"] == "completed"
