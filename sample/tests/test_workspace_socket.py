import pytest
import asyncio
import importlib
from types import SimpleNamespace
from contextlib import contextmanager
from fastapi.testclient import TestClient
from sample.src.workspace import Workspace, Subscription


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


@pytest.mark.parametrize(
    ("command", "validation"),
    [
        ({"type": "run", "instruction": "x" * 8001}, "8000"),
        ({"type": "run", "instruction": 123}, "valid string"),
        ({"type": "run", "instruction": None}, "valid string"),
        ({"type": "run", "instruction": " \n\t"}, None),
        ({"type": "run"}, None),
        ({"type": "run", "instruction": "calculate", "extra": True}, "Extra inputs"),
        ({"type": "other", "instruction": "calculate"}, "'run'"),
    ],
)
def test_browser_validates_shared_workspace_submission(browser, command, validation):
    workspace = browser.workspace
    status = workspace.status()
    listener = Subscription(workspace)
    listener.queue.get_nowait()

    with browser.connect() as socket:
        assert socket.receive_json() == {"type": "snapshot"}
        socket.send_json(command)
        error = socket.receive_json()
        assert error["type"] == "error"
        if validation:
            assert validation in error["text"]
        else:
            assert error["text"] == "Enter an instruction."

        socket.send_json({"type": "run", "instruction": 123})
        assert socket.receive_json()["type"] == "error"

    assert workspace.status() == status
    assert workspace.task is None
    assert not workspace.recent_tasks
    assert listener.queue.empty()
    assert workspace.listeners == {listener}
    assert not browser.executions


def test_browser_accepts_trimmed_instruction_at_limit(browser):
    with browser.connect() as socket:
        assert socket.receive_json() == {"type": "snapshot"}
        socket.send_json({"type": "run", "instruction": " " + "x" * 7998 + " "})
        assert socket.receive_json() == {"type": "instruction", "text": "x" * 7998}
    assert browser.executions == ["x" * 7998]
    assert browser.workspace.recent_tasks[-1]["outcome"] == "completed"


def test_command_schema_and_default_share_instruction_constraint(browser):
    main = importlib.import_module("sample.main")
    instruction = main.Command.model_json_schema()["properties"]["instruction"]
    assert instruction["type"] == "string"
    assert instruction["maxLength"] == 8000
    assert instruction["default"] == ""
    assert main.Command(type="run").instruction == ""


def test_browser_busy_rejection_preserves_shared_work(browser):
    workspace = browser.workspace
    workspace.current_instruction = "existing"
    status = workspace.status()
    with browser.connect() as socket:
        assert socket.receive_json() == {"type": "snapshot"}
        socket.send_json({"type": "run", "instruction": "calculate"})
        error = socket.receive_json()
        assert error["type"] == "error"
        assert "busy" in error["text"]
    assert workspace.status() == status
    assert workspace.task is None
    assert not browser.executions
