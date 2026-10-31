import json
import pytest
import asyncio
from types import SimpleNamespace
from api.main import ExecutionRequest, create_app
from threading import Event
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect
from src.api.tests.helpers import execute


def test_libraries_discovers_extra_packages(monkeypatch):
    api_client = TestClient(create_app())
    packages = [
        SimpleNamespace(metadata={"Name": "z-extra"}, version="1.2.3"),
        SimpleNamespace(metadata={"Name": "Another"}, version="4.5.6"),
    ]
    monkeypatch.setattr("api.main.distributions", lambda: iter(packages))
    assert api_client.get("/libraries").json() == [{"name": "Another", "version": "4.5.6"}, {"name": "z-extra", "version": "1.2.3"}]


def test_execution_preserves_namespace_and_captures_output():
    api_client = TestClient(create_app())
    assert execute(api_client, "import math\nvalue = math.sqrt(16)")["exception"] is None
    result = execute(api_client, "print(value)\nimport sys\nprint('warning', file=sys.stderr)\ndone()")
    assert result == {"stdout": "4.0\n", "stderr": "warning\n", "exception": None, "done": True}
    assert execute(api_client, "pass") == {"stdout": "", "stderr": "", "exception": None, "done": False}


def test_invalid_request_is_rejected():
    api_client = TestClient(create_app())
    assert api_client.post("/execute/stream", json={"command": "print(42)"}).status_code == 422


def test_synchronous_execution_endpoint_is_retired(tmp_path):
    app = create_app()
    app.routes[-1].app.directory = str(tmp_path)
    app.routes[-1].app.all_directories = [str(tmp_path)]
    api_client = TestClient(app)
    assert api_client.post("/", json={"code": "raise RuntimeError('must not execute')"}).status_code == 405
    assert "/" not in api_client.get("/openapi.json").json()["paths"]


def test_stream_requests_are_serialized():
    api_client = TestClient(create_app())
    execute(api_client, "value = 0\nimport time")

    def increment(_):
        return execute(api_client, "previous = value\ntime.sleep(0.01)\nvalue = previous + 1")

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(increment, range(8)))
    assert all(result["exception"] is None for result in results)
    assert execute(api_client, "print(value)")["stdout"] == "8\n"


def test_stream_only_confirms_execution_while_widgets_publish_directly():
    api_client = TestClient(create_app())
    response = api_client.post(
        "/execute/stream",
        json={
            "code": (
                "print('before')\n"
                "chart = ui.plot([{'type': 'bar', 'x': ['A'], 'y': [1]}], surface_id='sales')\n"
                "chart.pin()\n"
                "print('after')\n"
                "done()"
            )
        },
    )
    events = [json.loads(line) for line in response.text.splitlines()]

    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert [event["type"] for event in events] == ["result", "execution_finished"]
    assert events[-2] == {"type": "result", "stdout": "before\nafter\n", "stderr": "", "exception": None, "done": True}
    with api_client.websocket_connect("/ui/ws", headers={"origin": "http://localhost:8000"}) as socket:
        snapshot = socket.receive_json()
        assert snapshot["events"][0]["messages"][0]["createSurface"]["surfaceId"] == "sales"
        assert snapshot["placements"] == {"sales": {"pinned": True}}
    assert execute(api_client, "print(chart.surface_id)")["stdout"] == "sales\n"


def test_stream_reports_partial_output_and_python_failure():
    # Arrange
    api_client = TestClient(create_app())

    # Act
    response = api_client.post(
        "/execute/stream",
        json={
            "code": (
                "value = 42\n"
                "print('before')\n"
                "ui.plot([{'type': 'bar', 'x': ['A'], 'y': [1]}], surface_id='notice')\n"
                "done()\n1 / 0\nvalue = 99"
            )
        },
    )
    events = [json.loads(line) for line in response.text.splitlines()]

    # Assert
    assert response.status_code == 200
    assert [event["type"] for event in events] == ["result", "execution_finished"]
    assert api_client.get("/ui/context").json()["charts"][0]["surface_id"] == "notice"
    assert events[-2]["stdout"] == "before\n"
    assert "ZeroDivisionError" in events[-2]["exception"]
    assert events[-2]["done"] is False
    assert events[-1] == {"type": "execution_finished"}
    subsequent = execute(api_client, "print(value)")
    assert subsequent["stdout"] == "42\n"
    for source, error in (("if :", "SyntaxError"), ("raise SystemExit(1)", "SystemExit"), ("raise KeyboardInterrupt", "KeyboardInterrupt")):
        assert error in execute(api_client, source)["exception"]
    assert api_client.get("/health").status_code == 200


async def open_stream(app, code="pass"):
    tasks = asyncio.all_tasks()
    endpoint = next(route.endpoint for route in app.routes if route.path == "/execute/stream")
    response = await endpoint(ExecutionRequest(code=code))
    workers = asyncio.all_tasks() - tasks
    assert len(workers) == 1
    return response, workers


def test_stream_worker_failure_still_emits_terminal_event(monkeypatch):
    app = create_app()

    def execute(_shell, _code):
        raise RuntimeError("worker failed")

    monkeypatch.setattr("api.main.PythonShell.execute", execute)

    async def check():
        response, workers = await open_stream(app)
        errors = await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True), timeout=2)
        assert isinstance(errors[0], RuntimeError)
        assert str(errors[0]) == "worker failed"
        assert [json.loads(line) async for line in response.body_iterator] == [{"type": "execution_finished"}]

    asyncio.run(check())


def test_stream_disconnect_does_not_cancel_or_resubmit_execution(monkeypatch):
    app = create_app()
    disconnected = Event()
    calls = []

    def execute(_shell, code):
        calls.append(code)
        assert disconnected.wait(timeout=2)
        calls.append("completed")
        return {"stdout": "complete", "ui": []}

    monkeypatch.setattr("api.main.PythonShell.execute", execute)

    async def check():
        response, workers = await open_stream(app)
        sent = []

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.start":
                raise OSError("connection closed")

        async def receive():
            pytest.fail("ASGI 2.4 detects disconnects through send")

        try:
            with pytest.raises(ClientDisconnect):
                await asyncio.wait_for(response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send), timeout=2)
        finally:
            disconnected.set()
        await asyncio.wait_for(asyncio.gather(*workers), timeout=2)
        assert sent[-1]["type"] == "http.response.start"
        events = [json.loads(line) async for line in response.body_iterator]
        assert events == [{"type": "result", "stdout": "complete"}, {"type": "execution_finished"}]

    asyncio.run(check())
    assert calls == ["pass", "completed"]
