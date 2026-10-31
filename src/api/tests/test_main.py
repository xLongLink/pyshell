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


def test_health_without_model_credentials():
    api_client = TestClient(create_app(), headers={"authorization": "Bearer test-token"})
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_libraries_discovers_extra_packages(monkeypatch):
    api_client = TestClient(create_app(), headers={"authorization": "Bearer test-token"})
    packages = [
        SimpleNamespace(metadata={"Name": "z-extra"}, version="1.2.3"),
        SimpleNamespace(metadata={"Name": "Another"}, version="4.5.6"),
    ]
    monkeypatch.setattr("api.main.distributions", lambda: iter(packages))
    assert api_client.get("/libraries").json() == [{"name": "Another", "version": "4.5.6"}, {"name": "z-extra", "version": "1.2.3"}]


def test_execution_preserves_namespace_and_captures_output():
    api_client = TestClient(create_app(), headers={"authorization": "Bearer test-token"})
    assert execute(api_client, "import math\nvalue = math.sqrt(16)")["exception"] is None
    result = execute(api_client, "print(value)\nimport sys\nprint('warning', file=sys.stderr)\ndone()")
    assert result == {"stdout": "4.0\n", "stderr": "warning\n", "exception": None, "done": True, "ui": []}
    assert execute(api_client, "pass") == {"stdout": "", "stderr": "", "exception": None, "done": False, "ui": []}


def test_errors_preserve_prior_updates_and_do_not_kill_server():
    api_client = TestClient(create_app(), headers={"authorization": "Bearer test-token"})
    result = execute(api_client, "value = 42\nprint('before')\n1 / 0\nvalue = 99")
    assert result["stdout"] == "before\n"
    assert "ZeroDivisionError" in result["exception"]
    assert execute(api_client, "print(value)")["stdout"] == "42\n"
    for source in ("if :", "raise SystemExit(1)", "raise KeyboardInterrupt"):
        assert execute(api_client, source)["exception"]
    assert api_client.get("/health").status_code == 200


def test_invalid_request_is_rejected():
    api_client = TestClient(create_app(), headers={"authorization": "Bearer test-token"})
    assert api_client.post("/execute/stream", json={"command": "print(42)"}).status_code == 422


def test_synchronous_execution_endpoint_is_retired(tmp_path):
    app = create_app()
    app.routes[-1].app.directory = str(tmp_path)
    app.routes[-1].app.all_directories = [str(tmp_path)]
    api_client = TestClient(app, headers={"authorization": "Bearer test-token"})
    assert api_client.post("/", json={"code": "raise RuntimeError('must not execute')"}).status_code == 405
    assert "/" not in api_client.get("/openapi.json").json()["paths"]


def test_stream_requests_are_serialized():
    api_client = TestClient(create_app(), headers={"authorization": "Bearer test-token"})
    execute(api_client, "value = 0\nimport time")

    def increment(_):
        return execute(api_client, "previous = value\ntime.sleep(0.01)\nvalue = previous + 1")

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(increment, range(8)))
    assert all(result["exception"] is None for result in results)
    assert execute(api_client, "print(value)")["stdout"] == "8\n"


def test_stream_preserves_widget_placement_and_final_output_order():
    api_client = TestClient(create_app(), headers={"authorization": "Bearer test-token"})
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
    assert [event["type"] for event in events] == ["a2ui", "widget", "result", "execution_finished"]
    assert events[0]["messages"][0]["createSurface"]["surfaceId"] == "sales"
    assert events[1] == {"type": "widget", "surface_id": "sales", "action": "pin"}
    assert events[-2] == {"type": "result", "stdout": "before\nafter\n", "stderr": "", "exception": None, "done": True}
    assert execute(api_client, "print(chart.surface_id)")["stdout"] == "sales\n"


def test_stream_reports_partial_output_and_python_failure():
    api_client = TestClient(create_app(), headers={"authorization": "Bearer test-token"})
    response = api_client.post("/execute/stream", json={"code": "value = 42\nprint('before')\ndone()\n1 / 0"})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == ["result", "execution_finished"]
    assert events[-2]["stdout"] == "before\n"
    assert "ZeroDivisionError" in events[-2]["exception"]
    assert events[-2]["done"] is False
    assert events[-1] == {"type": "execution_finished"}
    assert execute(api_client, "print(value)")["stdout"] == "42\n"


async def open_stream(app, code="pass"):
    tasks = asyncio.all_tasks()
    endpoint = next(route.endpoint for route in app.routes if route.path == "/execute/stream")
    response = await endpoint(ExecutionRequest(code=code))
    workers = asyncio.all_tasks() - tasks
    assert len(workers) == 1
    return response, workers


def test_stream_saturated_queue_drains_without_resubmitting(monkeypatch):
    app = create_app()
    calls = []

    def execute(_shell, code, on_event):
        calls.append(code)
        for index in range(300):
            on_event({"type": "widget", "surface_id": str(index), "action": "focus"})
        calls.append("completed")
        return {"stdout": "complete", "ui": []}

    monkeypatch.setattr("api.main.PythonShell.execute", execute)

    async def check():
        response, workers = await open_stream(app)
        await asyncio.wait_for(asyncio.gather(*workers), timeout=2)
        events = [json.loads(line) async for line in response.body_iterator]
        assert events == [{"type": "widget", "surface_id": str(index), "action": "focus"} for index in range(256)]

    asyncio.run(check())
    assert calls == ["pass", "completed"]


def test_stream_worker_failure_still_emits_terminal_event(monkeypatch):
    app = create_app()

    def execute(_shell, _code, on_event):
        on_event({"type": "widget", "surface_id": "before", "action": "pin"})
        raise RuntimeError("worker failed")

    monkeypatch.setattr("api.main.PythonShell.execute", execute)

    async def check():
        response, workers = await open_stream(app)
        errors = await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True), timeout=2)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert str(errors[0]) == "worker failed"
        assert [json.loads(line) async for line in response.body_iterator] == [
            {"type": "widget", "surface_id": "before", "action": "pin"},
            {"type": "execution_finished"},
        ]

    asyncio.run(check())


def test_stream_disconnect_does_not_cancel_or_resubmit_execution(monkeypatch):
    app = create_app()
    disconnected = Event()
    calls = []

    def execute(_shell, code, on_event):
        calls.append(code)
        on_event({"type": "widget", "surface_id": "before", "action": "pin"})
        assert disconnected.wait(timeout=2)
        for index in range(300):
            on_event({"type": "widget", "surface_id": str(index), "action": "focus"})
        calls.append("completed")
        return {"stdout": "complete", "ui": []}

    monkeypatch.setattr("api.main.PythonShell.execute", execute)

    async def check():
        response, workers = await open_stream(app)
        sent = []

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body":
                raise OSError("connection closed")

        async def receive():
            pytest.fail("ASGI 2.4 detects disconnects through send")

        try:
            with pytest.raises(ClientDisconnect):
                await asyncio.wait_for(response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send), timeout=2)
        finally:
            disconnected.set()
        await asyncio.wait_for(asyncio.gather(*workers), timeout=2)
        assert json.loads(sent[-1]["body"]) == {"type": "widget", "surface_id": "before", "action": "pin"}
        events = [json.loads(line) async for line in response.body_iterator]
        assert events == [{"type": "widget", "surface_id": str(index), "action": "focus"} for index in range(256)]

    asyncio.run(check())
    assert calls == ["pass", "completed"]
