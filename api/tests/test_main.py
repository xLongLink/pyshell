import json
import pytest
import asyncio
from main import ExecutionRequest, create_app
from queue import Queue
from types import SimpleNamespace
from threading import Event
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from starlette.requests import ClientDisconnect


def test_health_without_model_credentials(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_libraries_reports_installed_versions(api_client):
    response = api_client.get("/libraries")
    assert response.status_code == 200
    libraries = response.json()
    assert {"name": "fastapi", "version": version("fastapi")} in libraries
    assert all(library["name"] and library["version"] for library in libraries)
    assert libraries == sorted(libraries, key=lambda library: library["name"].casefold())


def test_libraries_discovers_extra_packages(api_client, monkeypatch):
    packages = [
        SimpleNamespace(metadata={"Name": "z-extra"}, version="1.2.3"),
        SimpleNamespace(metadata={"Name": "Another"}, version="4.5.6"),
    ]
    monkeypatch.setattr("main.distributions", lambda: iter(packages))
    assert api_client.get("/libraries").json() == [{"name": "Another", "version": "4.5.6"}, {"name": "z-extra", "version": "1.2.3"}]


def test_execution_preserves_namespace_and_captures_output(api_client):
    assert api_client.post("/", json={"code": "import math\nvalue = math.sqrt(16)"}).status_code == 200
    response = api_client.post("/", json={"code": "print(value)\nimport sys\nprint('warning', file=sys.stderr)\ndone()"})
    assert response.json() == {"stdout": "4.0\n", "stderr": "warning\n", "exception": None, "done": True, "ui": []}


def test_reset_clears_namespace(api_client):
    api_client.post("/", json={"code": "value = 42"})
    assert api_client.post("/reset").status_code == 200
    assert "NameError" in api_client.post("/", json={"code": "print(value)"}).json()["exception"]


def test_errors_preserve_prior_updates_and_do_not_kill_server(api_client):
    result = api_client.post("/", json={"code": "value = 42\nprint('before')\n1 / 0\nvalue = 99"}).json()
    assert result["stdout"] == "before\n"
    assert "ZeroDivisionError" in result["exception"]
    assert api_client.post("/", json={"code": "print(value)"}).json()["stdout"] == "42\n"
    for source in ("if :", "raise SystemExit(1)", "raise KeyboardInterrupt"):
        assert api_client.post("/", json={"code": source}).json()["exception"]
    assert api_client.get("/health").status_code == 200


def test_output_and_done_are_per_request(api_client):
    api_client.post("/", json={"code": "print('old')\ndone()"})
    assert api_client.post("/", json={"code": "pass"}).json() == {"stdout": "", "stderr": "", "exception": None, "done": False, "ui": []}


def test_invalid_request_is_rejected(api_client):
    assert api_client.post("/", json={"command": "print(42)"}).status_code == 422


def test_requests_are_serialized(api_client):
    api_client.post("/", json={"code": "value = 0\nimport time"})

    def increment(_):
        return api_client.post("/", json={"code": "previous = value\ntime.sleep(0.01)\nvalue = previous + 1"})

    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(increment, range(8)))
    assert all(response.status_code == 200 for response in responses)
    assert api_client.post("/", json={"code": "print(value)"}).json()["stdout"] == "8\n"


def test_stream_preserves_output_widget_and_placement_order(api_client):
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

    # Content and placement must stay interleaved with Python output.
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert [event["type"] for event in events] == [
        "stdout_delta",
        "stdout_delta",
        "a2ui",
        "widget",
        "stdout_delta",
        "stdout_delta",
        "result",
        "execution_finished",
    ]
    assert events[2]["messages"][0]["createSurface"]["surfaceId"] == "sales"
    assert events[3] == {"type": "widget", "surface_id": "sales", "action": "pin"}
    assert events[-2] == {"type": "result", "stdout": "before\nafter\n", "stderr": "", "exception": None, "done": True}
    assert api_client.post("/", json={"code": "print(chart.surface_id)"}).json()["stdout"] == "sales\n"


def test_stream_reports_partial_output_and_python_failure(api_client):
    response = api_client.post("/execute/stream", json={"code": "value = 42\nprint('before')\ndone()\n1 / 0"})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert "".join(event["text"] for event in events if event["type"] == "stdout_delta") == "before\n"
    assert "ZeroDivisionError" in events[-2]["exception"]
    assert events[-2]["done"] is False
    assert events[-1] == {"type": "execution_finished"}
    assert api_client.post("/", json={"code": "print(value)"}).json()["stdout"] == "42\n"


@pytest.fixture
def stream_app(monkeypatch):
    queues = []

    class BoundedWaitQueue(Queue):
        def put(self, item, block=True, timeout=None):
            return super().put(item, block=block, timeout=0.05)

        def get(self, block=True, timeout=None):
            return super().get(block=block, timeout=0.2)

    def queue_factory(maxsize):
        queue = BoundedWaitQueue(maxsize=maxsize)
        queues.append(queue)
        return queue

    monkeypatch.setattr("main.Queue", queue_factory)
    return create_app(), queues


async def open_stream(app, code="pass"):
    tasks = asyncio.all_tasks()
    endpoint = next(route.endpoint for route in app.routes if route.path == "/execute/stream")
    response = await endpoint(ExecutionRequest(code=code))
    workers = asyncio.all_tasks() - tasks
    assert len(workers) == 1
    return response, workers


def test_stream_terminal_event_fits_at_queue_capacity(stream_app, monkeypatch):
    app, queues = stream_app
    calls = []

    def execute(_shell, code, on_event):
        calls.append(code)
        for index in range(254):
            on_event({"type": "stdout_delta", "text": str(index)})
        return {"stdout": "complete", "ui": []}

    monkeypatch.setattr("main.PythonShell.execute", execute)

    async def check():
        response, workers = await open_stream(app)
        await asyncio.wait_for(asyncio.gather(*workers), timeout=2)
        assert queues[0].qsize() == queues[0].maxsize == 256
        events = [json.loads(line) async for line in response.body_iterator]
        assert [event["text"] for event in events[:-2]] == [str(index) for index in range(254)]
        assert events[-2:] == [{"type": "result", "stdout": "complete"}, {"type": "execution_finished"}]
        assert queues[0].empty()

    asyncio.run(check())
    assert calls == ["pass"]


def test_stream_saturated_queue_drains_without_resubmitting(stream_app, monkeypatch):
    app, queues = stream_app
    calls = []

    def execute(_shell, code, on_event):
        calls.append(code)
        for index in range(300):
            on_event({"type": "stdout_delta", "text": str(index)})
        calls.append("completed")
        return {"stdout": "complete", "ui": []}

    monkeypatch.setattr("main.PythonShell.execute", execute)

    async def check():
        response, workers = await open_stream(app)
        await asyncio.wait_for(asyncio.gather(*workers), timeout=2)
        events = [json.loads(line) async for line in response.body_iterator]
        assert events == [{"type": "stdout_delta", "text": str(index)} for index in range(256)]
        assert queues[0].empty()

    asyncio.run(check())
    assert calls == ["pass", "completed"]


def test_stream_worker_failure_still_emits_terminal_event(stream_app, monkeypatch):
    app, _queues = stream_app

    def execute(_shell, _code, on_event):
        on_event({"type": "stdout_delta", "text": "before"})
        raise RuntimeError("worker failed")

    monkeypatch.setattr("main.PythonShell.execute", execute)

    async def check():
        response, workers = await open_stream(app)
        errors = await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True), timeout=2)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert str(errors[0]) == "worker failed"
        assert [json.loads(line) async for line in response.body_iterator] == [
            {"type": "stdout_delta", "text": "before"},
            {"type": "execution_finished"},
        ]

    asyncio.run(check())


def test_stream_disconnect_does_not_cancel_or_resubmit_execution(stream_app, monkeypatch):
    app, queues = stream_app
    disconnected = Event()
    calls = []

    def execute(_shell, code, on_event):
        calls.append(code)
        on_event({"type": "stdout_delta", "text": "before"})
        assert disconnected.wait(timeout=2)
        for index in range(300):
            on_event({"type": "stdout_delta", "text": str(index)})
        calls.append("completed")
        return {"stdout": "complete", "ui": []}

    monkeypatch.setattr("main.PythonShell.execute", execute)

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
        assert json.loads(sent[-1]["body"]) == {"type": "stdout_delta", "text": "before"}
        assert queues[0].qsize() == 256
        events = [json.loads(line) async for line in response.body_iterator]
        assert events == [{"type": "stdout_delta", "text": str(index)} for index in range(256)]
        assert queues[0].empty()

    asyncio.run(check())
    assert calls == ["pass", "completed"]


def test_stream_and_regular_requests_share_execution_lock(stream_app, monkeypatch):
    app, _queues = stream_app
    entered = Event()
    release = Event()
    calls = []

    def execute(_shell, code, on_event=None):
        calls.append(code)
        if code == "stream":
            entered.set()
            assert release.wait(timeout=2)
        return {"stdout": code, "ui": []}

    monkeypatch.setattr("main.PythonShell.execute", execute)

    async def check():
        response, workers = await open_stream(app, "stream")
        assert await asyncio.to_thread(entered.wait, 2)
        endpoint = next(route.endpoint for route in app.routes if route.path == "/")
        submitted = Event()

        def regular_request():
            submitted.set()
            return endpoint(ExecutionRequest(code="regular"))

        regular = asyncio.create_task(asyncio.to_thread(regular_request))
        try:
            assert await asyncio.to_thread(submitted.wait, 2)
            assert calls == ["stream"]
            assert not regular.done()
        finally:
            release.set()
        await asyncio.wait_for(asyncio.gather(*workers, regular), timeout=2)
        assert [json.loads(line) async for line in response.body_iterator] == [
            {"type": "result", "stdout": "stream"},
            {"type": "execution_finished"},
        ]

    asyncio.run(check())
    assert calls == ["stream", "regular"]
