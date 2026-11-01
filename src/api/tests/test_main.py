import json
import pytest
import asyncio
import logging
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
    result = execute(api_client, "print(value)\nimport sys\nprint('warning', file=sys.stderr)\ndone('Calculated 4.0')")
    assert result == {"stdout": "4.0\n", "stderr": "warning\n", "exception": None, "done": "Calculated 4.0"}
    assert execute(api_client, "pass") == {"stdout": "", "stderr": "", "exception": None, "done": None}


def test_invalid_request_is_rejected():
    api_client = TestClient(create_app())
    assert api_client.post("/execute/stream", json={"command": "print(42)"}).status_code == 422


def test_interactive_stream_buffers_lines_and_reset_clears_pending_input():
    api_client = TestClient(create_app())

    def submit(line):
        response = api_client.post("/execute/stream", json={"code": line, "interactive": True})
        assert response.status_code == 200
        result, finished = [json.loads(event) for event in response.text.splitlines()]
        assert finished == {"type": "execution_finished"}
        return result

    assert submit("for i in range(2):")["more_input"] is True
    assert submit("    print(i)")["more_input"] is True
    assert submit("")["stdout"] == "0\n1\n"
    assert submit("if True:")["more_input"] is True
    assert api_client.post("/reset").status_code == 200
    assert submit("6 * 7")["stdout"] == "42\n"
    assert api_client.post("/execute/stream", json={"code": "x = 1\nx = 2", "interactive": True}).status_code == 422


def test_reset_preserves_capture_for_logging_handlers_retained_by_modules(monkeypatch):
    api_client = TestClient(create_app())
    logger = logging.getLogger("reset_capture")
    monkeypatch.setattr(logger, "propagate", False)
    try:
        result = execute(
            api_client,
            "import logging, sys\n"
            "logger = logging.getLogger('reset_capture')\n"
            "logger.addHandler(logging.StreamHandler(sys.stderr))\n"
            "logger.warning('before reset')",
        )
        assert result["stderr"] == "before reset\n"
        assert api_client.post("/reset").status_code == 200
        result = execute(api_client, "import logging\nlogging.getLogger('reset_capture').warning('after reset')")
        assert result["stderr"] == "after reset\n"
        assert result["exception"] is None
        assert "NameError" in execute(api_client, "logger")["exception"]
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()


def test_stream_requests_are_serialized():
    api_client = TestClient(create_app())
    execute(api_client, "value = 0\nimport time")

    def increment(_):
        return execute(api_client, "previous = value\ntime.sleep(0.01)\nvalue = previous + 1")

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(increment, range(8)))
    assert all(result["exception"] is None for result in results)
    assert execute(api_client, "print(value)")["stdout"] == "8\n"


def test_stream_reports_partial_output_and_python_failure():
    # Arrange
    api_client = TestClient(create_app())

    # Act
    result = execute(
        api_client,
        "value = 42\nprint('before')\nui.plot([{'type': 'bar', 'x': ['A'], 'y': [1]}], surface_id='notice')\ndone('Updated chart')\n1 / 0\nvalue = 99",
    )

    # Assert
    assert api_client.get("/ui/context").json()["charts"][0]["surface_id"] == "notice"
    assert result["stdout"] == "before\n"
    assert "ZeroDivisionError" in result["exception"]
    assert result["done"] is None
    subsequent = execute(api_client, "print(value)")
    assert subsequent["stdout"] == "42\n"
    for source, error in (("if :", "SyntaxError"), ("raise SystemExit(1)", "SystemExit"), ("raise KeyboardInterrupt", "KeyboardInterrupt")):
        assert error in execute(api_client, source)["exception"]
    assert api_client.get("/health").status_code == 200


async def open_stream(app):
    tasks = asyncio.all_tasks()
    endpoint = next(route.endpoint for route in app.routes if route.path == "/execute/stream")
    response = await endpoint(ExecutionRequest(code="pass"))
    workers = asyncio.all_tasks() - tasks
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
        return {"stdout": "complete"}

    monkeypatch.setattr("api.main.PythonShell.execute", execute)

    async def check():
        response, workers = await open_stream(app)

        async def send(message):
            assert message["type"] == "http.response.start"
            raise OSError("connection closed")

        async def receive():
            pytest.fail("ASGI 2.4 detects disconnects through send")

        try:
            with pytest.raises(ClientDisconnect):
                await asyncio.wait_for(response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send), timeout=2)
        finally:
            disconnected.set()
        await asyncio.wait_for(asyncio.gather(*workers), timeout=2)
        events = [json.loads(line) async for line in response.body_iterator]
        assert events == [{"type": "result", "stdout": "complete"}, {"type": "execution_finished"}]

    asyncio.run(check())
    assert calls == ["pass", "completed"]
