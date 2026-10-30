import json
import httpx
import pytest
import asyncio
import threading
from web import main as server
from types import SimpleNamespace
from web.main import app
from unittest.mock import AsyncMock
from web.src.workspace import Command
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


@pytest.fixture
def browser_api(monkeypatch, model_for, remote_shell, agent_settings):
    requests = []
    model = model_for(["print('browser result')\ndone()"], ["print('next result')\ndone()"], requests=requests)
    monkeypatch.setattr(server, "settings", agent_settings)
    monkeypatch.setattr(server, "OpenAIResponsesModel", lambda *args, **kwargs: model)
    monkeypatch.setattr(server, "AsyncClient", lambda **kwargs: remote_shell.client)
    monkeypatch.setattr(server, "RemoteShell", lambda client: remote_shell)
    with TestClient(app) as client:
        yield client, requests


def receive_until_idle(socket):
    events = []
    while True:
        event = socket.receive_json()
        events.append(event)
        if event["type"] == "state" and not event["busy"]:
            return events


@pytest.fixture
def lifespan_resources(monkeypatch, model_for, remote_shell, agent_settings):
    provider = SimpleNamespace(client=SimpleNamespace(close=AsyncMock()))
    monkeypatch.setattr(server, "settings", agent_settings)
    monkeypatch.setattr(server, "OpenAIProvider", lambda **kwargs: provider)
    monkeypatch.setattr(server, "AsyncClient", lambda **kwargs: remote_shell.client)
    monkeypatch.setattr(server, "OpenAIResponsesModel", lambda *args, **kwargs: model_for(["done()"]))
    monkeypatch.setattr(server, "OpenAIRealtimeModel", lambda *args, **kwargs: object())
    return remote_shell.client, provider.client


def test_lifespan_drains_accepted_work_before_closing_transports(lifespan_resources):
    client, provider = lifespan_resources

    async def run():
        started = asyncio.Event()
        exiting = asyncio.Event()
        release = asyncio.Event()

        async def execute(instruction, emit):
            started.set()
            await release.wait()
            assert not client.is_closed
            provider.close.assert_not_awaited()
            return "Confirmed before shutdown"

        async def application():
            async with server.lifespan(app):
                workspace = app.state.workspace
                workspace.agent.arun = execute
                workspace.start(Command(type="run", instruction="Finish accepted work"))
                await started.wait()
                exiting.set()

        owner = asyncio.create_task(application())
        try:
            await asyncio.wait_for(exiting.wait(), 1)
            workspace = app.state.workspace
            assert workspace.closing
            assert not owner.done()
            assert not client.is_closed
            with pytest.raises(RuntimeError, match="shutting down"):
                workspace.start(Command(type="run", instruction="Do not accept"))
        finally:
            release.set()
            await asyncio.wait_for(owner, 1)
        assert client.is_closed
        provider.close.assert_awaited_once_with()
        assert workspace.status()["latest_outcome"]["result"] == "Confirmed before shutdown"
        assert workspace.task.result() is None

    asyncio.run(run())


def test_lifespan_releases_transports_when_workspace_shutdown_fails(lifespan_resources, monkeypatch):
    client, provider = lifespan_resources

    async def run():
        with pytest.raises(RuntimeError, match="Shutdown failed"):
            async with server.lifespan(app):
                monkeypatch.setattr(app.state.workspace, "close", AsyncMock(side_effect=RuntimeError("Shutdown failed")))
        assert client.is_closed
        provider.close.assert_awaited_once_with()

    asyncio.run(run())


def test_lifespan_releases_transports_when_initialization_fails(lifespan_resources, monkeypatch):
    client, provider = lifespan_resources

    def fail(*args, **kwargs):
        raise RuntimeError("Initialization failed")

    monkeypatch.setattr(server, "OpenAIRealtimeModel", fail)

    async def run():
        with pytest.raises(RuntimeError, match="Initialization failed"):
            async with server.lifespan(app):
                pytest.fail("Failed initialization must not enter the application")
        assert client.is_closed
        provider.close.assert_awaited_once_with()

    asyncio.run(run())


def test_web_server_serves_website_and_assets(browser_api, monkeypatch, tmp_path):
    client, _ = browser_api
    (tmp_path / "index.html").write_text("<!doctype html><title>PyShell</title>")
    (tmp_path / "app.js").write_text("console.log('pyshell')")
    monkeypatch.setattr(server.frontend, "directory", str(tmp_path))
    monkeypatch.setattr(server.frontend, "all_directories", [str(tmp_path)])
    response = client.get("/")

    # The HTML mount must serve real assets without masking missing files.
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "PyShell" in response.text
    assert client.get("/app.js").text == "console.log('pyshell')"
    assert client.get("/missing.js").status_code == 404


def test_geography_proxy_serves_backend_data_and_cache_headers_without_execution(browser_api, api_client):
    client, requests = browser_api
    workspace = client.app.state.workspace
    workspace.agent.shell.uncertain = True
    response = client.get("/geographies/CHE")
    upstream = api_client.get("/geographies/CHE")
    assert response.status_code == 200
    assert response.content == upstream.content
    assert response.headers["content-type"] == "application/geo+json"
    for name in ("cache-control", "etag", "last-modified"):
        assert response.headers[name] == upstream.headers[name]
    assert requests == []
    assert workspace.surfaces == {}
    assert workspace.events == []
    assert workspace.agent.shell.uncertain is True


@pytest.mark.parametrize("geography", ["ZZZ", "sources", "USA.json", "__proto__"])
def test_geography_proxy_rejects_unknown_assets(browser_api, geography):
    client, _ = browser_api
    response = client.get(f"/geographies/{geography}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Unsupported geography"}


@pytest.mark.parametrize("status", [None, 500])
def test_geography_proxy_reports_backend_failure_without_leaking_details(browser_api, monkeypatch, status):
    client, _ = browser_api

    async def fail(geography):
        request = httpx.Request("GET", "http://execution-private/geographies/CHE")
        if status is None:
            raise httpx.ConnectError("private backend address", request=request)
        httpx.Response(status, request=request).raise_for_status()

    monkeypatch.setattr(client.app.state.workspace.agent.shell, "geography", fail)
    response = client.get("/geographies/CHE")
    assert response.status_code == 502
    assert response.json() == {"detail": "Map data is unavailable"}


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
        assert not any(event["type"] in {"code_delta", "done", "code_start", "execution_start"} for event in events)
        assert {"type": "result", "exception": None} in events
        assert "".join(event["text"] for event in events if event["type"] == "stdout_delta") == "browser result\n"
        assert events[-1] == {"type": "state", "busy": False}
        assert all("run_id" not in event for event in events)
        feedback = json.loads(client.app.state.workspace.agent.pending_feedback)
        assert feedback["stdout"] == "browser result\n"
        assert feedback["done"] is True

    # Reconnection restores history without another model request.
    with client.websocket_connect("/ws", headers={"origin": "http://127.0.0.1:5180"}) as socket:
        snapshot = socket.receive_json()
        assert snapshot["busy"] is False
        assert {"type": "result", "exception": None} in snapshot["events"]
        assert "".join(event["text"] for event in snapshot["events"] if event["type"] == "stdout_delta") == "browser result\n"
        assert "model" not in snapshot
        assert "run_id" not in snapshot
        assert len(requests) == 1


def test_browser_rejects_reset_without_changing_shared_state(browser_api, api_client):
    client, _ = browser_api
    assert api_client.post("/", json={"code": "value = 42"}).json()["exception"] is None
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:5180"}) as socket:
        socket.receive_json()
        socket.send_json({"type": "reset"})
        events = receive_until_idle(socket)

        # Unsupported reset commands must not clear execution or agent state.
        assert events[0]["type"] == "error"
        assert not any(event["type"] == "reset" for event in events)
        assert api_client.post("/", json={"code": "print(value)"}).json()["stdout"] == "42\n"
        assert client.app.state.workspace.agent.message_history == []


def test_browser_reuses_the_provider_event_loop(browser_api, monkeypatch):
    client, requests = browser_api
    agent = client.app.state.workspace.agent
    run = agent.arun
    threads = []

    async def tracked_run(instruction, emit):
        threads.append((threading.get_ident(), asyncio.get_event_loop()))
        return await run(instruction, emit)

    monkeypatch.setattr(agent, "arun", tracked_run)
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:5180"}) as socket:
        socket.receive_json()

        # Consecutive turns must reuse the application's event loop.
        for instruction in ("first", "second"):
            socket.send_json({"type": "run", "instruction": instruction})
            assert {"type": "result", "exception": None} in receive_until_idle(socket)
    assert len(requests) == 2
    assert len(threads) == 2
    assert threads[0] == threads[1]


def test_browser_reports_model_failure_without_executing(browser_api, model_for, remote_shell, api_client):
    client, _ = browser_api
    client.app.state.workspace.agent = server.Agent(model_for(["value = 42\n", ConnectionError("stream interrupted")]), shell=remote_shell)
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:5180"}) as socket:
        socket.receive_json()
        socket.send_json({"type": "run", "instruction": "set value"})
        events = receive_until_idle(socket)
        assert any(event["type"] == "error" and "stream interrupted" in event["text"] for event in events)

        # An incomplete model stream must never reach execution.
        assert not any(event["type"] == "result" for event in events)
    assert "NameError" in api_client.post("/", json={"code": "print(value)"}).json()["exception"]


def test_python_widgets_stream_and_replay(browser_api, model_for, remote_shell):
    client, _ = browser_api
    client.app.state.workspace.agent = server.Agent(
        model_for(["ui.plot([{'type': 'bar', 'x': ['Apples'], 'y': [42]}], surface_id='sales')\nprint('Sales ready')\ndone()"]),
        shell=remote_shell,
    )

    # Widget content must be streamed and retained for reconnecting tabs.
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
        socket.receive_json()
        socket.send_json({"type": "run", "instruction": "show sales"})
        events = receive_until_idle(socket)
        ui = next(event for event in events if event["type"] == "a2ui")

        # The replayed payload must match the Python figure's serialized content.
        assert ui["messages"][1]["updateComponents"]["components"][0]["data"] == [{"type": "bar", "name": "", "x": ["Apples"], "y": [42]}]
        assert {"type": "result", "exception": None} in events
        assert "".join(event["text"] for event in events if event["type"] == "stdout_delta") == "Sales ready\n"
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
        assert ui in socket.receive_json()["events"]


def test_browser_rejects_untrusted_origin(browser_api):
    client, _ = browser_api
    with pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect("/ws", headers={"origin": "https://example.com"}):
            pass
    assert error.value.code == 1008


def test_python_placement_snapshot_survives_updates(browser_api, model_for, remote_shell):
    client, requests = browser_api
    model = model_for(
        ["chart = ui.plot([{'type': 'scatter', 'x': ['Jan'], 'y': [42]}], surface_id='sales')\nchart.pin()\nchart.focus()\ndone()"],
        ["chart.update(layout={'title': {'text': 'Updated revenue'}})\ndone()"],
        requests=requests,
    )
    client.app.state.workspace.agent = server.Agent(model, shell=remote_shell)

    # Content changes must retain the placement established by Python actions.
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
        socket.receive_json()
        socket.send_json({"type": "run", "instruction": "show sales"})
        events = receive_until_idle(socket)
        placements = [event for event in events if event["type"] == "placement"]
        assert placements[-1]["widgets"] == {"sales": {"pinned": False}}
        initial = placements[0]["widgets"]
        assert initial == {"sales": {"pinned": False}}

        # Placement snapshots must not alias later mutations or content updates.
        assert any(event["widgets"] == {"sales": {"pinned": True}} for event in placements)
        assert len(requests) == 1
        socket.send_json({"type": "run", "instruction": "update the title"})
        events = receive_until_idle(socket)
        updates = [message for event in events if event["type"] == "a2ui" for message in event["messages"]]
        assert len(updates) == 1
        assert updates[0]["updateComponents"]["components"][0]["layout"]["title"]["text"] == "Updated revenue"

    # A fresh tab sees the current placement and latest content.
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
        assert socket.receive_json()["placements"] == {"sales": {"pinned": False}}


def test_placement_history_is_not_retained_but_reconnect_and_live_events_are_current(browser_api, model_for, remote_shell):
    client, _ = browser_api
    workspace = client.app.state.workspace
    workspace.agent = server.Agent(
        model_for(
            ["chart = ui.plot([{'type': 'bar', 'x': ['A'], 'y': [1]}], surface_id='sales')\nchart.pin()\ndone()"],
            ["chart.unpin()\nchart.unfocus()\nchart.remove()\ndone()"],
        ),
        shell=remote_shell,
    )

    # The initial turn emits content before applying its placement command.
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
        socket.receive_json()
        socket.send_json({"type": "run", "instruction": "create and pin sales"})
        events = receive_until_idle(socket)
        assert [event["type"] for event in events if event["type"] in {"a2ui", "placement"}] == ["a2ui", "placement", "placement"]
        assert [event["widgets"] for event in events if event["type"] == "placement"] == [
            {"sales": {"pinned": False}},
            {"sales": {"pinned": True}},
        ]

    # Placement is sent live but kept out of retained transcript history.
    history = [event for event in events if event["type"] not in {"state", "placement", "a2ui"}]
    content = [event for event in events if event["type"] == "a2ui"]
    assert workspace.events == history
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
        snapshot = socket.receive_json()
        assert snapshot["events"] == history + content
        assert snapshot["placements"] == {"sales": {"pinned": True}}

        # Removal must preserve the ordering of preceding placement commands.
        socket.send_json({"type": "run", "instruction": "unpin, unfocus, and remove sales"})
        events = receive_until_idle(socket)
        assert [event["type"] for event in events if event["type"] in {"a2ui", "placement"}] == [
            "placement",
            "placement",
            "a2ui",
            "placement",
        ]
        assert [event["widgets"] for event in events if event["type"] == "placement"] == [
            {"sales": {"pinned": False}},
            {"sales": {"pinned": True}},
            {},
        ]

    # Reconnection uses the latest placement map, not obsolete placement events.
    history += [event for event in events if event["type"] not in {"state", "placement", "a2ui"}]
    assert workspace.events == history
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
        snapshot = socket.receive_json()
        assert snapshot["events"] == history
        assert snapshot["placements"] == {}


def test_control_order_is_preserved_when_a_named_widget_is_replaced(browser_api, model_for, remote_shell):
    client, _ = browser_api
    client.app.state.workspace.agent = server.Agent(
        model_for(
            [
                "first = ui.plot([{'type': 'bar', 'x': ['A'], 'y': [1]}], surface_id='notice')\nfirst.pin()\nfirst.remove()\nui.plot([{'type': 'scatter', 'x': ['A'], 'y': [2]}], surface_id='notice')\ndone()"
            ]
        ),
        shell=remote_shell,
    )

    # Replacing a named surface must not inherit its old pinned state.
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
        socket.receive_json()
        socket.send_json({"type": "run", "instruction": "replace the result"})
        events = receive_until_idle(socket)
        placements = [event["widgets"] for event in events if event["type"] == "placement"]
        assert any(item["notice"]["pinned"] for item in placements if "notice" in item)
        assert placements[-1] == {"notice": {"pinned": False}}


@pytest.mark.parametrize("surface_id", ["missing", "sales"])
def test_browser_widget_commands_are_rejected_without_starting_work(browser_api, surface_id):
    client, requests = browser_api
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
        socket.receive_json()
        socket.send_json({"type": "widget", "surface_id": surface_id, "action": "pin"})
        events = receive_until_idle(socket)
        assert events[0]["type"] == "error"
        assert "validation error" in events[0]["text"]
        assert requests == []


def test_focusing_another_widget_and_unfocusing_the_old_one_preserves_new_focus(browser_api, model_for, remote_shell):
    client, _ = browser_api
    client.app.state.workspace.agent = server.Agent(
        model_for(
            [
                "first = ui.plot([{'type': 'bar', 'x': ['A'], 'y': [1]}], surface_id='first')\nsecond = ui.plot([{'type': 'scatter', 'x': ['A'], 'y': [2]}], surface_id='second')\nfirst.focus()\nsecond.focus()\nfirst.unfocus()\ndone()"
            ]
        ),
        shell=remote_shell,
    )

    # Unfocusing the old widget must not displace the newly focused widget.
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
        socket.receive_json()
        socket.send_json({"type": "run", "instruction": "focus second"})
        events = receive_until_idle(socket)
        placements = [event["widgets"] for event in events if event["type"] == "placement"]
        assert placements[-1] == {"first": {"pinned": True}, "second": {"pinned": False}}


def test_python_placement_remains_available_during_agent_work(browser_api, model_for, remote_shell, monkeypatch):
    client, _ = browser_api
    agent = server.Agent(model_for(["ui.plot([{'type': 'bar', 'x': ['A'], 'y': [1]}], surface_id='notice')\ndone()"]), shell=remote_shell)
    client.app.state.workspace.agent = agent
    started = threading.Event()
    release = threading.Event()

    async def work(instruction, emit):
        emit({"type": "widget", "surface_id": "notice", "action": "pin"})
        started.set()
        assert await asyncio.to_thread(release.wait, 5)
        return "Finished"

    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
        socket.receive_json()
        socket.send_json({"type": "run", "instruction": "create a result"})
        receive_until_idle(socket)

        # Block the next run while checking that its placement event arrives live.
        monkeypatch.setattr(agent, "arun", work)
        socket.send_json({"type": "run", "instruction": "keep working"})
        assert socket.receive_json()["busy"] is True
        assert socket.receive_json()["type"] == "instruction"
        try:
            assert started.wait(timeout=5)
            assert socket.receive_json()["widgets"] == {"notice": {"pinned": True}}

        # Always release the worker so a failed assertion cannot hang teardown.
        finally:
            release.set()
        assert receive_until_idle(socket)[-1]["busy"] is False
        assert client.app.state.workspace.status()["latest_outcome"]["outcome"] == "completed"


def test_center_capacity_eviction_and_return_from_sidebar(browser_api, model_for, remote_shell):
    client, _ = browser_api
    client.app.state.workspace.agent = server.Agent(
        model_for(
            ["charts = [ui.plot([{'type': 'bar', 'x': ['A'], 'y': [n]}], surface_id=f'chart_{n}') for n in range(5)]\ndone()"],
            ["charts[0].focus()\ndone()"],
            ["charts[2].pin()\ncharts[2].unpin()\ncharts[4].remove()\ndone()"],
        ),
        shell=remote_shell,
    )

    # Creating a fifth centered widget moves the oldest one into the sidebar.
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
        socket.receive_json()
        socket.send_json({"type": "run", "instruction": "create five widgets"})
        events = receive_until_idle(socket)
        placements = [event["widgets"] for event in events if event["type"] == "placement"]
        assert all(sum(not value["pinned"] for value in state.values()) <= 4 for state in placements)
        assert placements[-1]["chart_0"] == {"pinned": True}
        assert all(placements[-1][f"chart_{n}"] == {"pinned": False} for n in range(1, 5))

        # Refocusing a sidebar widget evicts the next oldest centered widget.
        socket.send_json({"type": "run", "instruction": "focus the first chart"})
        events = receive_until_idle(socket)
        state = [event["widgets"] for event in events if event["type"] == "placement"][-1]
        assert state["chart_0"] == {"pinned": False}
        assert state["chart_1"] == {"pinned": True}

        # Explicit unpin and removal must also be reflected in reconnect snapshots.
        socket.send_json({"type": "run", "instruction": "unpin a chart and remove another"})
        events = receive_until_idle(socket)
        state = [event["widgets"] for event in events if event["type"] == "placement"][-1]
        assert state["chart_2"] == {"pinned": False}
        assert "chart_4" not in state
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8010"}) as socket:
        assert socket.receive_json()["placements"] == state


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

    async def run(instruction, emit):
        started.set()
        assert await asyncio.to_thread(release.wait, 5)
        emit({"type": "result", "stdout": "finished\n", "stderr": "", "exception": None, "done": True})
        finished.set()
        return "Finished"

    monkeypatch.setattr(client.app.state.workspace.agent, "arun", run)

    # Submit work from the tab that will disconnect.
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:5180"}) as socket:
        socket.receive_json()
        socket.send_json({"type": "run", "instruction": "work"})

        # Wait until work is active before disconnecting its originating tab.
        assert started.wait(timeout=5)
    try:
        with client.websocket_connect("/ws", headers={"origin": "http://localhost:5180"}) as socket:
            assert socket.receive_json()["busy"] is True
            socket.send_json({"type": "run", "instruction": "overlapping work"})
            assert socket.receive_json()["type"] == "error"
            assert socket.receive_json() == {"type": "state", "busy": True}

            # The disconnected tab's work must finish without accepting overlap.
            release.set()
            events = receive_until_idle(socket)
            assert {"type": "result", "exception": None} in events
            assert finished.is_set()
    finally:
        release.set()
