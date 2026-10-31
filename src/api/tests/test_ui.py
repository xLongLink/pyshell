import pytest
import asyncio
from api.main import create_app
from threading import Thread
from api.ui.state import UIState
from api.ui.session import UI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from src.api.tests.helpers import execute


def test_container_serves_widget_frontend_with_restricted_framing(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>Widgets</title>")
    (tmp_path / "app.js").write_text("console.log('widgets')")
    app = create_app()
    frontend = app.routes[-1].app
    frontend.directory = str(tmp_path)
    frontend.all_directories = [str(tmp_path)]
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Widgets" in response.text
        policy = response.headers["content-security-policy"]
        assert "frame-ancestors http://127.0.0.1:8010 http://localhost:8010" in policy
        assert "script-src 'self'" in policy
        assert client.get("/app.js").text == "console.log('widgets')"
        assert client.get("/missing.js").status_code == 404


@pytest.mark.parametrize(
    ("path", "method"), [("/execute/stream", "post"), ("/reset", "post"), ("/libraries", "get"), ("/ui/context", "get")]
)
def test_control_endpoints_accept_server_requests_and_reject_browser_requests(path, method):
    with TestClient(create_app()) as client:
        request = getattr(client, method)
        kwargs = {"json": {"code": "pass"}} if path == "/execute/stream" else {}
        assert request(path, **kwargs).status_code == 200
        for browser in [{"origin": "http://localhost:8000"}, {"sec-fetch-mode": "same-origin"}]:
            assert request(path, headers=browser, **kwargs).status_code == 403


def test_health():
    with TestClient(create_app()) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("origin", [None, "http://localhost:8010", "https://evil.example", "http://localhost:8000.evil.example"])
def test_widget_socket_rejects_origins_other_than_its_own(origin):
    with TestClient(create_app()) as client:
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect("/ui/ws", headers={"origin": origin} if origin else {}):
                pass
        assert error.value.code == 1008


def test_widgets_update_directly_and_reconnect_without_reexecuting_python():
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ui/ws", headers={"origin": "http://localhost:8000"}) as first:
            assert first.receive_json() == {"type": "snapshot", "events": [], "placements": {}}
            with client.websocket_connect("/ui/ws", headers={"origin": "http://localhost:8000"}) as second:
                assert second.receive_json()["events"] == []
                result = execute(client, "count = 1\nchart = ui.plot([{'type': 'bar', 'x': ['A'], 'y': [count]}], surface_id='sales')")
                assert result["exception"] is None
                snapshot = first.receive_json()
                assert second.receive_json() == snapshot
                assert snapshot["placements"] == {"sales": {"pinned": False}}
                execute(client, "chart.pin()")
                assert first.receive_json()["placements"] == {"sales": {"pinned": True}}
                assert second.receive_json()["placements"] == {"sales": {"pinned": True}}
        execute(client, "chart.update(layout={'title': {'text': 'Updated'}})")
        with client.websocket_connect("/ui/ws", headers={"origin": "http://localhost:8000"}) as restored:
            snapshot = restored.receive_json()
            component = snapshot["events"][0]["messages"][1]["updateComponents"]["components"][0]
            assert component["layout"]["title"]["text"] == "Updated"
            assert execute(client, "print(count)")["stdout"] == "1\n"
            assert client.get("/ui/context").json()["charts"][0]["title"] == "Updated"
            assert client.post("/reset").status_code == 200
            assert restored.receive_json() == {"type": "snapshot", "events": [], "placements": {}}
            assert client.get("/ui/context").json() == {"charts": []}
            assert "NameError" in execute(client, "print(count)")["exception"]
            fresh = execute(client, "ui.plot([{'type': 'bar', 'x': ['A'], 'y': [2]}], surface_id='sales')")["ui"]
            assert fresh[0]["createSurface"]["surfaceId"] == "sales"


def test_widget_connection_cannot_submit_python():
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ui/ws", headers={"origin": "http://localhost:8000"}) as socket:
            socket.receive_json()
            socket.send_json({"code": "print('not allowed')"})
            with pytest.raises(WebSocketDisconnect) as error:
                socket.receive_json()
            assert error.value.code == 1008


def test_container_state_preserves_focus_replacement_and_partial_updates():
    ui = UI()
    charts = [ui.plot([{"type": "bar", "x": ["A"], "y": [1]}], surface_id=f"chart_{index}") for index in range(5)]
    before = ui.state.snapshot()
    assert before["placements"]["chart_0"] == {"pinned": True}
    charts[0].focus()
    charts[2].pin()
    charts[1].focus()
    assert list(ui.state.snapshot()["placements"]) == ["chart_2", "chart_3", "chart_4", "chart_0", "chart_1"]
    charts[0].update(data=[{"type": "bar", "x": ["A"], "y": [2]}])
    assert before["events"][0]["messages"][1]["updateComponents"]["components"][0]["data"][0]["y"] == [1]
    replacement = ui.plot([{"type": "bar", "x": ["A"], "y": [3]}], surface_id="chart_0")
    assert len(ui.state.snapshot()["events"]) == 5
    replacement.remove()
    assert "chart_0" not in ui.state.snapshot()["placements"]


def test_surface_snapshots_preserve_creation_and_only_the_latest_update():
    state = UIState()
    creation = {"createSurface": {"surfaceId": "sales"}}
    state.publish([creation])
    initial = state.snapshot()
    assert initial == {"type": "snapshot", "events": [{"type": "a2ui", "messages": [creation]}], "placements": {"sales": {"pinned": False}}}
    assert state.context() == {"charts": []}

    updates = [
        {"updateComponents": {"surfaceId": "sales", "components": [{"layout": {"title": {"text": title}}, "data": []}]}}
        for title in ["First", "Latest"]
    ]
    state.publish([updates[0]])
    previous = state.snapshot()
    state.publish([updates[1]])
    assert initial["events"][0]["messages"] == [creation]
    assert previous["events"][0]["messages"] == [creation, updates[0]]
    assert state.snapshot()["events"] == [{"type": "a2ui", "messages": [creation, updates[1]]}]
    assert state.context() == {"charts": [{"surface_id": "sales", "title": "Latest", "trace_types": [], "series": [], "pinned": False}]}
    snapshot = state.snapshot()
    snapshot["events"][0]["messages"].clear()
    assert state.snapshot()["events"][0]["messages"] == [creation, updates[1]]

    state.publish([creation])
    assert state.snapshot() == initial
    assert state.context() == {"charts": []}


def test_slow_widget_subscriber_receives_latest_snapshot_without_blocking_execution():
    async def run():
        ui = UI()
        queue = ui.state.subscribe()
        chart = ui.plot([{"type": "bar", "x": ["A"], "y": [0]}])

        def update():
            for number in range(20):
                chart.update(data=[{"type": "bar", "x": ["A"], "y": [number]}])

        worker = Thread(target=update)
        worker.start()
        await asyncio.to_thread(worker.join, 2)
        assert not worker.is_alive()
        await asyncio.sleep(0)
        assert queue.qsize() == 1
        snapshot = queue.get_nowait()
        assert snapshot == ui.state.snapshot()
        assert snapshot["events"][0]["messages"][1]["updateComponents"]["components"][0]["data"][0]["y"] == [19]
        ui.state.unsubscribe(queue)
        assert not ui.state.listeners

    asyncio.run(run())
