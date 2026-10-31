import pytest
from api.shell import PythonShell
from api.ui.session import UI
from src.api.tests.ui_helpers import RecordingUIState


@pytest.fixture
def shell():
    return PythonShell(RecordingUIState())


def test_display_limits_reject_excess_surfaces(shell):
    result = shell.execute("for number in range(21):\n    ui.plot([{'type': 'bar', 'x': ['A'], 'y': [number]}])")
    assert "20 surfaces" in result["exception"]
    assert sum("createSurface" in message for message in shell.ui.state.messages) == 20
    assert len(shell.ui.state.snapshot()["events"]) == 20


def test_pending_message_limit_keeps_last_successful_surface(shell):
    result = shell.execute("for number in range(40):\n    ui.plot([{'type': 'bar', 'x': ['A'], 'y': [number]}], surface_id='counter')")
    assert "100 messages" in result["exception"]
    assert len(shell.ui.state.messages) == 98
    assert shell.ui.state.messages[-1]["updateComponents"]["surfaceId"] == "counter"
    assert shell.ui.state.messages[-1]["updateComponents"]["components"][0]["data"][0]["y"] == [32]
    snapshot = shell.ui.state.snapshot()
    assert snapshot["events"] == [{"type": "a2ui", "messages": shell.ui.state.messages[-2:]}]
    assert shell.execute("ui.remove('counter')")["exception"] is None
    assert shell.ui.state.snapshot()["events"] == []


def test_automatic_surface_ids_skip_consecutive_named_collisions():
    ui = UI(RecordingUIState())
    data = [{"type": "bar", "x": ["A"], "y": [1]}]
    for number in range(1, 4):
        ui.plot(data, surface_id=f"surface_{number}")

    assert ui.plot(data).surface_id == "surface_4"
    assert list(ui.state.snapshot()["placements"]) == ["surface_1", "surface_2", "surface_3", "surface_4"]
    assert not any("deleteSurface" in message for message in ui.state.messages)


def test_automatic_surface_ids_remain_monotonic_after_removal():
    ui = UI(RecordingUIState())
    data = [{"type": "bar", "x": ["A"], "y": [1]}]
    named = ui.plot(data, surface_id="surface_1")
    automatic = ui.plot(data)
    assert automatic.surface_id == "surface_2"

    named.remove()
    automatic.remove()

    assert ui.plot(data).surface_id == "surface_3"
    assert list(ui.state.snapshot()["placements"]) == ["surface_3"]
