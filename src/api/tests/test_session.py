import pytest
from api.shell import PythonShell
from src.api.tests.ui_helpers import RecordingUIState


@pytest.fixture
def shell():
    return PythonShell(RecordingUIState())


def test_widgets_publish_a2ui_messages_and_preserve_python_state(shell):
    result = shell.execute(
        "amount = 42\nui.plot([{'type': 'bar', 'x': ['Revenue'], 'y': [amount]}], surface_id='revenue')\nprint(amount)\ndone()"
    )
    assert result["exception"] is None
    assert result["done"] is True
    assert result["stdout"] == "42\n"
    created, updated = shell.ui.state.messages
    assert created["createSurface"]["surfaceId"] == "revenue"
    assert updated["updateComponents"]["surfaceId"] == "revenue"
    (component,) = updated["updateComponents"]["components"]
    assert component["id"] == "root"
    assert component["component"] == "Figure"
    (trace,) = component["data"]
    assert (trace["type"], trace["x"], trace["y"]) == ("bar", ["Revenue"], [42])
    assert shell.ui.state.snapshot()["events"] == [{"type": "a2ui", "messages": shell.ui.state.messages}]
    shell.ui.state.messages.clear()
    assert shell.execute("print(amount)")["exception"] is None
    assert shell.ui.state.messages == []

    # Replacing a named surface deletes its old protocol state first.
    assert shell.execute("ui.plot([{'type': 'scatter', 'x': ['Revenue'], 'y': [50]}], surface_id='revenue')")["exception"] is None
    replacement = list(shell.ui.state.messages)
    assert replacement[0] == {"version": "v0.9", "deleteSurface": {"surfaceId": "revenue"}}
    assert replacement[1]["createSurface"]["surfaceId"] == "revenue"
    assert replacement[2]["updateComponents"]["components"][0]["data"][0]["mode"] == "lines"
    shell.ui.state.messages.clear()
    assert shell.execute("ui.remove('revenue')")["exception"] is None
    assert shell.ui.state.messages == [replacement[0]]
    assert shell.ui.state.snapshot()["events"] == []


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


def test_automatic_surface_ids_do_not_replace_named_results(shell):
    result = shell.execute(
        "ui.plot([{'type': 'bar', 'x': ['A'], 'y': [1]}], surface_id='surface_1')\nui.plot([{'type': 'bar', 'x': ['B'], 'y': [2]}])"
    )
    assert result["exception"] is None
    created = [message["createSurface"]["surfaceId"] for message in shell.ui.state.messages if "createSurface" in message]
    assert created == ["surface_1", "surface_2"]
    assert not any("deleteSurface" in message for message in shell.ui.state.messages)
