import json
import pytest
from api.shell import PythonShell
from api.ui.session import UI
from api.ui.plot.figure import Figure
from src.api.tests.ui_helpers import RecordingUIState


@pytest.mark.parametrize("text", ["plain", 'é雪😀\n\t"\\', "\ud800"])
@pytest.mark.parametrize("split", [False, True])
def test_byte_budget_matches_default_json_encoding_at_exact_limit(text, split):
    state = RecordingUIState()
    ui = UI(state)
    first = {"text": text}
    last = {"text": ""}
    last["text"] = "a" * (1048576 - len(json.dumps([first, last], allow_nan=False).encode()))
    if split:
        ui._publish([first])
        with pytest.raises(ValueError, match="1 MiB"):
            ui._publish([{"text": last["text"] + "a"}])
        assert state.messages == [first]
        ui._publish([])
        ui._publish([last])
    else:
        with pytest.raises(ValueError, match="1 MiB"):
            ui._publish([first, {"text": last["text"] + "a"}])
        assert state.messages == []
        ui._publish([first, last])
    assert len(json.dumps(state.messages, allow_nan=False).encode()) == 1048576
    with pytest.raises(ValueError, match="1 MiB"):
        ui._publish([{}])
    assert state.messages == [first, last]


def test_node_budget_counts_one_combined_root_and_accumulates_batches():
    state = RecordingUIState()
    ui = UI(state)
    ui._publish([{"items": [0] * 9997}])
    ui._publish([{"items": [0] * 9996}])
    with pytest.raises(ValueError, match="complexity"):
        ui._publish([{"items": [0]}])
    ui._publish([{"value": 0}])
    assert len(state.messages) == 3
    with pytest.raises(ValueError, match="complexity"):
        ui._publish([{}])
    assert len(state.messages) == 3


def test_message_budget_accepts_exact_limit_after_rejected_batch():
    state = RecordingUIState()
    ui = UI(state)
    ui._publish([{}] * 99)
    with pytest.raises(ValueError, match="100 messages"):
        ui._publish([{}, {}])
    ui._publish([{}])
    assert len(state.messages) == 100
    with pytest.raises(ValueError, match="100 messages"):
        ui._publish([{}])


def test_failed_state_publication_does_not_spend_any_budget(monkeypatch):
    state = RecordingUIState()
    ui = UI(state)
    message = {"items": ["a" * 900000] + [0] * 19000}

    def fail(messages):
        raise RuntimeError("publication failed")

    with monkeypatch.context() as patch:
        patch.setattr(state, "publish", fail)
        with pytest.raises(RuntimeError, match="publication failed"):
            ui._publish([message] + [{}] * 99)
    ui._publish([message] + [{}] * 99)
    assert len(state.messages) == 100


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_serialization_does_not_spend_budget(value):
    state = RecordingUIState()
    ui = UI(state)
    with pytest.raises(ValueError, match="JSON"):
        ui._publish([{"value": value}])
    ui._publish([{}] * 100)
    assert len(state.messages) == 100


@pytest.mark.parametrize("text", ["a", "é", "😀", '"'])
def test_surface_size_uses_default_json_encoding(text):
    ui = UI()
    attributes = {"data": [{"type": "bar", "x": ["a" * 2550] * 100, "y": [1] * 100}], "layout": {"title": {"text": text}}}
    component = Figure.model_validate(attributes).model_dump(mode="json")
    padding = 262144 - len(json.dumps(component, allow_nan=False).encode())
    attributes["layout"]["title"]["text"] = text + "a" * padding
    widget = ui.plot(**attributes)
    snapshot = ui.state.snapshot()
    with pytest.raises(ValueError, match="256 KiB"):
        widget.update(layout={"title": {"text": text + "a" * (padding + 1)}})
    assert ui.state.snapshot() == snapshot


@pytest.mark.parametrize("messages", ["[{}] * 100", "[{'text': 'a' * 1048562}]", "[{'items': [0] * 19997}]"])
@pytest.mark.parametrize("fail", [False, True])
def test_execution_resets_all_output_budgets_without_resetting_handles(messages, fail):
    shell = PythonShell()
    assert shell.execute("chart = ui.plot([{'type': 'bar', 'x': ['A'], 'y': [1]}])")["exception"] is None
    source = f"ui._publish({messages})" + ("\nraise RuntimeError('failed execution')" if fail else "")
    for _ in range(2):
        result = shell.execute(source)
        assert (result["exception"] is not None) == fail
        if fail:
            assert "failed execution" in result["exception"]
    result = shell.execute("chart.update(layout={'title': {'text': 'Still usable'}})")
    assert result["exception"] is None
    assert set(result) == {"stdout", "stderr", "exception", "done"}
    assert shell.namespace["chart"].layout.title.text == "Still usable"
