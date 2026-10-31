import json
import pytest
import asyncio
from types import SimpleNamespace
from src.ui.session import UI
from web.src.events import WidgetCommand
from web.src.workspace import Command, Workspace


def test_focus_recency_controls_eviction_without_reordering_surface_replay():
    workspace = Workspace(SimpleNamespace())
    ui = UI()
    ui.on_publish = workspace.publish
    charts = [ui.plot([{"type": "bar", "x": ["A"], "y": [1]}], surface_id=f"chart_{index}") for index in range(5)]
    before = workspace.snapshot()
    assert before["placements"]["chart_0"] == {"pinned": True}
    charts[0].focus()
    charts[2].pin()
    charts[1].focus()
    snapshot = workspace.snapshot()
    assert snapshot["placements"] == {
        "chart_2": {"pinned": True},
        "chart_3": {"pinned": False},
        "chart_4": {"pinned": False},
        "chart_0": {"pinned": False},
        "chart_1": {"pinned": False},
    }
    assert list(snapshot["placements"]) == ["chart_2", "chart_3", "chart_4", "chart_0", "chart_1"]
    assert snapshot["events"] == before["events"]
    assert before["placements"]["chart_0"] == {"pinned": True}
    charts[2].focus()
    assert workspace.snapshot()["placements"]["chart_3"] == {"pinned": True}


def test_typed_widget_commands_preserve_unknown_surface_rejection_and_wire_events():
    workspace = Workspace(SimpleNamespace())
    command = WidgetCommand(type="widget", surface_id="sales", action="pin")
    with pytest.raises(ValueError, match="Unknown or removed widget"):
        workspace.publish(command)
    ui = UI()
    ui.on_publish = workspace.publish
    ui.plot([{"type": "bar", "x": ["A"], "y": [1]}], surface_id="sales")
    queue = asyncio.Queue(maxsize=10)
    workspace.listeners.add(queue)
    workspace.publish(command)
    assert queue.get_nowait() == {"type": "placement", "widgets": {"sales": {"pinned": True}}}


def test_workspace_inspection_tracks_phases_charts_and_recent_results():
    async def run():
        started = asyncio.Event()
        release = asyncio.Event()
        ui = UI()

        async def execute(instruction, emit):
            emit({"type": "code_start"})
            assert workspace.status()["phase"] == "generating_code"
            emit({"type": "execution_start"})
            started.set()
            await release.wait()
            ui.on_publish = emit
            ui.plot(
                [{"type": "bar", "name": "Sales", "x": ["Jan"], "y": [42]}], layout={"title": {"text": "Monthly sales"}}, surface_id="sales"
            )
            return "Confirmed sales: 42"

        workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(uncertain=False), arun=execute))
        queue = asyncio.Queue(maxsize=20)
        workspace.listeners.add(queue)
        reports = []
        workspace.results.add(reports.append)
        task = workspace.start(Command(type="run", instruction="Plot monthly sales"))
        await started.wait()
        assert workspace.status()["instruction"] == "Plot monthly sales"
        assert workspace.status()["phase"] == "executing_python"
        assert workspace.context() == {"recent_tasks": [], "charts": []}
        assert workspace.task is task
        assert workspace.snapshot()["events"] == []
        while not queue.empty():
            assert queue.get_nowait()["type"] not in {"code_start", "execution_start"}
        release.set()
        await task
        assert workspace.status()["phase"] == "idle"
        assert workspace.status()["instruction"] is None
        context = workspace.context()
        assert context["charts"] == [
            {"surface_id": "sales", "title": "Monthly sales", "trace_types": ["bar"], "series": ["Sales"], "pinned": False}
        ]
        assert json.loads(reports[0]) == context["recent_tasks"][0]
        assert context["recent_tasks"][0]["result"] == "Confirmed sales: 42"
        ui.on_publish = workspace.publish
        ui.remove("sales")
        assert workspace.context()["charts"] == []

    asyncio.run(run())


@pytest.mark.parametrize("event", [{"type": "code_start"}, {"type": "execution_start"}])
def test_idle_execution_events_are_not_retained_or_broadcast_and_remain_unchanged(event):
    workspace = Workspace(SimpleNamespace())
    queue = asyncio.Queue(maxsize=10)
    workspace.listeners.add(queue)
    original = dict(event)
    workspace.publish(event)
    assert event == original
    assert queue.empty()
    assert workspace.phase == "idle"
    assert workspace.snapshot()["events"] == []


@pytest.mark.parametrize("event", [{"type": "instruction", "text": "Plot sales"}, {"type": "error", "text": "Task failed"}])
def test_instructions_and_errors_are_delivered_live_but_not_replayed(event):
    workspace = Workspace(SimpleNamespace())
    queue = asyncio.Queue(maxsize=10)
    workspace.listeners.add(queue)
    workspace.publish(event)
    assert queue.get_nowait() == event
    assert workspace.snapshot()["events"] == []


def test_snapshots_always_restore_current_surfaces_after_updates_replacement_and_removal():
    workspace = Workspace(SimpleNamespace())
    ui = UI()
    ui.on_publish = workspace.publish
    chart = ui.plot([{"type": "bar", "x": ["Jan"], "y": [1]}], surface_id="sales")
    old_snapshot = workspace.snapshot()
    chart.update(data=[{"type": "bar", "x": ["Jan"], "y": [2]}])
    chart.pin()
    discarded = ui.plot([{"type": "bar", "x": ["Jan"], "y": [3]}], surface_id="discarded")
    discarded.remove()
    replacement = ui.plot([{"type": "scatter", "x": ["Jan"], "y": [4]}], surface_id="sales")
    replacement.update(layout={"title": {"text": "Latest sales"}})
    snapshot = workspace.snapshot()
    content = [event for event in snapshot["events"] if event["type"] == "a2ui"]
    assert len(content) == 1
    create, update = content[0]["messages"]
    assert create["createSurface"]["surfaceId"] == "sales"
    assert update["updateComponents"]["surfaceId"] == "sales"
    figure = update["updateComponents"]["components"][0]
    assert figure["data"][0]["type"] == "scatter"
    assert figure["data"][0]["y"] == [4]
    assert figure["layout"]["title"]["text"] == "Latest sales"
    assert snapshot["placements"] == {"sales": {"pinned": False}}
    assert old_snapshot["events"][0]["messages"][1]["updateComponents"]["components"][0]["data"][0]["y"] == [1]
    replacement.remove()
    assert not any(event["type"] == "a2ui" for event in workspace.snapshot()["events"])
    assert workspace.snapshot()["placements"] == {}


def test_recent_work_is_bounded_and_preserves_failed_partial_changes():
    async def run():
        async def execute(instruction, emit):
            if instruction == "fail":
                raise RuntimeError("Earlier changes may have applied")
            return "x" * 5000

        workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(uncertain=False), arun=execute))
        for index in range(10):
            await workspace.start(Command(type="run", instruction=str(index)))
        context = workspace.context()
        assert len(context["recent_tasks"]) == 8
        assert context["recent_tasks"][0]["instruction"] == "2"
        assert len(context["recent_tasks"][-1]["result"]) == 4000
        await workspace.start(Command(type="run", instruction="fail"))
        outcome = workspace.status()["latest_outcome"]
        assert outcome["outcome"] == "failed"
        assert "Earlier changes may have applied" in outcome["result"]
        assert workspace.status()["busy"] is False

    asyncio.run(run())


def test_tasks_preserve_current_charts_and_partial_changes_on_failure():
    async def run():
        ui = UI()
        chart = None

        async def execute(instruction, emit):
            nonlocal chart
            ui.on_publish = emit
            if instruction == "create":
                chart = ui.plot([{"type": "bar", "x": ["Jan"], "y": [42]}], surface_id="sales")
            elif instruction == "update":
                chart.update(layout={"title": {"text": "Updated sales"}})
            elif instruction == "pin":
                chart.pin()
            else:
                chart.remove()
                raise RuntimeError("Failed after removal")
            return "Confirmed"

        workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(uncertain=False), arun=execute))
        for instruction in ("create", "update", "pin"):
            await workspace.start(Command(type="run", instruction=instruction))
        assert workspace.context()["charts"][0]["title"] == "Updated sales"
        assert workspace.context()["charts"][0]["pinned"] is True
        await workspace.start(Command(type="run", instruction="remove"))
        result = workspace.status()["latest_outcome"]
        assert result["outcome"] == "failed"
        assert workspace.context()["charts"] == []

    asyncio.run(run())


def test_placement_snapshots_and_broadcasts_remain_detached():
    workspace = Workspace(SimpleNamespace())
    ui = UI()
    ui.on_publish = workspace.publish
    chart = ui.plot([{"type": "bar", "x": ["Jan"], "y": [42]}], surface_id="sales")
    snapshot = workspace.snapshot()
    queue = asyncio.Queue(maxsize=10)
    workspace.listeners.add(queue)
    chart.pin()
    pinned = queue.get_nowait()
    chart.focus()
    assert snapshot["placements"] == {"sales": {"pinned": False}}
    assert pinned["widgets"] == {"sales": {"pinned": True}}
    assert workspace.snapshot()["placements"] == {"sales": {"pinned": False}}


def test_instructions_are_normalized_for_acceptance_execution_and_history():
    async def run():
        received = []

        async def execute(instruction, emit):
            received.append(instruction)
            return "Confirmed"

        workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(uncertain=False), arun=execute))
        with pytest.raises(ValueError, match="Enter an instruction"):
            workspace.start(Command(type="run", instruction=" \t\n "))
        assert workspace.snapshot()["events"] == []
        assert workspace.busy is False
        queue = asyncio.Queue(maxsize=10)
        workspace.listeners.add(queue)
        task = workspace.start(Command(type="run", instruction=" \t Plot sales \n"))
        assert workspace.busy is True
        assert workspace.current_instruction == "Plot sales"
        assert queue.get_nowait() == {"type": "instruction", "text": "Plot sales"}
        await task
        assert workspace.busy is False
        await workspace.start(Command(type="run", instruction=" \t Next call \n"))
        assert received == ["Plot sales", "Next call"]
        assert [record["instruction"] for record in workspace.recent_tasks] == received

    asyncio.run(run())
