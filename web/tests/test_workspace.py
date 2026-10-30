import json
import pytest
import asyncio
from types import SimpleNamespace
from src.ui.session import UI
from web.src.workspace import Command, Workspace


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

        workspace = Workspace(SimpleNamespace(model="test", shell=SimpleNamespace(uncertain=False), arun=execute))
        reports = []
        workspace.results.add(reports.append)
        task = workspace.start(Command(type="run", instruction="Plot monthly sales"))
        await started.wait()
        assert workspace.status()["instruction"] == "Plot monthly sales"
        assert workspace.status()["phase"] == "executing_python"
        assert workspace.context() == {"recent_tasks": [], "charts": []}
        assert workspace.task is task
        release.set()
        await task
        assert workspace.status()["phase"] == "idle"
        assert workspace.status()["instruction"] is None
        context = workspace.context()
        assert context["charts"] == [
            {"surface_id": "sales", "title": "Monthly sales", "trace_types": ["bar"], "series": ["Sales"], "pinned": False}
        ]
        assert json.loads(reports[0])["changed_charts"] == ["sales"]
        assert context["recent_tasks"][0]["result"] == "Confirmed sales: 42"
        ui.on_publish = workspace.publish
        ui.remove("sales")
        assert workspace.context()["charts"] == []

    asyncio.run(run())


def test_recent_work_is_bounded_and_preserves_failed_partial_changes():
    async def run():
        async def execute(instruction, emit):
            if instruction == "fail":
                raise RuntimeError("Earlier changes may have applied")
            return "x" * 5000

        workspace = Workspace(SimpleNamespace(model="test", shell=SimpleNamespace(uncertain=False), arun=execute))
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


def test_task_outcomes_track_chart_updates_placements_and_partial_removal():
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

        workspace = Workspace(SimpleNamespace(model="test", shell=SimpleNamespace(uncertain=False), arun=execute))
        for instruction in ("create", "update", "pin"):
            await workspace.start(Command(type="run", instruction=instruction))
            assert workspace.status()["latest_outcome"]["changed_charts"] == ["sales"]
        assert workspace.context()["charts"][0]["title"] == "Updated sales"
        assert workspace.context()["charts"][0]["pinned"] is True
        await workspace.start(Command(type="run", instruction="remove"))
        result = workspace.status()["latest_outcome"]
        assert result["outcome"] == "failed"
        assert result["removed_charts"] == ["sales"]
        assert workspace.context()["charts"] == []

    asyncio.run(run())


def test_placement_snapshots_and_broadcasts_remain_detached():
    workspace = Workspace(SimpleNamespace(model="test"))
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

        workspace = Workspace(SimpleNamespace(model="test", shell=SimpleNamespace(uncertain=False), arun=execute))
        with pytest.raises(ValueError, match="Enter an instruction"):
            workspace.start(Command(type="run", instruction=" \t\n "))
        assert workspace.events == []
        assert workspace.busy is False
        task = workspace.start(Command(type="run", instruction=" \t Plot sales \n"))
        assert workspace.current_instruction == "Plot sales"
        assert [event["text"] for event in workspace.events if event["type"] == "instruction"] == ["Plot sales"]
        await task
        await workspace.execute(Command(type="run", instruction=" \t Direct call \n"))
        assert received == ["Plot sales", "Direct call"]
        assert [record["instruction"] for record in workspace.recent_tasks] == received

    asyncio.run(run())
