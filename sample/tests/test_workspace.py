import json
import pytest
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from sample.src.workspace import Command, Workspace


def test_workspace_inspection_tracks_phases_and_fetches_container_chart_context():
    async def run():
        # Arrange
        started = asyncio.Event()
        release = asyncio.Event()
        charts = [{"surface_id": "sales", "title": "Sales", "pinned": False}]
        updated_charts = [{"surface_id": "sales", "title": "Updated sales", "pinned": True}]
        context = AsyncMock(side_effect=[{"charts": charts}, {"charts": updated_charts}])

        async def execute(instruction, emit):
            emit({"type": "code_start"})
            assert workspace.status()["phase"] == "generating_code"
            emit({"type": "execution_start"})
            started.set()
            await release.wait()
            return "Confirmed sales: 42"

        workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(uncertain=False, context=context), arun=execute))
        reports = []
        workspace.results.add(reports.append)

        # Act
        task = workspace.start(Command(type="run", instruction="Plot sales"))
        await started.wait()
        assert workspace.status()["phase"] == "executing_python"
        assert (await workspace.context())["charts"] == charts
        assert workspace.snapshot() == {"type": "snapshot"}
        release.set()
        await task

        # Assert
        assert workspace.status()["phase"] == "idle"
        completed_context = await workspace.context()
        assert completed_context["charts"] == updated_charts
        assert json.loads(reports[0]) == completed_context["recent_tasks"][0]

    asyncio.run(run())


def test_context_failure_is_explicit_without_losing_recent_results():
    async def run():
        workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(context=AsyncMock(side_effect=ConnectionError("private address")))))
        workspace.recent_tasks.append({"result": "Confirmed"})
        assert await workspace.context() == {"recent_tasks": [{"result": "Confirmed"}], "charts": [], "charts_unavailable": True}

    asyncio.run(run())


@pytest.mark.parametrize("event", [{"type": "code_start"}, {"type": "execution_start"}, {"type": "a2ui", "messages": []}])
def test_internal_events_are_not_retained_or_broadcast(event):
    workspace = Workspace(SimpleNamespace())
    queue = asyncio.Queue(maxsize=10)
    workspace.listeners.add(queue)
    workspace.publish(event)
    assert queue.empty()
    assert workspace.phase == "idle"
    assert workspace.snapshot() == {"type": "snapshot"}


@pytest.mark.parametrize("event", [{"type": "instruction", "text": "Plot sales"}, {"type": "error", "text": "Task failed"}])
def test_instructions_and_errors_are_delivered_live_but_not_replayed(event):
    workspace = Workspace(SimpleNamespace())
    queue = asyncio.Queue(maxsize=10)
    workspace.listeners.add(queue)
    workspace.publish(event)
    assert queue.get_nowait() == event
    assert workspace.snapshot() == {"type": "snapshot"}


def test_recent_work_is_bounded_and_preserves_failed_partial_changes():
    async def run():
        async def execute(instruction, emit):
            if instruction == "fail":
                raise RuntimeError("Earlier changes may have applied")
            return "x" * 5000

        workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(uncertain=False), arun=execute))
        for index in range(10):
            await workspace.start(Command(type="run", instruction=str(index)))
        assert len(workspace.recent_tasks) == 8
        assert workspace.recent_tasks[0]["instruction"] == "2"
        assert len(workspace.recent_tasks[-1]["result"]) == 4000
        await workspace.start(Command(type="run", instruction="fail"))
        outcome = workspace.status()["latest_outcome"]
        assert outcome["outcome"] == "failed"
        assert "Earlier changes may have applied" in outcome["result"]
        assert workspace.status()["busy"] is False

    asyncio.run(run())


def test_instructions_are_normalized_for_acceptance_execution_and_history():
    async def run():
        received = []

        async def execute(instruction, emit):
            received.append(instruction)
            return "Confirmed"

        workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(uncertain=False), arun=execute))
        with pytest.raises(ValueError, match="Enter an instruction"):
            workspace.start(Command(type="run", instruction=" \t\n "))
        queue = asyncio.Queue(maxsize=10)
        workspace.listeners.add(queue)
        task = workspace.start(Command(type="run", instruction=" \t Plot sales \n"))
        assert workspace.busy is True
        assert queue.get_nowait() == {"type": "instruction", "text": "Plot sales"}
        await task
        assert workspace.busy is False
        await workspace.start(Command(type="run", instruction=" \t Next call \n"))
        assert received == ["Plot sales", "Next call"]
        assert [record["instruction"] for record in workspace.recent_tasks] == received

    asyncio.run(run())
