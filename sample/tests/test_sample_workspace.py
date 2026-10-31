import json
import pytest
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from sample.src.events import Result
from sample.src.workspace import Workspace, Subscription
from pydantic_ai.models.test import TestModel
from sample.src.agents.script import Agent


def make_agent(responses, results):
    agent = Agent(TestModel(), shell=SimpleNamespace(stream=AsyncMock(side_effect=results), uncertain=False))
    agent.agent = SimpleNamespace(run=AsyncMock(side_effect=responses))
    return agent


def response(code, *, finish_reason="stop"):
    return SimpleNamespace(output=code, response=SimpleNamespace(finish_reason=finish_reason), all_messages=lambda: [code])


def result(text, *, done):
    return Result(type="result", stdout=text, stderr="", exception=None, done=done)


def test_agent_reports_phases_in_order_for_each_execution():
    agent = make_agent([response("print('first')"), response("done()")], [result("first", done=False), result("finished", done=True)])
    phases = []

    output = asyncio.run(agent.arun("work", phases.append))

    assert phases == ["generating_code", "executing_python", "generating_code", "executing_python"]
    assert output == "first\nfinished"
    assert [call.args[0] for call in agent.shell.stream.await_args_list] == ["print('first')", "done()"]


@pytest.mark.parametrize(
    ("generated", "message"),
    [
        (RuntimeError("generation failed"), "generation failed"),
        (response("partial code", finish_reason="length"), "code was not executed"),
        (response("  "), "no Python code"),
    ],
)
def test_generation_failure_never_reports_or_starts_execution(generated, message):
    agent = make_agent([generated], [])
    phases = []

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(agent.arun("work", phases.append))

    assert phases == ["generating_code"]
    agent.shell.stream.assert_not_awaited()


def test_workspace_ignores_phase_updates_while_idle():
    workspace = Workspace(make_agent([], []))
    subscription = Subscription(workspace)
    subscription.queue.get_nowait()

    workspace.update_phase("generating_code")
    assert workspace.phase == "idle"

    workspace.current_instruction = "work"
    workspace.update_phase("executing_python")
    assert workspace.phase == "executing_python"
    assert subscription.queue.empty()


@pytest.mark.parametrize("fail", [False, True])
def test_workspace_returns_to_idle_and_only_publishes_browser_messages(fail):
    async def run():
        agent = make_agent([], [])
        workspace = Workspace(agent)
        subscription = Subscription(workspace)
        reports = []
        workspace.results.add(reports.append)
        observed = []

        async def arun(instruction, on_phase):
            assert instruction == "work"
            for phase in ("generating_code", "executing_python"):
                on_phase(phase)
                observed.append(workspace.status())
            if fail:
                raise RuntimeError("execution failed")
            return "finished"

        agent.arun = arun
        workspace.start(" work ")
        assert workspace.phase == "starting"
        await workspace.task

        assert [status["phase"] for status in observed] == ["generating_code", "executing_python"]
        assert all(status["busy"] and status["instruction"] == "work" for status in observed)
        assert workspace.status() == {"busy": False, "phase": "idle", "instruction": None, "execution_uncertain": False}
        events = []
        while not subscription.queue.empty():
            events.append(subscription.queue.get_nowait())
        expected = [{"type": "snapshot"}, {"type": "instruction"}]
        if fail:
            expected.append({"type": "error", "text": "execution failed"})
        assert events == expected
        record = workspace.recent_tasks[-1]
        assert record["outcome"] == ("failed" if fail else "completed")
        assert record["result"] == (
            "Task failed: execution failed. Earlier changes may have applied; do not automatically resubmit." if fail else "finished"
        )
        assert [json.loads(report) for report in reports] == [record]

    asyncio.run(run())


@pytest.mark.parametrize("unavailable", [False, True])
def test_context_snapshots_history_before_fetching_charts(unavailable):
    async def run():
        agent = make_agent([], [])
        workspace = Workspace(agent)
        workspace.recent_tasks.append({"instruction": "first", "outcome": "completed", "result": "finished"})
        fetching = asyncio.Event()
        release = asyncio.Event()

        async def context():
            fetching.set()
            await release.wait()
            if unavailable:
                raise RuntimeError("Execution container unavailable")
            return {"charts": [{"surface_id": "sales"}]}

        agent.shell.context = context
        pending = asyncio.create_task(workspace.context())
        await asyncio.wait_for(fetching.wait(), 1)
        workspace.recent_tasks.append({"instruction": "second", "outcome": "completed", "result": "new result"})
        release.set()
        snapshot = await asyncio.wait_for(pending, 1)

        assert snapshot["recent_tasks"] == [{"instruction": "first", "outcome": "completed", "result": "finished"}]
        assert snapshot["charts"] == ([] if unavailable else [{"surface_id": "sales"}])
        assert snapshot.get("charts_unavailable", False) is unavailable
        assert len(workspace.recent_tasks) == 2

    asyncio.run(run())
