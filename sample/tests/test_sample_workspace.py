import json
import pytest
import asyncio
from types import SimpleNamespace
from pydantic import ValidationError
from unittest.mock import AsyncMock
from sample.src.shell import EXECUTION_UNCERTAIN, Result
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


@pytest.mark.parametrize("text", ["first", 'caf\u00e9\n"quoted"'])
def test_agent_retains_execution_feedback_across_generation_failure(text):
    agent = make_agent(
        [response("print('first')"), RuntimeError("generation failed"), response("done()")],
        [result(text, done=True), result("finished", done=True)],
    )

    assert asyncio.run(agent.arun("work", lambda phase: None)) == text
    with pytest.raises(RuntimeError, match="generation failed"):
        asyncio.run(agent.arun("next", lambda phase: None))

    feedback = result(text, done=True).model_dump_json(exclude={"type"})
    error = json.dumps({"model_error": "RuntimeError: generation failed", "executed": False})
    retained = feedback + "\n" + error
    assert agent.pending_feedback == retained
    assert agent.message_history == ["print('first')"]

    assert asyncio.run(agent.arun("continue", lambda phase: None)) == "finished"
    prompts = [call.args[0] for call in agent.agent.run.await_args_list]
    assert prompts == ["work", feedback + "\nnext", retained + "\ncontinue"]
    assert [call.args[0] for call in agent.shell.stream.await_args_list] == ["print('first')", "done()"]


@pytest.mark.parametrize(
    ("generated", "error"),
    [
        (RuntimeError("generation failed"), RuntimeError),
        (asyncio.CancelledError(), asyncio.CancelledError),
        (response("partial", finish_reason="length"), RuntimeError),
        (response(" "), RuntimeError),
    ],
)
def test_agent_retains_incomplete_step_feedback_when_next_generation_fails(generated, error):
    executed = result("created record 123", done=False)
    feedback = executed.model_dump_json(exclude={"type"})
    agent = make_agent([response("create_record()"), generated, response("done()")], [executed, result("finished", done=True)])

    with pytest.raises(error):
        asyncio.run(agent.arun("create one record", lambda phase: None))

    assert agent.message_history == ["create_record()"]
    assert feedback in agent.pending_feedback
    agent.shell.stream.assert_awaited_once_with("create_record()")
    assert asyncio.run(agent.arun("continue without creating another record", lambda phase: None)) == "finished"
    prompts = [call.args[0] for call in agent.agent.run.await_args_list]
    assert prompts[1] == feedback
    assert prompts[2].startswith(feedback + "\n")
    assert prompts[2].endswith("\ncontinue without creating another record")
    assert [call.args[0] for call in agent.shell.stream.await_args_list] == ["create_record()", "done()"]


@pytest.mark.parametrize("error", [RuntimeError("stream failed"), asyncio.CancelledError()])
def test_agent_retains_history_when_execution_is_not_confirmed(error):
    agent = make_agent([response("print('first')")], [error])

    with pytest.raises(type(error)):
        asyncio.run(agent.arun("work", lambda phase: None))

    assert agent.message_history == ["print('first')"]
    assert json.loads(agent.pending_feedback) == {
        "execution_error": "Execution was not confirmed; earlier changes may have applied. Do not replay."
    }


@pytest.mark.parametrize(
    ("instruction", "uncertain", "error", "message"),
    [
        (None, False, ValidationError, "valid string"),
        (1, False, ValidationError, "valid string"),
        ("x" * 8001, False, ValidationError, "at most 8000 characters"),
        (" " * 8001, False, ValidationError, "at most 8000 characters"),
        ("work", True, RuntimeError, EXECUTION_UNCERTAIN),
    ],
)
def test_workspace_rejects_instructions_before_changing_state(instruction, uncertain, error, message):
    workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(uncertain=uncertain)))
    subscription = Subscription(workspace)
    subscription.queue.get_nowait()

    with pytest.raises(error) as raised:
        workspace.start(instruction)

    assert message in str(raised.value)
    assert workspace.current_instruction is None
    assert workspace.phase == "idle"
    assert workspace.task is None
    assert subscription.queue.empty()


@pytest.mark.parametrize("fail", [False, True])
def test_workspace_returns_to_idle_and_only_publishes_browser_messages(fail):
    async def run():
        agent = SimpleNamespace(shell=SimpleNamespace(uncertain=False))
        workspace = Workspace(agent)
        subscription = Subscription(workspace)
        reports = []
        workspace.results.add(reports.append)
        observed = []

        async def arun(instruction, on_phase):
            assert instruction == "work"
            for phase in ("generating_code", "executing_python"):
                on_phase(phase)
                observed.append((workspace.phase, workspace.busy, workspace.current_instruction))
            if fail:
                raise RuntimeError("execution failed")
            return "finished"

        agent.arun = arun
        workspace.start(" work ")
        assert workspace.phase == "starting"
        await workspace.task

        assert observed == [("generating_code", True, "work"), ("executing_python", True, "work")]
        assert (await workspace.context())["status"] == {"busy": False, "phase": "idle", "instruction": None, "execution_uncertain": False}
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
        assert reports == [record]

    asyncio.run(run())


def test_workspace_shutdown_waits_for_accepted_work_and_rejects_new_submissions():
    async def run():
        working = asyncio.Event()
        release = asyncio.Event()

        async def arun(instruction, on_phase):
            working.set()
            await release.wait()
            return "finished"

        workspace = Workspace(SimpleNamespace(arun=arun, shell=SimpleNamespace(uncertain=False)))
        workspace.start("accepted")
        task = workspace.task
        await asyncio.wait_for(working.wait(), 1)
        closing = asyncio.create_task(workspace.close())
        try:
            await asyncio.sleep(0)
            assert workspace.closing
            assert not closing.done()
            assert not task.done()
            with pytest.raises(RuntimeError, match="shutting down"):
                workspace.start("rejected")
            assert workspace.task is task
            assert workspace.current_instruction == "accepted"
        finally:
            release.set()
            await asyncio.wait_for(closing, 1)

        assert task.done()
        assert not task.cancelled()
        assert not workspace.busy
        assert workspace.recent_tasks == [{"instruction": "accepted", "outcome": "completed", "result": "finished"}]
        await workspace.close()
        with pytest.raises(RuntimeError, match="shutting down"):
            workspace.start("still rejected")

    asyncio.run(run())


@pytest.mark.parametrize("unavailable", [False, True])
def test_context_snapshots_history_before_fetching_charts(unavailable):
    async def run():
        agent = SimpleNamespace(shell=SimpleNamespace(uncertain=False))
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
        workspace.current_instruction = "second"
        workspace.phase = "executing_python"
        agent.shell.uncertain = True
        workspace.recent_tasks.append({"instruction": "second", "outcome": "completed", "result": "new result"})
        release.set()
        snapshot = await asyncio.wait_for(pending, 1)

        assert snapshot["status"] == {"busy": False, "phase": "idle", "instruction": None, "execution_uncertain": False}
        assert snapshot["recent_tasks"] == [{"instruction": "first", "outcome": "completed", "result": "finished"}]
        assert snapshot["charts"] == ([] if unavailable else [{"surface_id": "sales"}])
        assert snapshot.get("charts_unavailable", False) is unavailable
        assert len(workspace.recent_tasks) == 2

    asyncio.run(run())
