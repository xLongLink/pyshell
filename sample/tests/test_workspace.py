import json
import httpx
import pytest
import asyncio
from types import SimpleNamespace
from pydantic import ValidationError
from sample.src.shell import RemoteShell
from pydantic_ai.messages import TextPart, ModelResponse
from sample.src.workspace import Workspace, Subscription
from sample.src.agents.script import Agent
from pydantic_ai.models.function import FunctionModel


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
@pytest.mark.parametrize("charts_available", [True, False])
def test_inspection_keeps_outcomes_only_in_recent_tasks(finish_reason, charts_available):
    async def run():
        started = asyncio.Event()
        release = asyncio.Event()
        executions = []
        charts = [{"surface_id": "sales", "title": "Sales", "trace_types": ["bar"], "series": ["Revenue"], "pinned": False}]

        async def respond(_messages, _info):
            started.set()
            await release.wait()
            return ModelResponse(parts=[TextPart("done()")], finish_reason=finish_reason)

        def handle(request):
            if request.url.path == "/libraries":
                return httpx.Response(200, json=[])
            if request.url.path == "/ui/context":
                return httpx.Response(200, json={"charts": charts}) if charts_available else httpx.Response(503)
            executions.append(json.loads(request.content)["code"])
            result = {"type": "result", "stdout": "Finished\n", "stderr": "", "exception": None, "done": True}
            return httpx.Response(200, text=json.dumps(result) + "\n" + json.dumps({"type": "execution_finished"}) + "\n")

        async with httpx.AsyncClient(base_url="http://execution", transport=httpx.MockTransport(handle)) as client:
            workspace = Workspace(Agent(FunctionModel(respond), shell=RemoteShell(client)))
            idle = {"busy": False, "phase": "idle", "instruction": None, "execution_uncertain": False}
            assert workspace.status() == idle
            assert (await workspace.context())["recent_tasks"] == []
            task = workspace.start("Summarize sales")
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                assert workspace.status() == {
                    "busy": True,
                    "phase": "generating_code",
                    "instruction": "Summarize sales",
                    "execution_uncertain": False,
                }
                assert (await workspace.context())["recent_tasks"] == []
                release.set()
                await task
                assert workspace.status() == idle
                context = await workspace.context()
                assert len(context["recent_tasks"]) == 1
                record = context["recent_tasks"][0]
                assert record["instruction"] == "Summarize sales"
                if finish_reason == "stop":
                    assert record["outcome"] == "completed"
                    assert record["result"] == "Finished\n"
                    assert executions == ["done()"]
                else:
                    assert record["outcome"] == "failed"
                    assert "Model response length; code was not executed." in record["result"]
                    assert "Earlier changes may have applied" in record["result"]
                    assert not executions
                assert context["charts"] == (charts if charts_available else [])
                assert context.get("charts_unavailable", False) is not charts_available
                record["result"] = "Changed by caller"
                assert (await workspace.context())["recent_tasks"][0]["result"] != "Changed by caller"
            finally:
                release.set()
                await workspace.close()

    asyncio.run(run())


@pytest.mark.parametrize("closing", [False, True])
@pytest.mark.parametrize("busy", [False, True])
@pytest.mark.parametrize("uncertain", [False, True])
@pytest.mark.parametrize("instruction", [None, 123, b"calculate", "x" * 8001, " " * 8001, " \n\t", "calculate"])
def test_direct_submission_admission_order(instruction, closing, busy, uncertain):
    async def arun(instruction, publish):
        return "completed"

    async def accepted():
        await workspace.start(instruction)

    workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(uncertain=False)))
    workspace.agent.arun = arun
    workspace.closing = closing
    workspace.current_instruction = "existing" if busy else None
    workspace.agent.shell.uncertain = uncertain
    status = workspace.status()
    listener = Subscription(workspace)
    assert listener.queue.get_nowait() == {"type": "snapshot"}
    if not isinstance(instruction, str) or len(instruction) > 8000:
        error, message = ValidationError, "valid string|8000"
    elif closing:
        error, message = RuntimeError, "shutting down"
    elif busy:
        error, message = RuntimeError, "busy"
    elif not instruction.strip():
        error, message = ValueError, "Enter an instruction."
    elif uncertain:
        error, message = RuntimeError, "uncertain"
    else:
        asyncio.run(accepted())
        assert listener.queue.get_nowait() == {"type": "instruction", "text": instruction}
        assert workspace.recent_tasks[-1]["outcome"] == "completed"
        return
    with pytest.raises(error, match=message):
        workspace.start(instruction)
    assert workspace.status() == status
    assert workspace.task is None
    assert not workspace.recent_tasks
    assert listener.queue.empty()


def test_direct_submission_accepts_limit_and_strips_whitespace():
    async def run():
        instructions = []

        async def arun(instruction, publish):
            instructions.append(instruction)
            return "completed"

        workspace = Workspace(SimpleNamespace(shell=SimpleNamespace(uncertain=False), arun=arun))
        await workspace.start(" " + "x" * 7998 + " ")
        assert instructions == ["x" * 7998]
        assert not workspace.busy

    asyncio.run(run())
