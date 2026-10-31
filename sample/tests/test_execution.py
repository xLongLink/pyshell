import json
import httpx
import pytest
import asyncio
from sample.src.shell import RemoteShell
from sample.src.events import Result
from pydantic_ai.messages import TextPart, ModelResponse
from sample.src.agents.script import INSTRUCTIONS, Agent
from pydantic_ai.models.function import FunctionModel


def stream_response(events):
    return httpx.Response(200, text="".join(json.dumps(event) + "\n" for event in events))


def test_stream_returns_validated_result():
    expected = {"type": "result", "stdout": "before\n", "stderr": "warning\n", "exception": "ValueError: failed", "done": False}
    transport = httpx.MockTransport(lambda _request: stream_response([expected, {"type": "execution_finished"}]))

    async def run():
        async with httpx.AsyncClient(base_url="http://execution", transport=transport) as client:
            shell = RemoteShell(client)
            result = await shell.stream("raise ValueError('failed')")
        assert isinstance(result, Result)
        assert result.model_dump() == expected
        assert not shell.uncertain

    asyncio.run(run())


@pytest.mark.parametrize("fail_first", [False, True])
def test_agent_reuses_lazy_instructions_after_successful_inventory_fetch(fail_first):
    instructions = []
    inventory_requests = []
    executions = []

    def respond(_messages, info):
        instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart("done()")], finish_reason="stop")

    def handle(request):
        if request.url.path == "/libraries":
            inventory_requests.append(request)
            if fail_first and len(inventory_requests) == 1:
                return httpx.Response(503)
            return httpx.Response(200, json=[{"name": "example", "version": str(len(inventory_requests))}])
        executions.append(json.loads(request.content)["code"])
        return stream_response(
            [{"type": "result", "stdout": "Finished\n", "stderr": "", "exception": None, "done": True}, {"type": "execution_finished"}]
        )

    async def run():
        async with httpx.AsyncClient(base_url="http://execution", transport=httpx.MockTransport(handle)) as client:
            agent = Agent(FunctionModel(respond), shell=RemoteShell(client))
            assert not inventory_requests
            if fail_first:
                with pytest.raises(httpx.HTTPStatusError):
                    await agent.arun("First attempt")
                assert not instructions
                assert not executions
            assert await agent.arun("First task") == "Finished\n"
            assert await agent.arun("Second task") == "Finished\n"

        expected = (
            INSTRUCTIONS
            + "\nInstalled packages in the execution environment (distribution names may differ from import names):\n"
            + json.dumps([{"name": "example", "version": "2" if fail_first else "1"}])
        )
        assert instructions == [expected, expected]
        assert len(inventory_requests) == (2 if fail_first else 1)
        assert executions == ["done()", "done()"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "sequence", ["empty", "missing_finish", "missing_result", "duplicate_result", "after_finish", "invalid_result", "widget", "a2ui"]
)
def test_invalid_stream_blocks_subsequent_execution(sequence):
    result = {"type": "result", "stdout": "", "stderr": "", "exception": None, "done": True}
    finished = {"type": "execution_finished"}
    events = {
        "empty": [],
        "missing_finish": [result],
        "missing_result": [finished],
        "duplicate_result": [result, result, finished],
        "after_finish": [result, finished, result],
        "invalid_result": [{**result, "done": "true"}, finished],
        "widget": [{"type": "widget", "surface_id": "sales", "action": "pin"}, result, finished],
        "a2ui": [{"type": "a2ui", "messages": []}, result, finished],
    }[sequence]
    requests = []

    def handle(request):
        requests.append((request.method, request.url.path))
        return stream_response(events)

    transport = httpx.MockTransport(handle)

    async def run():
        async with httpx.AsyncClient(base_url="http://execution", transport=transport) as client:
            shell = RemoteShell(client)
            with pytest.raises(RuntimeError, match="Execution was not confirmed"):
                await shell.stream("done()")
            assert shell.uncertain
            with pytest.raises(RuntimeError, match="Execution state is uncertain"):
                await shell.stream("done()")
        assert requests == [("POST", "/execute/stream")]

    asyncio.run(run())


def test_agent_preserves_feedback_shape_and_combines_outputs():
    results = [
        {"type": "result", "stdout": "before\n", "stderr": "warning\n", "exception": "ValueError: failed", "done": False},
        {"type": "result", "stdout": "finished\n", "stderr": "", "exception": None, "done": True},
    ]
    responses = iter(results)
    prompts = []

    def respond(messages, _info):
        prompts.append(messages[-1].parts[0].content)
        return ModelResponse(parts=[TextPart("done()")], finish_reason="stop")

    def handle(request):
        if request.url.path == "/libraries":
            return httpx.Response(200, json=[])
        return stream_response([next(responses), {"type": "execution_finished"}])

    async def run():
        async with httpx.AsyncClient(base_url="http://execution", transport=httpx.MockTransport(handle)) as client:
            agent = Agent(FunctionModel(respond), shell=RemoteShell(client))
            output = await agent.arun("Finish the task")
        assert output == "before\nwarning\nValueError: failed\nfinished\n"
        assert json.loads(prompts[1]) == {"stdout": "before\n", "stderr": "warning\n", "exception": "ValueError: failed", "done": False}
        assert json.loads(agent.pending_feedback) == {"stdout": "finished\n", "stderr": "", "exception": None, "done": True}

    asyncio.run(run())


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        pytest.param([("A" * 6000, "B" * 800, None)], "A" * 3200 + "B" * 800, id="single-step-overflow"),
        pytest.param(
            [("A" * 4500, "B" * 1500, "error:" + "C" * 1500), ("D" * 2500, "E" * 800, None)],
            "C" * 699 + "\n" + "D" * 2500 + "E" * 800,
            id="multi-step-overflow",
        ),
        pytest.param([("A" * 4500, "", None), ("", "", None)], "A" * 3999 + "\n", id="trailing-separator"),
        pytest.param([("", "", None)], "Task completed without printed output.", id="no-output"),
        pytest.param([("", "", None), ("", "", None)], "\n", id="empty-step-separator"),
    ],
)
def test_agent_returns_bounded_summary_without_truncating_model_feedback(outputs, expected):
    results = [
        {"type": "result", "stdout": stdout, "stderr": stderr, "exception": exception, "done": index == len(outputs) - 1}
        for index, (stdout, stderr, exception) in enumerate(outputs)
    ]
    responses = iter(results)
    prompts = []

    def respond(messages, _info):
        prompts.append(messages[-1].parts[0].content)
        return ModelResponse(parts=[TextPart("done()")], finish_reason="stop")

    def handle(request):
        if request.url.path == "/libraries":
            return httpx.Response(200, json=[])
        return stream_response([next(responses), {"type": "execution_finished"}])

    async def run():
        async with httpx.AsyncClient(base_url="http://execution", transport=httpx.MockTransport(handle)) as client:
            agent = Agent(FunctionModel(respond), shell=RemoteShell(client))
            summary = await agent.arun("Finish the task")
        assert summary == expected
        assert len(summary) <= 4000
        for prompt, result in zip(prompts[1:], results[:-1], strict=True):
            assert json.loads(prompt) == {key: value for key, value in result.items() if key != "type"}
        assert json.loads(agent.pending_feedback) == {key: value for key, value in results[-1].items() if key != "type"}

    asyncio.run(run())
