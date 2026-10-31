import json
import httpx
import pytest
import asyncio
from sample.src.shell import RemoteShell
from sample.src.events import Result
from pydantic_ai.messages import TextPart, ModelResponse
from sample.src.agents.script import Agent
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
    transport = httpx.MockTransport(lambda _request: stream_response(events))

    async def run():
        async with httpx.AsyncClient(base_url="http://execution", transport=transport) as client:
            shell = RemoteShell(client)
            with pytest.raises(RuntimeError, match="Execution was not confirmed"):
                await shell.stream("done()")
            assert shell.uncertain
            with pytest.raises(RuntimeError, match="Execution state is uncertain"):
                await shell.stream("done()")

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
