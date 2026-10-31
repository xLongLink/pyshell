import json
import httpx
import pytest
import asyncio
from types import SimpleNamespace
from api.main import create_app
from unittest.mock import AsyncMock
from sample.src.shell import RemoteShell
from importlib.metadata import version
from pydantic_ai.messages import TextPart, ModelResponse
from sample.src.agents.script import Agent
from pydantic_ai.models.function import FunctionModel


def test_continues_until_done_and_preserves_execution_context():
    requests = []
    responses = iter(["import math\nvalue = 42\nprint(1 / 0)", "print(math.sqrt(value + 22))\ndone()", "print(value)\ndone()"])

    def respond(messages, info):
        requests.append((messages, info))
        return ModelResponse(parts=[TextPart(next(responses))], finish_reason="stop")

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(execution_token="test-token")),
            base_url="http://testserver",
            headers={"authorization": "Bearer test-token"},
        ) as client:
            shell = RemoteShell(client)
            libraries = await shell.inventory()
            assert {"name": "fastapi", "version": version("fastapi")} in libraries
            assert all(library["name"] and library["version"] for library in libraries)
            agent = Agent(FunctionModel(respond), shell=shell)
            assert "8.0\n" in await agent.arun("calculate")
            feedback = json.loads(requests[1][0][-1].parts[0].content)
            assert set(feedback) == {"stdout", "stderr", "exception", "done"}
            assert "ZeroDivisionError" in feedback["exception"]
            assert json.loads(agent.pending_feedback)["stdout"] == "8.0\n"
            assert await agent.arun("inspect value") == "42\n"
            assert "import math" in str(requests[2][0])
            assert "Installed packages" in requests[0][1].instructions

    asyncio.run(run())
    assert len(requests) == 3


def test_execution_transport_failure_is_not_retried():
    calls = []

    def handle(request):
        if request.url.path == "/libraries":
            return httpx.Response(200, json=[])
        calls.append(request)
        raise httpx.ReadError("disconnected")

    def respond(messages, info):
        return ModelResponse(parts=[TextPart("value = 42\ndone()")], finish_reason="stop")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://testserver") as client:
            shell = RemoteShell(client)
            agent = Agent(FunctionModel(respond), shell=shell)
            with pytest.raises(RuntimeError, match="may already have run"):
                await agent.arun("set value")
            assert agent.message_history
            assert "Do not replay" in agent.pending_feedback
            assert shell.uncertain is True
            with pytest.raises(RuntimeError, match="uncertain"):
                await shell.stream("value = 42")

    asyncio.run(run())
    assert len(calls) == 1


@pytest.mark.parametrize("error_type", [ConnectionError, asyncio.CancelledError])
def test_failed_model_request_never_executes_code(capsys, error_type):
    calls = []

    async def respond(messages, info):
        calls.append(messages)
        raise error_type("disconnected")

    shell = SimpleNamespace(inventory=AsyncMock(return_value=[]), stream=AsyncMock())
    agent = Agent(FunctionModel(respond), shell=shell)
    agent.pending_feedback = "previous feedback"
    with pytest.raises(error_type, match="disconnected"):
        asyncio.run(agent.arun("set value"))
    shell.stream.assert_not_awaited()
    assert len(calls) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""
    assert agent.pending_feedback.startswith("previous feedback\n")
    assert '"executed": false' in agent.pending_feedback
    assert '"model_error":' in agent.pending_feedback
    assert agent.message_history == []


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", None])
def test_incomplete_response_never_executes(finish_reason):
    requests = []

    def respond(messages, info):
        requests.append(messages)
        return ModelResponse(parts=[TextPart("value = 42\n")], finish_reason=finish_reason)

    shell = SimpleNamespace(inventory=AsyncMock(return_value=[]), stream=AsyncMock())
    agent = Agent(FunctionModel(respond), shell=shell)
    with pytest.raises(RuntimeError, match=f"Model response {finish_reason}"):
        asyncio.run(agent.arun("set value"))
    shell.stream.assert_not_awaited()
    assert agent.message_history == []
    assert len(requests) == 1


def test_library_discovery_failure_prevents_model_request_and_retries():
    requests = []
    instructions = []

    def respond(messages, info):
        requests.append(messages)
        instructions.append(info.instructions)
        return ModelResponse(parts=[TextPart("done()")], finish_reason="stop")

    libraries = [{"name": "example", "version": "1.0"}]
    shell = SimpleNamespace(
        inventory=AsyncMock(side_effect=[httpx.ConnectError("unavailable"), libraries]),
        stream=AsyncMock(return_value={"stdout": "", "stderr": "", "exception": None, "done": True}),
    )
    agent = Agent(FunctionModel(respond), shell=shell)
    shell.inventory.assert_not_awaited()
    agent.pending_feedback = "previous feedback"

    async def run():
        with pytest.raises(httpx.ConnectError, match="unavailable"):
            await agent.arun("work")
        assert requests == []
        assert agent.message_history == []
        shell.stream.assert_not_awaited()
        feedback = agent.pending_feedback.splitlines()
        assert feedback[0] == "previous feedback"
        assert json.loads(feedback[1]) == {"model_error": "ConnectError: unavailable", "executed": False}

        await agent.arun("retry")
        assert "previous feedback" in requests[0][-1].parts[0].content
        assert "ConnectError: unavailable" in requests[0][-1].parts[0].content
        await agent.arun("next turn")
        assert json.dumps(libraries) in instructions[0]
        assert instructions[1] == instructions[0]

    asyncio.run(run())
    assert shell.inventory.await_count == 2
    assert shell.stream.await_count == 2
    assert len(requests) == 2
