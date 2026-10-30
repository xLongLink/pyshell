import json
import httpx
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from web.src.shell import RemoteShell
from pydantic_ai.usage import UsageLimits
from importlib.metadata import version
from web.src.agents.script import Agent
from pydantic_ai.exceptions import UsageLimitExceeded


def test_installed_libraries_are_in_model_context(model_for, remote_shell):
    requests = []
    packages = [{"name": "custom-package", "version": "1.2.3"}]
    model = model_for(["print('working')"], ["done()"], ["done()"], requests=requests)

    async def run():
        async with remote_shell.client:
            with patch.object(remote_shell, "inventory", new_callable=AsyncMock, return_value=packages) as inventory:
                agent = Agent(model, shell=remote_shell)
                await agent.arun("work")
                await agent.arun("continue")
                inventory.assert_awaited_once_with()

    asyncio.run(run())

    assert len(requests) == 3
    assert all(json.dumps(packages) in info.instructions for _, info in requests)


def test_library_discovery_failure_prevents_model_request(model_for, remote_shell):
    requests = []
    agent = Agent(model_for(["done()"], requests=requests), shell=remote_shell)

    async def run():
        async with remote_shell.client:
            with patch.object(remote_shell, "inventory", new_callable=AsyncMock, side_effect=httpx.ConnectError("unavailable")):
                with pytest.raises(httpx.ConnectError, match="unavailable"):
                    await agent.arun("work")

            assert requests == []
            assert agent.message_history == []
            await agent.arun("retry")

    asyncio.run(run())
    assert len(requests) == 1


def test_remote_libraries(remote_shell):
    async def run():
        async with remote_shell.client:
            assert {"name": "fastapi", "version": version("fastapi")} in await remote_shell.inventory()

    asyncio.run(run())


def test_remote_and_sync_clients_share_the_execution_namespace(api_client, remote_shell):
    assert api_client.post("/", json={"code": "shared_value = 42"}).json()["exception"] is None

    async def run():
        async with remote_shell.client:
            result = await remote_shell.stream("print(shared_value)\nshared_value += 1\ndone()", lambda _event: None)
            assert result["stdout"] == "42\n"
            assert result["done"] is True

    asyncio.run(run())
    assert api_client.post("/", json={"code": "print(shared_value)"}).json()["stdout"] == "43\n"


def test_execution_transport_failure_is_not_retried(model_for):
    calls = []

    def handle(request):
        if request.url.path == "/libraries":
            return httpx.Response(200, json=[])
        calls.append(request)
        raise httpx.ReadError("disconnected")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://testserver") as client:
            shell = RemoteShell(client)
            agent = Agent(model_for(["value = 42\ndone()"]), shell=shell)
            with pytest.raises(RuntimeError, match="may already have run"):
                await agent.arun("set value")
            assert agent.message_history
            assert "Do not replay" in agent.pending_feedback
            assert shell.uncertain is True
            with pytest.raises(RuntimeError, match="uncertain"):
                await shell.stream("value = 42", lambda _event: None)

    asyncio.run(run())
    assert len(calls) == 1


@pytest.mark.parametrize("with_callback", [False, True])
def test_agent_event_lifecycle_with_optional_callback(model_for, remote_shell, with_callback):
    code = "print('confirmed')\ndone()"
    agent = Agent(model_for([code]), shell=remote_shell)
    events = []

    async def run():
        async with remote_shell.client:
            result = await agent.arun("finish", events.append if with_callback else None)
            assert "confirmed" in result

    asyncio.run(run())
    if not with_callback:
        assert events == []
        return
    assert events[0] == {"type": "code_start"}
    types = [event["type"] for event in events]
    assert types.index("code_start") < types.index("execution_start") < types.index("result")
    assert "code_delta" not in types
    assert types.count("result") == 1
    assert "execution_finished" not in types
    assert events[-1] == {"type": "result", "stdout": "confirmed\n", "stderr": "", "exception": None, "done": True}


def test_ui_feedback_keeps_content_batches_around_placement(model_for, remote_shell):
    code = """chart = ui.plot([{"type": "bar", "x": ["Jan"], "y": [1]}], surface_id="sales")
chart.pin()
chart.update(data=[{"type": "bar", "x": ["Jan"], "y": [2]}])
done()
"""
    events = []

    async def run():
        async with remote_shell.client:
            await Agent(model_for([code]), shell=remote_shell).arun("plot and pin sales", events.append)

    asyncio.run(run())
    feedback = [event for event in events if event["type"] in {"result", "a2ui", "widget"}]
    assert [event["type"] for event in feedback] == ["a2ui", "widget", "a2ui", "result"]
    assert "createSurface" in feedback[0]["messages"][0]
    assert feedback[1] == {"type": "widget", "surface_id": "sales", "action": "pin"}
    assert feedback[2]["messages"][0]["updateComponents"]["components"][0]["data"][0]["y"] == [2]


def test_generation_shares_usage_limits_across_requests(model_for, remote_shell, api_client):
    # Arrange
    requests = []
    agent = Agent(model_for(["steps = 1"], ["steps += 1\ndone()"], requests=requests), shell=remote_shell)
    limits = UsageLimits(request_limit=1)

    async def run():
        async with remote_shell.client:
            # Act / Assert
            with patch("web.src.agents.script.UsageLimits", return_value=limits):
                with pytest.raises(UsageLimitExceeded, match="request_limit"):
                    await agent.arun("finish")

    asyncio.run(run())
    # Assert
    assert len(requests) == 1
    assert api_client.post("/", json={"code": "print(steps)"}).json()["stdout"] == "1\n"
    assert json.loads(agent.pending_feedback)["executed"] is False


def test_model_imports_persist_between_requests(model_for, remote_shell):
    requests = []
    model = model_for(["import math\ndone()"], ["print(math.sqrt(16))\ndone()"], requests=requests)
    agent = Agent(model, shell=remote_shell)

    async def run():
        async with remote_shell.client:
            await agent.arun("import math")
            assert await agent.arun("square root of 16") == "4.0\n"

    asyncio.run(run())
    assert json.loads(agent.pending_feedback)["stdout"] == "4.0\n"
    assert any("import math" in str(message) for message in requests[1][0][:-1])
    assert '"done": true' in str(requests[1][0][-1])


def test_agent_does_not_log_code_or_results(model_for, remote_shell, capsys):
    code = "import sys\nprint('result')\nprint('warning', file=sys.stderr)\ndone()"
    agent = Agent(model_for([code]), shell=remote_shell)

    async def run():
        async with remote_shell.client:
            result = {"stdout": "result\n", "stderr": "warning\n", "exception": None, "done": True}
            with patch.object(remote_shell, "stream", new_callable=AsyncMock, return_value=result) as execute:
                assert await agent.arun("work") == "result\nwarning\n"
                assert execute.await_count == 1
                assert execute.await_args.args[0] == code

    asyncio.run(run())
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""
    assert agent.message_history[-1].parts[0].content == code


def test_continues_until_done_and_sends_execution_feedback(model_for, remote_shell):
    requests = []
    model = model_for(["print(1 / 0)"], ["print('fixed')\ndone()"], requests=requests)
    agent = Agent(model, shell=remote_shell)

    async def run():
        async with remote_shell.client:
            await agent.arun("calculate")

    asyncio.run(run())
    feedback = json.loads(requests[1][0][-1].parts[0].content)
    assert set(feedback) == {"stdout", "stderr", "exception", "done"}
    assert "ZeroDivisionError" in feedback["exception"]
    assert json.loads(agent.pending_feedback)["stdout"] == "fixed\n"


@pytest.mark.parametrize("error_type", [ConnectionError, asyncio.CancelledError])
def test_failed_model_request_never_executes_code(model_for, remote_shell, api_client, capsys, error_type):
    model = model_for(["value = 42\n", error_type("disconnected")])
    agent = Agent(model, shell=remote_shell)
    agent.pending_feedback = "previous feedback"

    async def run():
        async with remote_shell.client:
            with patch.object(remote_shell, "stream", wraps=remote_shell.stream) as execute:
                with pytest.raises(error_type, match="disconnected"):
                    await agent.arun("set value")
                execute.assert_not_awaited()

    asyncio.run(run())
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""
    assert "NameError" in api_client.post("/", json={"code": "print(value)"}).json()["exception"]
    assert agent.pending_feedback.startswith("previous feedback\n")
    assert '"executed": false' in agent.pending_feedback
    assert '"model_error":' in agent.pending_feedback
    assert agent.message_history == []


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", None])
def test_incomplete_response_never_executes(model_for, remote_shell, api_client, finish_reason):
    requests = []
    agent = Agent(model_for(["value = 42\n"], finish_reason=finish_reason, requests=requests), shell=remote_shell)

    async def run():
        async with remote_shell.client:
            with patch.object(remote_shell, "stream") as execute:
                with pytest.raises(RuntimeError, match=f"Model response {finish_reason}"):
                    await agent.arun("set value")
                execute.assert_not_awaited()

    asyncio.run(run())
    assert "NameError" in api_client.post("/", json={"code": "print(value)"}).json()["exception"]
    assert agent.message_history == []
    assert len(requests) == 1
