import json
import httpx
import pytest
import asyncio
from contextlib import asynccontextmanager
from pydantic_ai import Agent as PydanticAgent
from unittest.mock import Mock, AsyncMock, patch
from web.src.shell import RemoteShell
from importlib.metadata import version
from web.src.agents.script import Agent, stream_response


@asynccontextmanager
async def opened_shell(shell):
    await shell.open()
    try:
        yield shell
    finally:
        await shell.close()


@pytest.mark.parametrize("inventory_loaded", [False, True])
def test_closed_transport_prevents_requests(model_for, remote_shell, inventory_loaded):
    requests = []
    agent = Agent(model_for(["done()"], requests=requests), shell=remote_shell)
    events = []

    async def run():
        if inventory_loaded:
            async with opened_shell(remote_shell):
                await agent.arun("finish")

        assert remote_shell.client is None
        history = list(agent.message_history)
        feedback = agent.pending_feedback
        request_count = len(requests)
        with (
            patch.object(remote_shell, "inventory") as inventory,
            patch.object(remote_shell, "stream") as execute,
            patch.object(agent.agent, "run_stream") as response,
        ):
            with pytest.raises(RuntimeError, match="Execution transport is closed"):
                await agent.arun("work", events.append)
            inventory.assert_not_called()
            response.assert_not_called()
            execute.assert_not_called()

        assert len(requests) == request_count == int(inventory_loaded)
        assert agent.message_history == history
        assert agent.pending_feedback == feedback
        assert events == []

    asyncio.run(run())


def test_installed_libraries_are_in_model_context(model_for, remote_shell):
    requests = []
    packages = [{"name": "custom-package", "version": "1.2.3"}]
    model = model_for(["print('working')"], ["done()"], ["done()"], requests=requests)

    async def run():
        async with opened_shell(remote_shell):
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
        async with opened_shell(remote_shell):
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
        async with opened_shell(remote_shell):
            assert {"name": "fastapi", "version": version("fastapi")} in await remote_shell.inventory()

    asyncio.run(run())


def test_remote_execution_and_reset(api_client):
    assert api_client.post("/", json={"code": "value = 42"}).json()["exception"] is None
    assert api_client.post("/", json={"code": "print(value)"}).json()["stdout"] == "42\n"
    assert api_client.post("/reset").is_success
    assert "NameError" in api_client.post("/", json={"code": "print(value)"}).json()["exception"]


def test_execution_transport_failure_is_not_retried(model_for):
    calls = []

    def handle(request):
        if request.url.path == "/libraries":
            return httpx.Response(200, json=[])
        calls.append(request)
        raise httpx.ReadError("disconnected")

    async def run():
        shell = RemoteShell()
        shell.client = httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url=shell.url)
        agent = Agent(model_for(["value = 42\ndone()"]), shell=shell)
        try:
            with pytest.raises(RuntimeError, match="may already have run"):
                await agent.arun("set value")
            assert agent.message_history
            assert "Do not replay" in agent.pending_feedback
            assert shell.uncertain is True
            with pytest.raises(RuntimeError, match="uncertain"):
                await shell.stream("value = 42", lambda _event: None)
        finally:
            await shell.close()

    asyncio.run(run())
    assert len(calls) == 1


def test_consumes_complete_model_response_without_console_logs(model_for, capsys):
    agent = PydanticAgent(model_for(["print(", "'hello')"]))
    code, history = asyncio.run(stream_response(agent, "say hello", []))
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""
    assert code == "print('hello')"
    assert history[-1].parts[0].content == code


@pytest.mark.parametrize("with_callback", [False, True])
def test_agent_event_lifecycle_with_optional_callback(model_for, remote_shell, with_callback):
    code = "print('confirmed')\ndone()"
    agent = Agent(model_for([code]), shell=remote_shell)
    events = []

    async def run():
        async with opened_shell(remote_shell):
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
        async with opened_shell(remote_shell):
            await Agent(model_for([code]), shell=remote_shell).arun("plot and pin sales", events.append)

    asyncio.run(run())
    feedback = [event for event in events if event["type"] in {"result", "a2ui", "widget"}]
    assert [event["type"] for event in feedback] == ["a2ui", "widget", "a2ui", "result"]
    assert "createSurface" in feedback[0]["messages"][0]
    assert feedback[1] == {"type": "widget", "surface_id": "sales", "action": "pin"}
    assert feedback[2]["messages"][0]["updateComponents"]["components"][0]["data"][0]["y"] == [2]


def test_stream_failure_propagates_without_timing(model_for, capsys):
    agent = PydanticAgent(model_for(["value = 42", ConnectionError("disconnected")]))
    with pytest.raises(ConnectionError, match="disconnected"):
        asyncio.run(stream_response(agent, "set value", []))
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_model_imports_persist_between_requests(model_for, remote_shell):
    requests = []
    model = model_for(["import math\ndone()"], ["print(math.sqrt(16))\ndone()"], requests=requests)
    agent = Agent(model, shell=remote_shell)

    async def run():
        async with opened_shell(remote_shell):
            await agent.arun("import math")
            assert await agent.arun("square root of 16") == "4.0\n"

    asyncio.run(run())
    assert json.loads(agent.pending_feedback)["stdout"] == "4.0\n"
    assert any("import math" in str(message) for message in requests[1][0][:-1])
    assert '"done": true' in str(requests[1][0][-1])


def test_requests_priority_processing(model_for, remote_shell):
    requests = []
    model = model_for(["done()"], requests=requests)

    async def run():
        async with opened_shell(remote_shell):
            await Agent(model, service_tier="priority", shell=remote_shell).arun("finish")

    asyncio.run(run())
    settings = requests[0][1].model_settings
    assert settings["openai_service_tier"] == "priority"
    assert settings["openai_reasoning_effort"] == "low"


def test_agent_does_not_log_code_or_results(model_for, remote_shell, capsys):
    code = "import sys\nprint('result')\nprint('warning', file=sys.stderr)\ndone()"
    agent = Agent(model_for([code]), shell=remote_shell)

    async def run():
        async with opened_shell(remote_shell):
            result = {"stdout": "result\n", "stderr": "warning\n", "exception": None, "done": True, "ui": []}
            with patch.object(remote_shell, "stream", new_callable=AsyncMock, return_value=result) as execute:
                assert await agent.arun("work") == "result\nwarning\n"
                assert execute.await_count == 1
                assert execute.await_args.args[0] == code

    asyncio.run(run())
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_continues_until_done_and_sends_execution_feedback(model_for, remote_shell):
    requests = []
    model = model_for(["print(1 / 0)"], ["print('fixed')\ndone()"], requests=requests)
    agent = Agent(model, shell=remote_shell)

    async def run():
        async with opened_shell(remote_shell):
            await agent.arun("calculate")

    asyncio.run(run())
    feedback = json.loads(requests[1][0][-1].parts[0].content)
    assert "ZeroDivisionError" in feedback["exception"]
    assert json.loads(agent.pending_feedback)["stdout"] == "fixed\n"


def test_failed_stream_never_executes_code(model_for, remote_shell, api_client):
    model = model_for(["value = 42\n", ConnectionError("disconnected")])
    agent = Agent(model, shell=remote_shell)
    agent.pending_feedback = "previous feedback"

    async def run():
        async with opened_shell(remote_shell):
            with patch.object(remote_shell, "stream", wraps=remote_shell.stream) as execute:
                with pytest.raises(ConnectionError, match="disconnected"):
                    await agent.arun("set value")
                execute.assert_not_awaited()

    asyncio.run(run())
    assert "NameError" in api_client.post("/", json={"code": "print(value)"}).json()["exception"]
    assert agent.pending_feedback.startswith("previous feedback\n")
    assert '"executed": false' in agent.pending_feedback
    assert agent.message_history == []


def test_incomplete_response_never_executes(model_for, remote_shell, api_client):
    agent = Agent(model_for(["value = 42\n"]), shell=remote_shell)
    stream = Mock()
    stream.get_output = AsyncMock(return_value="value = 42\n")
    stream.response.finish_reason = "length"

    async def run():
        async with opened_shell(remote_shell):
            with patch.object(agent.agent, "run_stream") as response, patch.object(remote_shell, "stream") as execute:
                response.return_value.__aenter__ = AsyncMock(return_value=stream)
                response.return_value.__aexit__ = AsyncMock(return_value=False)
                with pytest.raises(RuntimeError, match="Model response length"):
                    await agent.arun("set value")
                execute.assert_not_awaited()

    asyncio.run(run())
    assert "NameError" in api_client.post("/", json={"code": "print(value)"}).json()["exception"]
    assert agent.message_history == []
