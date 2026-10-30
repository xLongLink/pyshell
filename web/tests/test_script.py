import json
import httpx
import pytest
import asyncio
from pydantic_ai import Agent as PydanticAgent
from unittest.mock import Mock, AsyncMock, patch
from web.src.shell import RemoteShell
from importlib.metadata import version
from web.src.agents.script import Agent, stream_response


def test_installed_libraries_are_in_model_context(model_for, remote_shell):
    requests = []
    packages = [{"name": "custom-package", "version": "1.2.3"}]
    model = model_for(["print('working')"], ["done()"], ["done()"], ["done()"], requests=requests)
    with patch.object(remote_shell, "libraries", return_value=packages) as libraries:
        agent = Agent(model, shell=remote_shell)
        agent.run("work")
        agent.run("continue")
        libraries.assert_called_once_with()

        # Reset invalidates the inventory cache for the next model turn.
        agent.reset()
        agent.run("start again")
        assert libraries.call_count == 2
    assert len(requests) == 4
    assert all(json.dumps(packages) in info.instructions for _, info in requests)


def test_library_discovery_failure_prevents_model_request(model_for, remote_shell):
    requests = []
    agent = Agent(model_for(["done()"], requests=requests), shell=remote_shell)
    with patch.object(remote_shell, "libraries", side_effect=httpx.ConnectError("unavailable")):
        with pytest.raises(httpx.ConnectError, match="unavailable"):
            agent.run("work")

    # Failed discovery must not commit a model request or conversation history.
    assert requests == []
    assert agent.message_history == []

    # A discovery failure must leave the agent able to retry the same turn.
    agent.run("retry")
    assert len(requests) == 1


def test_remote_libraries(remote_shell):
    assert {"name": "fastapi", "version": version("fastapi")} in remote_shell.libraries()


def test_remote_execution_and_reset(remote_shell):
    assert remote_shell.execute("value = 42")["exception"] is None
    assert remote_shell.execute("print(value)")["stdout"] == "42\n"
    remote_shell.reset()
    assert "NameError" in remote_shell.execute("print(value)")["exception"]


def test_execution_transport_failure_is_not_retried(monkeypatch):
    calls = []

    def post(*args, **kwargs):
        calls.append(args)
        raise httpx.ReadError("disconnected")

    monkeypatch.setattr("web.src.shell.httpx.post", post)
    with pytest.raises(RuntimeError, match="may already have run"):
        RemoteShell().execute("value = 42")
    assert len(calls) == 1


def test_streams_code_to_callback_without_console_logs(model_for, capsys):
    agent = PydanticAgent(model_for(["print(", "'hello')"]))
    chunks = []
    code, history = asyncio.run(stream_response(agent, "say hello", [], on_code=chunks.append))
    output = capsys.readouterr()
    assert "".join(chunks) == "print('hello')"
    assert output.out == ""
    assert output.err == ""
    assert code == "print('hello')"
    assert history[-1].parts[0].content == code


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("with_callback", [False, True])
def test_agent_event_lifecycle_with_optional_callback(model_for, remote_shell, streaming, with_callback):
    code = "print('confirmed')\ndone()"
    agent = Agent(model_for([code]), shell=remote_shell)
    events = []

    async def run():
        if streaming:
            await remote_shell.open()
        try:
            result = await agent.arun("finish", events.append if with_callback else None)
            assert "confirmed" in result
        finally:
            await remote_shell.close()

    asyncio.run(run())
    if not with_callback:
        assert events == []
        return
    assert events[0] == {"type": "code_start"}
    assert "".join(event["text"] for event in events if event["type"] == "code_delta") == code
    types = [event["type"] for event in events]
    assert types.index("code_start") < types.index("code_delta") < types.index("execution_start") < types.index("result")
    assert types.count("result") == 1
    assert "execution_finished" not in types
    assert events[-1] == {"type": "result", "stdout": "confirmed\n", "stderr": "", "exception": None, "done": True}


def test_synchronous_ui_feedback_keeps_content_batches_around_placement(model_for, remote_shell):
    code = """chart = ui.plot([{"type": "bar", "x": ["Jan"], "y": [1]}], surface_id="sales")
chart.pin()
chart.update(data=[{"type": "bar", "x": ["Jan"], "y": [2]}])
done()
"""
    events = []
    Agent(model_for([code]), shell=remote_shell).run("plot and pin sales", events.append)
    feedback = [event for event in events if event["type"] in {"result", "a2ui", "widget"}]
    assert [event["type"] for event in feedback] == ["result", "a2ui", "widget", "a2ui"]
    assert "createSurface" in feedback[1]["messages"][0]
    assert feedback[2] == {"type": "widget", "surface_id": "sales", "action": "pin"}
    assert feedback[3]["messages"][0]["updateComponents"]["components"][0]["data"][0]["y"] == [2]


def test_stream_failure_propagates_without_timing(model_for, capsys):
    agent = PydanticAgent(model_for(["value = 42", ConnectionError("disconnected")]))
    with pytest.raises(ConnectionError, match="disconnected"):
        asyncio.run(stream_response(agent, "set value", []))
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_reset_clears_remote_session_and_conversation(model_for, remote_shell):
    agent = Agent(model_for(["done()"]), shell=remote_shell)
    agent.run("import math")
    remote_shell.execute("import math; value = 42")
    assert agent.message_history
    agent.reset()
    assert "NameError" in remote_shell.execute("print(math)")["exception"]
    assert "NameError" in remote_shell.execute("print(value)")["exception"]
    assert agent.message_history == []
    assert agent.pending_feedback is None


def test_model_imports_persist_between_requests(model_for, remote_shell, capsys):
    requests = []
    model = model_for(["import math\ndone()"], ["print(math.sqrt(16))\ndone()"], requests=requests)
    agent = Agent(model, shell=remote_shell)
    agent.run("import math")
    agent.run("square root of 16")
    assert json.loads(agent.pending_feedback)["stdout"] == "4.0\n"
    assert "4.0" in capsys.readouterr().out
    assert any("import math" in str(message) for message in requests[1][0][:-1])
    assert '"done": true' in str(requests[1][0][-1])


def test_requests_priority_processing(model_for, remote_shell):
    requests = []
    model = model_for(["done()"], requests=requests)
    Agent(model, service_tier="priority", shell=remote_shell).run("finish")
    settings = requests[0][1].model_settings
    assert settings["openai_service_tier"] == "priority"
    assert settings["openai_reasoning_effort"] == "low"


def test_agent_does_not_log_code_or_results(model_for, capsys):
    shell = Mock()
    shell.client = None
    shell.libraries.return_value = []
    shell.execute.return_value = {"stdout": "result\n", "stderr": "warning\n", "exception": None, "done": True, "ui": []}
    agent = Agent(model_for(["print('result')\ndone()"]), shell=shell)

    agent.run("work")

    # Execution output belongs to the result, not the web server's logs.
    shell.execute.assert_called_once_with("print('result')\ndone()")
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_continues_until_done_and_sends_execution_feedback(model_for, remote_shell):
    requests = []
    model = model_for(["print(1 / 0)"], ["print('fixed')\ndone()"], requests=requests)
    agent = Agent(model, shell=remote_shell)
    agent.run("calculate")
    feedback = json.loads(requests[1][0][-1].parts[0].content)
    assert "ZeroDivisionError" in feedback["exception"]
    assert json.loads(agent.pending_feedback)["stdout"] == "fixed\n"


def test_failed_stream_never_executes_code(model_for, remote_shell):
    model = model_for(["value = 42\n", ConnectionError("disconnected")])
    agent = Agent(model, shell=remote_shell)
    agent.pending_feedback = "previous feedback"
    with pytest.raises(ConnectionError, match="disconnected"):
        agent.run("set value")

    # A failed stream must preserve feedback without committing code or history.
    assert "NameError" in remote_shell.execute("print(value)")["exception"]
    assert agent.pending_feedback.startswith("previous feedback\n")
    assert '"executed": false' in agent.pending_feedback
    assert agent.message_history == []


def test_incomplete_response_never_executes(model_for, remote_shell):
    agent = Agent(model_for(["value = 42\n"]), shell=remote_shell)
    stream = Mock()
    stream.stream_text.return_value = iter(["value = 42\n"])
    stream.get_output = AsyncMock(return_value="value = 42\n")
    stream.response.finish_reason = "length"

    # A truncated response must not reach the execution API.
    async def chunks(*args, **kwargs):
        yield "value = 42\n"

    stream.stream_text = chunks
    with patch.object(agent.agent, "run_stream") as run:
        run.return_value.__aenter__ = AsyncMock(return_value=stream)
        run.return_value.__aexit__ = AsyncMock(return_value=False)

        # A length-limited response must leave the Python namespace untouched.
        with pytest.raises(RuntimeError, match="Model response length"):
            agent.run("set value")
    assert "NameError" in remote_shell.execute("print(value)")["exception"]
    assert agent.message_history == []
