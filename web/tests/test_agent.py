import json
import httpx
import pytest
from unittest.mock import Mock, patch
from web.src.script import Agent


def test_installed_libraries_are_in_model_context(model_for, remote_shell):
    requests = []
    packages = [{"name": "custom-package", "version": "1.2.3"}]
    model = model_for(["print('working')"], ["done()"], ["done()"], ["done()"], requests=requests)
    with patch.object(remote_shell, "libraries", return_value=packages) as libraries:
        agent = Agent(model, shell=remote_shell)
        agent.run("work")
        agent.run("continue")
        libraries.assert_called_once_with()
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
    assert requests == []
    assert agent.message_history == []
    agent.run("retry")
    assert len(requests) == 1


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
    shell.libraries.return_value = []
    shell.execute.return_value = {"stdout": "result\n", "stderr": "warning\n", "exception": None, "done": True, "ui": []}
    agent = Agent(model_for(["print('result')\ndone()"]), shell=shell)

    agent.run("work")

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
    stream.get_output.return_value = "value = 42\n"
    stream.response.finish_reason = "length"

    # A truncated response must not reach the execution API.
    with patch.object(agent.agent, "run_stream_sync") as run:
        run.return_value = stream
        with pytest.raises(RuntimeError, match="Model response length"):
            agent.run("set value")
    assert "NameError" in remote_shell.execute("print(value)")["exception"]
    assert agent.message_history == []
