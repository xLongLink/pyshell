import json
from unittest.mock import Mock

import pytest

from script import Agent


def test_reset_clears_remote_session_and_conversation(remote_shell):
    agent = Agent(Mock(), "test-model", shell=remote_shell)
    remote_shell.execute("import math; value = 42")
    agent.previous_response_id = "old"
    agent.pending_feedback = "old feedback"
    agent.reset()
    assert "NameError" in remote_shell.execute("print(math)")["exception"]
    assert "NameError" in remote_shell.execute("print(value)")["exception"]
    assert agent.previous_response_id is None
    assert agent.pending_feedback is None


def test_model_imports_persist_between_requests(stream_for, remote_shell, capsys):
    client = Mock()
    client.responses.stream.side_effect = [
        stream_for(["import math\ndone()"]),
        stream_for(["print(math.sqrt(16))\ndone()"]),
    ]
    agent = Agent(client, "test-model", shell=remote_shell)
    agent.run("import math")
    agent.run("square root of 16")
    assert json.loads(agent.pending_feedback)["stdout"] == "4.0\n"
    assert "4.0" in capsys.readouterr().out


def test_requests_priority_processing(stream_for, remote_shell):
    client = Mock()
    client.responses.stream.return_value = stream_for(["done()"])
    Agent(client, "test-model", service_tier="priority", shell=remote_shell).run("finish")
    assert client.responses.stream.call_args.kwargs["service_tier"] == "priority"
    assert client.responses.stream.call_args.kwargs["reasoning"] == {"effort": "low"}


def test_continues_until_done_and_sends_execution_feedback(stream_for, remote_shell):
    client = Mock()
    client.responses.stream.side_effect = [
        stream_for(["print(1 / 0)"]), stream_for(["print('fixed')\ndone()"]),
    ]
    agent = Agent(client, "test-model", shell=remote_shell)
    agent.run("calculate")
    feedback = json.loads(client.responses.stream.call_args_list[1].kwargs["input"][-1]["content"])
    assert "ZeroDivisionError" in feedback["exception"]
    assert json.loads(agent.pending_feedback)["stdout"] == "fixed\n"


def test_failed_stream_never_executes_code(stream_for, remote_shell):
    client = Mock()
    stream = stream_for(["value = 42\n"])
    stream.get_final_response.side_effect = ConnectionError("disconnected")
    client.responses.stream.return_value = stream
    agent = Agent(client, "test-model", shell=remote_shell)
    agent.pending_feedback = "previous feedback"
    with pytest.raises(ConnectionError, match="disconnected"):
        agent.run("set value")
    assert "NameError" in remote_shell.execute("print(value)")["exception"]
    assert agent.pending_feedback.startswith("previous feedback\n")
    assert '"executed": false' in agent.pending_feedback


def test_incomplete_response_never_executes(stream_for, remote_shell):
    client = Mock()
    client.responses.stream.return_value = stream_for(["value = 42\n"], status="incomplete")
    agent = Agent(client, "test-model", shell=remote_shell)
    with pytest.raises(RuntimeError, match="Model response incomplete"):
        agent.run("set value")
    assert "NameError" in remote_shell.execute("print(value)")["exception"]
    assert agent.previous_response_id is None
