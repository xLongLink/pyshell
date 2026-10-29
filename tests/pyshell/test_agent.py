import json
import sys
from unittest.mock import Mock, patch

import pytest

from pyshell.agent import Agent


def test_reset_clears_imports_variables_and_conversation():
    agent = Agent(Mock(), "test-model")
    agent.shell.execute_line("import math; value = math.sqrt(16)")
    assert agent.shell.namespace["value"] == 4
    agent.previous_response_id = "old-response"
    agent.pending_feedback = "old-feedback"

    agent.reset()

    # Reset both the Python namespace and the model conversation.
    assert "math" not in agent.shell.namespace
    assert "value" not in agent.shell.namespace
    assert callable(agent.shell.namespace["done"])
    assert agent.previous_response_id is None
    assert agent.pending_feedback is None


def test_warmup_uses_same_client_without_changing_conversation_or_executing(capsys):
    client = Mock()
    client.responses.create.return_value.output_text = "raise RuntimeError('never execute')"
    agent = Agent(client, "test-model", service_tier="priority")

    with patch("pyshell.agent.time.perf_counter", side_effect=[10.0, 11.0]):
        agent.warm_up()

    # Warm-up uses a short, non-stored request on the configured client.
    client.responses.create.assert_called_once_with(
        model="test-model",
        service_tier="priority",
        reasoning={"effort": "low"},
        input="Reply with only 0.",
        max_output_tokens=16,
        store=False,
    )

    # Warm-up must not execute the returned code or alter conversation state.
    assert agent.previous_response_id is None
    assert agent.pending_feedback is None
    assert agent.shell.result()["exception"] is None
    assert "1.000s" in capsys.readouterr().err


def test_warmup_failure_is_reported_without_raising(capsys):
    client = Mock()
    client.responses.create.side_effect = ConnectionError("disconnected")

    Agent(client, "test-model").warm_up()

    assert "Warm-up failed: disconnected" in capsys.readouterr().err


def test_model_imports_persist_between_instructions(stream_for):
    client = Mock()
    client.responses.stream.side_effect = [
        stream_for(["import math\ndone()"]),
        stream_for(["result = math.sqrt(16)\ndone()"]),
    ]
    agent = Agent(client, "test-model")

    agent.run("import math")
    agent.run("calculate the square root of 16")

    # The second instruction can use imports from the first.
    assert agent.shell.namespace["result"] == 4
    assert json.loads(agent.pending_feedback)["exception"] is None


def test_requests_priority_processing(stream_for):
    client = Mock()
    client.responses.stream.return_value = stream_for(["done()"])
    agent = Agent(client, "test-model", service_tier="priority")

    agent.run("finish")

    assert client.responses.stream.call_args.kwargs["service_tier"] == "priority"
    assert client.responses.stream.call_args.kwargs["reasoning"] == {"effort": "low"}


def test_continues_past_twenty_steps_until_done(stream_for):
    client = Mock()
    client.responses.stream.side_effect = [
        stream_for(["pass"]) for _ in range(21)
    ] + [stream_for(["done()"])]
    agent = Agent(client, "test-model")

    agent.run("continue until finished")

    assert client.responses.stream.call_count == 22
    assert json.loads(agent.pending_feedback)["done"] is True


def test_stream_failure_is_not_executed_and_preserves_feedback(stream_for):
    client = Mock()
    stream = stream_for(["x = 42"])
    stream.get_final_response.side_effect = ConnectionError("disconnected")
    client.responses.stream.return_value = stream
    agent = Agent(client, "test-model")
    agent.pending_feedback = "previous output"

    with pytest.raises(ConnectionError, match="disconnected"):
        agent.run("set x")

    # An unfinished line must not execute or replace earlier feedback.
    assert "x" not in agent.shell.namespace
    assert agent.pending_feedback.startswith("previous output\n")
    assert "ConnectionError" in agent.pending_feedback


def test_incomplete_response_does_not_advance_conversation(stream_for):
    client = Mock()
    client.responses.stream.return_value = stream_for(["x = 42"], status="incomplete")
    agent = Agent(client, "test-model")

    with pytest.raises(RuntimeError, match="Model response incomplete"):
        agent.run("set x")

    assert "x" not in agent.shell.namespace
    assert agent.previous_response_id is None


def test_error_output_is_sent_back_and_next_line_runs(stream_for, monkeypatch):
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    client = Mock()
    client.responses.stream.return_value = stream_for(["1 / 0\nprint('ok')\ndone()"])
    agent = Agent(client, "test-model")

    agent.run("test errors")

    result = json.loads(agent.pending_feedback)
    assert "ZeroDivisionError" in result["exception"]
    assert result["stdout"] == "ok\n"
