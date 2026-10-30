import pytest
from pydantic_ai import Agent
from web.src.script import stream_response


def test_streams_code_to_callback_without_console_logs(model_for, capsys):
    agent = Agent(model_for(["print(", "'hello')"]))
    chunks = []
    code, history = stream_response(agent, "say hello", [], on_code=chunks.append)
    output = capsys.readouterr()
    assert "".join(chunks) == "print('hello')"
    assert output.out == ""
    assert output.err == ""
    assert code == "print('hello')"
    assert history[-1].parts[0].content == code


def test_stream_failure_propagates_without_timing(model_for, capsys):
    agent = Agent(model_for(["value = 42", ConnectionError("disconnected")]))
    with pytest.raises(ConnectionError, match="disconnected"):
        stream_response(agent, "set value", [])
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""
