import pytest
from pydantic_ai import Agent
from web.src.script import stream_response


def test_prints_code_without_timing(model_for, capsys):
    agent = Agent(model_for(["print(", "'hello')"]))
    code, history = stream_response(agent, "say hello", [])
    output = capsys.readouterr()
    assert "print('hello')" in output.out
    assert output.err == ""
    assert code == "print('hello')"
    assert history[-1].parts[0].content == code


def test_stream_failure_propagates_without_timing(model_for, capsys):
    agent = Agent(model_for(["value = 42", ConnectionError("disconnected")]))
    with pytest.raises(ConnectionError, match="disconnected"):
        stream_response(agent, "set value", [])
    assert capsys.readouterr().err == ""
