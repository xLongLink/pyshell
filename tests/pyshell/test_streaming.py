import pytest
from types import SimpleNamespace
from pyshell.shell import PythonShell
from unittest.mock import Mock, patch
from pyshell.streaming import stream_response


def test_prints_first_token_time_and_duration(stream_for, capsys):
    client = Mock()
    client.responses.stream.return_value = stream_for(["", "done()"])

    with patch("pyshell.streaming.time.perf_counter", side_effect=[10.0, 10.25, 11.5]):
        stream_response(client, PythonShell(), {})

    assert capsys.readouterr().err == (
        "[Time to first token: 0.250s | Total duration: 1.500s]\n"
    )


def test_timing_when_stream_fails_before_first_token(capsys):
    client = Mock()
    client.responses.stream.side_effect = ConnectionError("disconnected")

    with patch("pyshell.streaming.time.perf_counter", side_effect=[10.0, 12.0]):
        with pytest.raises(ConnectionError, match="disconnected"):
            stream_response(client, PythonShell(), {})

    # Even a failure before any tokens arrive reports elapsed time.
    assert capsys.readouterr().err == (
        "[Time to first token: n/a | Total duration: 2.000s]\n"
    )


def test_executes_lines_before_response_finishes(stream_for, capsys):
    client = Mock()
    stream = stream_for(['print(', '"hello")\n', 'done()'])
    client.responses.stream.return_value = stream
    shell = PythonShell()

    def final_response():
        assert capsys.readouterr().out == '\n>>> print("hello")\nhello\n>>> done()'
        assert shell.finished is False
        return SimpleNamespace(
            id="r1", output_text='print("hello")\ndone()', status="completed",
        )

    stream.get_final_response.side_effect = final_response
    response = stream_response(client, shell, {})

    # The trailing done() executes only after the response completes.
    assert capsys.readouterr().out == "\n"
    assert shell.result()["stdout"] == "hello\n"
    assert shell.finished is True
    assert response.id == "r1"


def test_line_executes_before_next_stream_event(stream_for):
    client = Mock()
    stream = stream_for(["x = 42\n", "done()\n"])
    shell = PythonShell()

    def events():
        yield SimpleNamespace(type="response.output_text.delta", delta="x = 42\n")
        assert shell.namespace["x"] == 42
        yield SimpleNamespace(type="response.output_text.delta", delta="done()\n")

    stream.__iter__.side_effect = events
    client.responses.stream.return_value = stream

    stream_response(client, shell, {})

    assert shell.finished is True


def test_multiline_blocks_and_expression_display(stream_for, capsys):
    client = Mock()
    client.responses.stream.return_value = stream_for([
        "values = []\nfor i in range(3):\n",
        "    values.append(i)\n", "\nvalues\ndone()",
    ])
    shell = PythonShell()

    stream_response(client, shell, {})

    assert shell.namespace["values"] == [0, 1, 2]
    assert shell.result()["stdout"] == "[0, 1, 2]\n"
    assert "...     values.append(i)" in capsys.readouterr().out


def test_disconnect_keeps_executed_lines_but_discards_partial_line(stream_for):
    client = Mock()
    stream = stream_for(["x = 42\ny = 99"])
    stream.get_final_response.side_effect = ConnectionError("disconnected")
    client.responses.stream.return_value = stream
    shell = PythonShell()

    with pytest.raises(ConnectionError, match="disconnected"):
        stream_response(client, shell, {})

    # Keep completed work, but discard the unterminated assignment.
    assert shell.namespace["x"] == 42
    assert "y" not in shell.namespace
    assert shell.buffer == []
