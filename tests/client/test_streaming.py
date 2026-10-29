from unittest.mock import Mock

import pytest

from script import stream_response


def test_prints_code_without_timing(stream_for, capsys):
    client = Mock()
    client.responses.stream.return_value = stream_for(["", "print('hello')"])
    response = stream_response(client, {})
    output = capsys.readouterr()
    assert "print('hello')" in output.out
    assert output.err == ""
    assert response.output_text == "print('hello')"


def test_incomplete_stream_is_rejected(stream_for):
    client = Mock()
    client.responses.stream.return_value = stream_for(["value = 42"], status="incomplete")
    with pytest.raises(RuntimeError, match="code was not executed"):
        stream_response(client, {})


def test_stream_failure_propagates_without_timing(capsys):
    client = Mock()
    client.responses.stream.side_effect = ConnectionError("disconnected")
    with pytest.raises(ConnectionError, match="disconnected"):
        stream_response(client, {})
    assert capsys.readouterr().err == ""
