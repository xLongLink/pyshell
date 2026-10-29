from unittest.mock import Mock, patch

import pytest

from script import stream_response


def test_prints_code_and_timing(stream_for, capsys):
    client = Mock()
    client.responses.stream.return_value = stream_for(["", "print('hello')"])
    with patch("script.time.perf_counter", side_effect=[10.0, 10.25, 11.5]):
        response = stream_response(client, {})
    output = capsys.readouterr()
    assert "print('hello')" in output.out
    assert "Time to first token: 0.250s | Total duration: 1.500s" in output.err
    assert response.output_text == "print('hello')"


def test_incomplete_stream_is_rejected(stream_for):
    client = Mock()
    client.responses.stream.return_value = stream_for(["value = 42"], status="incomplete")
    with pytest.raises(RuntimeError, match="code was not executed"):
        stream_response(client, {})


def test_timing_when_stream_fails_before_first_token(capsys):
    client = Mock()
    client.responses.stream.side_effect = ConnectionError("disconnected")
    with patch("script.time.perf_counter", side_effect=[10.0, 12.0]):
        with pytest.raises(ConnectionError, match="disconnected"):
            stream_response(client, {})
    assert "Time to first token: n/a | Total duration: 2.000s" in capsys.readouterr().err
