import pytest
from api.shell import PythonShell


def test_execution_logs_source_and_streams_output_before_completion(capsys):
    shell = PythonShell()
    observed = []
    shell.namespace["observe"] = lambda: observed.append(capsys.readouterr())
    source = "import sys\nprint('running')\nprint('warning', file=sys.stderr)\nobserve()\nprint('finished')"

    result = shell.execute(source)

    assert observed[0].out == source + "\nrunning\n"
    assert observed[0].err == "warning\n"
    assert capsys.readouterr().out == "finished\n"

    # Streaming must not consume output needed by the HTTP response.
    assert result["stdout"] == "running\nfinished\n"
    assert result["stderr"] == "warning\n"
    assert result["exception"] is None


def test_execution_logs_errors_and_keeps_them_in_response(capsys):
    shell = PythonShell()
    source = "print('before failure')\nraise ValueError('failed')"

    result = shell.execute(source)

    output = capsys.readouterr()
    assert output.out == source + "\nbefore failure\n"
    assert output.err == result["exception"]
    assert "ValueError: failed" in output.err

    # Tracebacks remain separate from captured Python stderr.
    assert result["stdout"] == "before failure\n"
    assert result["stderr"] == ""
    assert result["done"] is False


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_execution_caps_each_output_stream_in_bytes_and_keeps_captured_output(stream, capsys):
    shell = PythonShell()
    source = f"import sys\nsys.{stream}.write('é' * 524288)\nsys.{stream}.write('x')\ndone()"

    result = shell.execute(source)

    assert result[stream] == "é" * 524288
    assert result["stderr" if stream == "stdout" else "stdout"] == ""
    assert "Python output exceeded 1 MiB per stream" in result["exception"]
    assert result["done"] is False
    output = capsys.readouterr()
    assert output.out == source + "\n" + (result["stdout"] if stream == "stdout" else "")
    assert output.err == result["stderr"] + result["exception"]
    assert shell.execute("pass")[stream] == ""
