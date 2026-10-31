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
def test_persistent_logging_handler_captures_each_execution(stream, capsys):
    shell = PythonShell()
    setup = (
        "import logging, sys\n"
        "logger = logging.Logger('persistent')\n"
        f"logger.addHandler(logging.StreamHandler(sys.{stream}))\n"
        "logger.warning('first')"
    )

    first = shell.execute(setup)
    second = shell.execute("logger.warning('second')")

    assert first[stream] == "first\n"
    assert second[stream] == "second\n"
    assert second["stderr" if stream == "stdout" else "stdout"] == ""
    assert second["exception"] is None
    assert "second\n" in getattr(capsys.readouterr(), "out" if stream == "stdout" else "err")


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_saved_stream_has_a_fresh_budget_and_does_not_capture_between_executions(stream, capsys):
    shell = PythonShell()
    first = shell.execute(f"import sys\nsaved = sys.{stream}\nsaved.write('é' * 524288)")
    assert first[stream] == "é" * 524288
    assert first["exception"] is None
    saved = shell.namespace["saved"]
    saved.write("outside execution\n")
    assert saved.getvalue() == first[stream]
    assert "outside execution\n" in getattr(capsys.readouterr(), "out" if stream == "stdout" else "err")

    source = "saved.write('é' * 524288)\nsaved.write('x')\ndone()"
    second = shell.execute(source)

    assert second[stream] == "é" * 524288
    assert second["stderr" if stream == "stdout" else "stdout"] == ""
    assert "Python output exceeded 1 MiB per stream" in second["exception"]
    assert second["done"] is False
    output = capsys.readouterr()
    assert output.out == source + "\n" + (second["stdout"] if stream == "stdout" else "")
    assert output.err == second["stderr"] + second["exception"]
    assert first[stream] == "é" * 524288
    assert shell.execute("print('recovered', file=saved)")[stream] == "recovered\n"
    assert shell.execute("pass")[stream] == ""
