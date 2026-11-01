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


def test_interactive_lines_distinguish_silent_completion_from_incomplete_input():
    shell = PythonShell()

    result = shell.execute("values = []", interactive=True)
    assert result == {"stdout": "", "stderr": "", "exception": None, "done": False, "more_input": False}
    for line in ("for i in range(3):", "    values.append(i)"):
        result = shell.execute(line, interactive=True)
        assert result["more_input"] is True
        assert result["exception"] is None
        assert shell.namespace["values"] == []
    assert shell.execute("", interactive=True)["more_input"] is False
    assert shell.execute("values", interactive=True)["stdout"] == "[0, 1, 2]\n"
    assert shell.execute("done()", interactive=True)["done"] is True


def test_interactive_multiline_expression_preserves_indentation():
    shell = PythonShell()

    for line in ("value = (", "    1 +", "    2"):
        assert shell.execute(line, interactive=True)["more_input"] is True
    result = shell.execute(")", interactive=True)
    assert result["exception"] is None
    assert result["more_input"] is False
    assert shell.execute("value", interactive=True)["stdout"] == "3\n"


@pytest.mark.parametrize(("lines", "error"), [(["if True:", "bad indentation"], "IndentationError"), (["1 / 0"], "ZeroDivisionError")])
def test_interactive_errors_clear_pending_input(lines, error):
    shell = PythonShell()

    for line in lines:
        result = shell.execute(line, interactive=True)
    assert error in result["exception"]
    assert result["more_input"] is False
    assert shell.execute("40 + 2", interactive=True)["stdout"] == "42\n"


def test_interactive_runtime_failure_keeps_successful_changes():
    shell = PythonShell()
    for line in ("values = []", "for i in range(3):", "    values.append(i)", "    1 / 0"):
        shell.execute(line, interactive=True)

    result = shell.execute("", interactive=True)

    assert "ZeroDivisionError" in result["exception"]
    assert shell.execute("values", interactive=True)["stdout"] == "[0]\n"
