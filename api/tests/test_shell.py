from src.shell import PythonShell


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
