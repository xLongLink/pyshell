"""Check subprocess output streaming and bounded interruption cleanup."""

import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from model.train.src import processes


@pytest.mark.parametrize(
    ("backend", "device", "extra"),
    [("gguf", "cpu", None), ("hf", "cpu", "local"), ("hf", "cuda", "cuda")],
)
def test_benchmark_runs_in_its_own_environment(backend, device, extra):
    command = processes.benchmark_command(
        "cpu", "--label", "pilot", backend=backend, device=device
    )
    assert command[:5] == [
        "uv",
        "run",
        "--locked",
        "--project",
        str(Path(__file__).resolve().parents[2] / "benchmark"),
    ]
    assert command[-7:] == [
        "python",
        "-u",
        "-m",
        "model.benchmark.main",
        "cpu",
        "--label",
        "pilot",
    ]
    if extra is None:
        assert "--extra" not in command
    else:
        assert command[command.index("--extra") + 1] == extra


class TestProcess:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.root = tmp_path
        self.log_path = self.root / "process.log"

    def test_real_subprocess_merges_streams_and_flushes_before_observing_each_line(
        self,
    ):
        command = [
            sys.executable,
            "-c",
            "import os, sys; "
            "print(os.getcwd(), flush=True); "
            "print('standard output', flush=True); "
            "print('standard error', file=sys.stderr, flush=True); "
            "sys.stdout.write('unterminated'); sys.stdout.flush(); sys.exit(7)",
        ]
        observed = []
        console = io.StringIO()

        def on_line(line):
            observed.append(line)
            assert self.log_path.read_text() == "".join(observed)
            assert console.getvalue() == "".join(observed)
            assert flush.call_count == len(observed)

        with (
            redirect_stdout(console),
            patch.object(console, "flush", wraps=console.flush) as flush,
        ):
            returncode = processes.run_logged(
                command, self.log_path, cwd=self.root, on_line=on_line
            )

        expected = [
            f"{self.root}\n",
            "standard output\n",
            "standard error\n",
            "unterminated",
        ]
        assert returncode == 7
        assert observed == expected
        assert self.log_path.read_text() == "".join(expected)
        assert console.getvalue() == "".join(expected)

    def test_real_subprocess_succeeds_without_an_observer_and_replaces_old_log(self):
        self.log_path.write_text("old output\n")

        with redirect_stdout(io.StringIO()) as console:
            returncode = processes.run_logged(
                [sys.executable, "-c", "print('fresh output')"],
                self.log_path,
                cwd=self.root,
            )

        assert returncode == 0
        assert self.log_path.read_text() == "fresh output\n"
        assert console.getvalue() == "fresh output\n"

    def test_stream_failures_terminate_and_wait_before_reraising_original_error(
        self, subtests
    ):
        for error in (KeyboardInterrupt(), SystemExit(3), OSError("stream failed")):
            with subtests.test(error=type(error).__name__):

                def lines():
                    yield "partial output\n"
                    raise error

                stdout = MagicMock(name="stdout")
                stdout.__iter__.return_value = lines()
                process = Mock(stdout=stdout)
                with (
                    patch.object(processes.subprocess, "Popen", return_value=process),
                    pytest.raises(type(error)) as raised,
                ):
                    processes.run_logged(["worker"], self.log_path, cwd=self.root)

                assert raised.value is error
                assert process.mock_calls == [call.terminate(), call.wait(timeout=30)]
                stdout.close.assert_called_once_with()
                assert self.log_path.read_text() == "partial output\n"

    def test_wait_interruption_terminates_and_reaps_before_reraising(self):
        error = KeyboardInterrupt()
        process = Mock(stdout=io.StringIO())
        process.wait.side_effect = [error, 0]

        with (
            patch.object(processes.subprocess, "Popen", return_value=process),
            pytest.raises(KeyboardInterrupt) as raised,
        ):
            processes.run_logged(["worker"], self.log_path, cwd=self.root)

        assert raised.value is error
        assert process.stdout.closed
        assert process.mock_calls == [
            call.wait(),
            call.terminate(),
            call.wait(timeout=30),
        ]

    def test_cleanup_timeout_kills_and_reaps_without_replacing_original_error(self):
        error = KeyboardInterrupt()
        stdout = MagicMock(name="stdout")
        stdout.__iter__.side_effect = error
        process = Mock(stdout=stdout)
        process.wait.side_effect = [subprocess.TimeoutExpired(["worker"], 30), -9]

        with (
            patch.object(processes.subprocess, "Popen", return_value=process),
            pytest.raises(KeyboardInterrupt) as raised,
        ):
            processes.run_logged(["worker"], self.log_path, cwd=self.root)

        assert raised.value is error
        stdout.close.assert_called_once_with()
        assert process.mock_calls == [
            call.terminate(),
            call.wait(timeout=30),
            call.kill(),
            call.wait(),
        ]

    def test_observer_failure_preserves_teed_line_and_stops_further_output(self):
        error = RuntimeError("observer failed")
        observer = Mock(side_effect=error)
        process = Mock(stdout=io.StringIO("first line\nmust not be read\n"))

        with (
            redirect_stdout(io.StringIO()) as console,
            patch.object(processes.subprocess, "Popen", return_value=process),
            pytest.raises(RuntimeError) as raised,
        ):
            processes.run_logged(
                ["worker"], self.log_path, cwd=self.root, on_line=observer
            )

        assert raised.value is error
        assert process.stdout.closed
        observer.assert_called_once_with("first line\n")
        assert self.log_path.read_text() == "first line\n"
        assert console.getvalue() == "first line\n"
        assert process.mock_calls == [call.terminate(), call.wait(timeout=30)]

    def test_success_closes_the_pipe_and_preserves_the_wait_return_code(self):
        process = Mock(stdout=io.StringIO("output\n"))
        process.wait.return_value = -15

        with patch.object(processes.subprocess, "Popen", return_value=process) as popen:
            returncode = processes.run_logged(
                ["worker", "argument"], self.log_path, cwd=self.root
            )

        assert returncode == -15
        assert process.stdout.closed
        assert process.mock_calls == [call.wait()]
        popen.assert_called_once_with(
            ["worker", "argument"],
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
