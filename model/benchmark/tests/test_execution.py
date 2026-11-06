import sys
import subprocess
import pytest
from unittest.mock import patch
from model.benchmark.src import execution


def test_worker_launches_directly_with_the_same_container_isolation():
    with patch.object(execution.subprocess, "Popen") as launch, patch.object(execution.selectors, "DefaultSelector"):
        execution.Worker()

    command = launch.call_args.args[0]
    for flag, value in (
        ("--network", "none"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--memory", "1g"),
        ("--cpus", "2"),
        ("--pids-limit", "64"),
        ("--tmpfs", "/tmp:size=64m"),
        ("--entrypoint", "/opt/venv/bin/python"),
    ):
        assert command[command.index(flag) + 1] == value
    assert "--read-only" in command
    assert "--rm" in command
    volumes = [command[index + 1] for index, value in enumerate(command) if value == "--volume"]
    assert volumes == [f"{execution.ROOT / 'src/worker.py'}:/model/benchmark/src/worker.py:ro"]
    assert command[command.index("--entrypoint") + 2] == "pyshell-pyshell:latest"
    assert command[command.index("--name") + 1].startswith("pyshell-benchmark-")
    assert command[-2] == "-c"
    with patch("model.benchmark.src.worker.main") as main:
        exec(command[-1])
    main.assert_called_once_with()


@pytest.fixture
def local_worker():
    workers = []
    launch = subprocess.Popen

    def start(source):
        def local_launch(command, **kwargs):
            return launch([sys.executable, "-u", "-c", source], **kwargs)

        with patch.object(execution.subprocess, "Popen", side_effect=local_launch):
            worker = execution.Worker()
        workers.append(worker)
        return worker

    yield start

    with patch.object(execution.subprocess, "run"):
        for worker in workers:
            worker.close()
            assert worker.process.poll() is not None


def test_partial_response_obeys_the_original_deadline(local_worker):
    worker = local_worker("import sys; sys.stdin.readline(); sys.stdout.write('{'); sys.stdout.flush(); sys.stdin.read()")
    with patch.object(execution.time, "monotonic", side_effect=[0, 0, 31]):
        with pytest.raises(TimeoutError, match="within 30 seconds"):
            worker.execute("pass")
    assert worker.stdout == b"{"


def test_stderr_is_drained_and_retained_as_a_bounded_tail(local_worker):
    worker = local_worker(
        "import sys\n"
        "for line in sys.stdin:\n"
        "    sys.stderr.write('x' * 262144 + 'diagnostic tail'); sys.stderr.flush()\n"
        "    sys.stdout.write('{\"ok\": true}\\n'); sys.stdout.flush()\n"
    )
    for _ in range(2):
        assert worker.execute("pass") == {"ok": True}
        assert worker.stderr == b"x" * (2000 - len(b"diagnostic tail")) + b"diagnostic tail"


def test_stdout_eof_does_not_wait_for_stderr_to_close(local_worker):
    worker = local_worker("import os, sys; sys.stdin.readline(); os.close(1); sys.stdin.read()")
    with pytest.raises(RuntimeError, match="Isolated worker exited"):
        worker.execute("pass")
    assert worker.process.poll() is None


def test_fragmented_unicode_response_is_decoded_after_the_full_line(local_worker):
    worker = local_worker(
        "import os, sys\n"
        "for line in sys.stdin:\n"
        "    for byte in '{\"text\": \"café\"}\\n'.encode():\n"
        "        os.write(1, bytes([byte]))\n"
    )
    for _ in range(2):
        assert worker.execute("pass") == {"text": "café"}


@pytest.mark.parametrize("timeouts", [0, 1, 2])
@pytest.mark.parametrize("docker_timeout", [False, True])
def test_cleanup_reaps_the_client_even_when_docker_removal_times_out(timeouts, docker_timeout):
    error = subprocess.TimeoutExpired("docker", 15)
    with (
        patch.object(execution.subprocess, "Popen") as launch,
        patch.object(execution.selectors, "DefaultSelector"),
        patch.object(execution.subprocess, "run", side_effect=error if docker_timeout else None) as remove,
    ):
        worker = execution.Worker()
        process = launch.return_value
        process.wait.side_effect = [subprocess.TimeoutExpired("client", 15)] * timeouts + [0]
        if docker_timeout:
            with pytest.raises(subprocess.TimeoutExpired) as raised:
                worker.close()
            assert raised.value is error
        else:
            worker.close()

    assert remove.call_args.kwargs["timeout"] == 15
    assert process.terminate.call_count == (timeouts >= 1)
    assert process.kill.call_count == (timeouts == 2)
    assert [call.kwargs["timeout"] for call in process.wait.call_args_list] == [15, 5, 5][: timeouts + 1]
    for stream in (process.stdin, process.stdout, process.stderr):
        stream.close.assert_called_once_with()
