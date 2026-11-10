import sys
import pytest
import subprocess
from unittest.mock import patch
from model.benchmark.src import execution


def test_worker_launches_directly_with_the_same_container_isolation():
    with (
        patch.object(execution.subprocess, "Popen") as launch,
        patch.object(execution.selectors, "DefaultSelector"),
        patch.object(execution.os, "set_blocking"),
    ):
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


@pytest.mark.parametrize(
    ("phase", "cleanup_failure"),
    [("stdin", None), ("create", None), ("stdout", None), ("stderr", None), ("stderr", "selector"), ("stderr", "docker")],
)
def test_failed_initialization_cleans_up_and_preserves_the_error(phase, cleanup_failure):
    error = RuntimeError("selector initialization failed")
    with (
        patch.object(execution.subprocess, "Popen") as launch,
        patch.object(execution.selectors, "DefaultSelector") as selector,
        patch.object(execution.subprocess, "run") as remove,
        patch.object(execution.os, "set_blocking") as set_blocking,
    ):
        if phase == "stdin":
            set_blocking.side_effect = error
        elif phase == "create":
            selector.side_effect = error
        elif phase == "stdout":
            selector.return_value.register.side_effect = error
        else:
            selector.return_value.register.side_effect = [None, error]
        if cleanup_failure == "selector":
            selector.return_value.close.side_effect = OSError("selector close failed")
        elif cleanup_failure == "docker":
            remove.side_effect = subprocess.TimeoutExpired("docker", 15)
        with pytest.raises(RuntimeError) as raised:
            execution.Worker()

    assert raised.value is error
    process = launch.return_value
    for stream in (process.stdin, process.stdout, process.stderr):
        stream.close.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=15)
    remove.assert_called_once()
    assert remove.call_args.args[0][:3] == ["docker", "rm", "--force"]
    if phase in {"stdin", "create"}:
        selector.return_value.close.assert_not_called()
    else:
        selector.return_value.close.assert_called_once_with()


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
    with patch.object(execution.time, "monotonic", side_effect=lambda: 31 if worker.stdout else 0):
        with pytest.raises(TimeoutError, match="within 30 seconds"):
            worker.execute("pass")
    assert worker.stdout == b"{"


def test_request_delivery_obeys_the_deadline_when_the_child_does_not_read(local_worker):
    worker = local_worker("import time; time.sleep(2)")
    write_pipe = execution.os.write
    sent = []

    def send(fd, data):
        size = write_pipe(fd, data)
        sent.append(size)
        return size

    try:
        with patch.object(execution.time, "monotonic", side_effect=[0, 0, 31]), patch.object(execution.os, "write", side_effect=send):
            with pytest.raises(TimeoutError, match="within 30 seconds"):
                worker.execute("x" * (2 * 1024 * 1024))
        assert 0 < sum(sent) < 2 * 1024 * 1024
        with pytest.raises(KeyError):
            worker.selector.get_key(worker.process.stdin)
    finally:
        worker.process.terminate()


def test_large_requests_are_delivered_while_stderr_is_drained(local_worker):
    worker = local_worker(
        "import json, signal, sys\n"
        "signal.alarm(10)\n"
        "sys.stderr.write('x' * 262144 + 'tail'); sys.stderr.flush()\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    print(json.dumps(request), flush=True)\n"
    )
    source = "café 雪\n" * 32768
    for _ in range(2):
        assert worker.execute(source) == {"action": "execute", "source": source}
        assert worker.stderr.endswith(b"tail")


def test_request_delivery_retries_after_a_nonblocking_write_would_block(local_worker):
    worker = local_worker("import sys\nfor line in sys.stdin:\n    print(line, end='', flush=True)\n")
    write_pipe = execution.os.write
    blocked = False

    def send(fd, data):
        nonlocal blocked
        if not blocked:
            blocked = True
            raise BlockingIOError
        return write_pipe(fd, data)

    with patch.object(execution.os, "write", side_effect=send):
        assert worker.execute("pass") == {"action": "execute", "source": "pass"}
    assert blocked


def test_response_at_the_size_limit_is_accepted(local_worker):
    length = execution.MAX_RESPONSE_BYTES - len(b'{"text": ""}\n')
    worker = local_worker(
        f"import json, sys\ntext = 'x' * {length}\nfor line in sys.stdin:\n    print(json.dumps({{'text': text}}), flush=True)\n"
    )
    result = worker.execute("pass")
    assert len(result["text"]) == length
    assert result["text"].count("x") == length


@pytest.mark.parametrize("terminated", [False, True])
def test_oversized_responses_are_rejected_with_bounded_buffering(local_worker, terminated):
    suffix = b"\n" if terminated else b""
    worker = local_worker(
        "import sys\n"
        "for line in sys.stdin:\n"
        f"    sys.stdout.buffer.write(b'x' * {execution.MAX_RESPONSE_BYTES + 1} + {suffix!r})\n"
        "    sys.stdout.buffer.flush()\n"
    )
    with pytest.raises(RuntimeError, match="response exceeded 16 MiB"):
        worker.execute("pass")
    assert len(worker.stdout) <= execution.MAX_RESPONSE_BYTES


@pytest.mark.parametrize("elapsed", [0, 31])
def test_timeout_does_not_poll_after_the_deadline(local_worker, elapsed):
    worker = local_worker("import sys; sys.stdin.read()")
    with (
        patch.object(execution.time, "monotonic", side_effect=[0, elapsed]),
        patch.object(worker.selector, "select", return_value=[]) as select,
    ):
        with pytest.raises(TimeoutError, match="within 30 seconds"):
            worker.execute("pass")
    if elapsed:
        select.assert_not_called()
    else:
        select.assert_called_once_with(30)


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
        'import os, sys\nfor line in sys.stdin:\n    for byte in \'{"text": "café"}\\n\'.encode():\n        os.write(1, bytes([byte]))\n'
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
        patch.object(execution.os, "set_blocking"),
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


@pytest.mark.parametrize(
    ("failed_streams", "timeouts"), [(("stdin",), 0), (("stdout",), 0), (("stderr",), 0), (("stdin", "stdout", "stderr"), 2)]
)
def test_pipe_close_failures_do_not_skip_other_cleanup(failed_streams, timeouts):
    with (
        patch.object(execution.subprocess, "Popen") as launch,
        patch.object(execution.selectors, "DefaultSelector") as selector,
        patch.object(execution.subprocess, "run") as remove,
        patch.object(execution.os, "set_blocking"),
    ):
        worker = execution.Worker()
        process = launch.return_value
        errors = {name: OSError(f"{name} close failed") for name in failed_streams}
        for name, error in errors.items():
            getattr(process, name).close.side_effect = error
        process.wait.side_effect = [subprocess.TimeoutExpired("client", 15)] * timeouts + [0]
        with pytest.raises(OSError, match="close failed") as raised:
            worker.close()

    assert raised.value is errors[failed_streams[-1]]
    selector.return_value.close.assert_called_once_with()
    remove.assert_called_once()
    for stream in (process.stdin, process.stdout, process.stderr):
        stream.close.assert_called_once_with()
    assert process.wait.call_count == timeouts + 1
    assert process.terminate.call_count == (timeouts >= 1)
    assert process.kill.call_count == (timeouts == 2)
