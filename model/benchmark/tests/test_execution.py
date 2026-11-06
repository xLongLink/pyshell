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
