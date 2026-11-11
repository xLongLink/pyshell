import os
import json
import time
import uuid
import logging
import selectors
import contextlib
import subprocess  # noqa: S404
from typing import Any
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class Worker:
    """Execute benchmark code in an isolated Docker container."""

    def __init__(self) -> None:
        """Start the container and register its output streams."""
        self.name = "pyshell-benchmark-" + uuid.uuid4().hex[:10]
        self.process = subprocess.Popen(  # noqa: S603
            [  # noqa: S607
                "docker",
                "run",
                "-i",
                "--name",
                self.name,
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--memory",
                "1g",
                "--cpus",
                "2",
                "--pids-limit",
                "64",
                "--tmpfs",
                "/tmp:size=64m",  # noqa: S108
                "--volume",
                f"{ROOT / 'src/worker.py'}:/model/benchmark/src/worker.py:ro",
                "--env",
                "PYTHONPATH=/",
                "--entrypoint",
                "/opt/venv/bin/python",
                "pyshell-pyshell:latest",
                "-c",
                "from model.benchmark.src.worker import main; main()",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.stdout = bytearray()
        self.stderr = b""
        self.selector = None
        try:
            os.set_blocking(self.process.stdin.fileno(), False)
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.process.stdout, selectors.EVENT_READ)
            self.selector.register(self.process.stderr, selectors.EVENT_READ)
        except BaseException:
            with contextlib.suppress(BaseException):
                self.close()
            raise

    def execute(self, source: str, action: str = "execute") -> dict[str, Any]:
        """Send code to the worker and return its execution feedback."""
        deadline = time.monotonic() + 30
        pending = memoryview((json.dumps({"action": action, "source": source}) + "\n").encode())
        self.selector.register(self.process.stdin, selectors.EVENT_WRITE)
        try:
            while pending or b"\n" not in self.stdout:
                remaining = deadline - time.monotonic()
                events = self.selector.select(remaining) if remaining > 0 else []
                if not events:
                    raise TimeoutError("Isolated execution worker did not respond within 30 seconds")
                for key, _ in sorted(events, key=lambda event: event[0].fileobj is self.process.stdout):
                    if key.fileobj is self.process.stdin:
                        try:
                            written = os.write(key.fileobj.fileno(), pending)
                        except BlockingIOError:
                            continue
                        pending = pending[written:]
                        if not pending:
                            self.selector.unregister(key.fileobj)
                        continue
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        self.selector.unregister(key.fileobj)
                    if key.fileobj is self.process.stderr:
                        self.stderr = (self.stderr + chunk)[-2000:]
                    elif not chunk:
                        raise RuntimeError("Isolated worker exited: " + self.stderr.decode(errors="replace"))
                    else:
                        if len(self.stdout) + len(chunk) > MAX_RESPONSE_BYTES:
                            raise RuntimeError("Isolated worker response exceeded 16 MiB")
                        self.stdout.extend(chunk)
        finally:
            if pending:
                self.selector.unregister(self.process.stdin)
        line, self.stdout = self.stdout.split(b"\n", 1)
        return json.loads(line)

    def close(self) -> None:
        """Remove the container and close the worker streams."""
        try:
            with contextlib.ExitStack() as cleanup:
                cleanup.callback(self.process.stderr.close)
                cleanup.callback(self.process.stdout.close)
                cleanup.callback(self.process.stdin.close)
                try:
                    if self.selector is not None:
                        self.selector.close()
                finally:
                    result = subprocess.run(  # noqa: S603
                        ["docker", "rm", "--force", self.name],  # noqa: S607
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        timeout=15,
                    )
                    if result.returncode:
                        logging.getLogger(__name__).error(
                            "Failed to remove container %s (exit %s): %s", self.name, result.returncode, result.stderr.strip()
                        )
        finally:
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
