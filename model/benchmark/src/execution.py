import os
import json
import time
import uuid
import selectors
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Worker:
    def __init__(self):
        self.name = "pyshell-benchmark-" + uuid.uuid4().hex[:10]
        self.process = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
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
                "/tmp:size=64m",
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
        self.stdout = b""
        self.stderr = b""
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.selector.register(self.process.stderr, selectors.EVENT_READ)

    def execute(self, source, action="execute"):
        self.process.stdin.write((json.dumps({"action": action, "source": source}) + "\n").encode())
        self.process.stdin.flush()
        deadline = time.monotonic() + 30
        while b"\n" not in self.stdout:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Isolated execution worker did not respond within 30 seconds")
            events = self.selector.select(remaining)
            if not events:
                raise TimeoutError("Isolated execution worker did not respond within 30 seconds")
            for key, _ in sorted(events, key=lambda event: event[0].fileobj is self.process.stdout):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    self.selector.unregister(key.fileobj)
                if key.fileobj is self.process.stderr:
                    self.stderr = (self.stderr + chunk)[-2000:]
                elif not chunk:
                    raise RuntimeError("Isolated worker exited: " + self.stderr.decode(errors="replace"))
                else:
                    self.stdout += chunk
        line, self.stdout = self.stdout.split(b"\n", 1)
        return json.loads(line)

    def close(self):
        self.selector.close()
        try:
            subprocess.run(
                ["docker", "rm", "--force", self.name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=15,
            )
        finally:
            self.process.stdin.close()
            self.process.stdout.close()
            self.process.stderr.close()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
