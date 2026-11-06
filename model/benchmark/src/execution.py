import json
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
            text=True,
            bufsize=1,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def execute(self, source, action="execute"):
        self.process.stdin.write(json.dumps({"action": action, "source": source}) + "\n")
        self.process.stdin.flush()
        if not self.selector.select(30):
            raise TimeoutError("Isolated execution worker did not respond within 30 seconds")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Isolated worker exited: " + self.process.stderr.read()[-2000:])
        return json.loads(line)

    def close(self):
        self.selector.close()
        subprocess.run(["docker", "rm", "--force", self.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        self.process.communicate(timeout=15)
