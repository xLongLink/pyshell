"""Measure fresh-workspace execution through a network-isolated Python container."""

import json
import time
import uuid
import hashlib
import argparse
import selectors
import threading
import subprocess
import urllib.request
from pathlib import Path
from collections import Counter
from model.train.cpu_resources import exclusive_cpu
from dataset.benchmark.src.serialization import CODE, TASK, CONSOLE, END_CODE, serialize

ROOT = Path(__file__).resolve().parent


class Worker:
    def __init__(self):
        self.name = "pyshell-cpu-proof-" + uuid.uuid4().hex[:10]
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
                f"{ROOT / 'cpu_worker.py'}:/cpu_worker.py:ro",
                "--entrypoint",
                "/opt/venv/bin/python",
                "pyshell-benchmark-pyshell:latest",
                "/cpu_worker.py",
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


def completion(prompt):
    payload = {
        "prompt": prompt,
        "temperature": 0,
        "seed": 713,
        "n_predict": 384,
        "stop": [END_CODE, TASK, CONSOLE, CODE],
        "stream": False,
        "cache_prompt": True,
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8032/completion", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.load(response)
    return result["content"], result.get("stopping_word") == END_CODE


def trial(worker, record, generate=None):
    generate = generate or completion
    setup = worker.execute(record["setup"], "reset")
    if setup["exception"]:
        raise RuntimeError(f"Invalid fixture {record['id']}: {setup['exception']}")
    turns = [{"role": "user", "content": record["instruction"]}]
    stages = [{"instruction": record["instruction"], "check": record["check"]}]
    if record["followup"]:
        stages.append(record["followup"])
    row = {"id": record["id"], "family": record["family"], "steps": [], "status": "call_limit"}
    for stage_index, stage in enumerate(stages):
        if stage_index:
            worker.execute("saved_result = result")
            turns.append({"role": "user", "content": stage["instruction"]})
        for _ in range(4):
            started = time.monotonic()
            source, ended = generate(serialize(turns))
            step = {"source": source, "stage": stage_index, "end_marker_seen": ended, "seconds": time.monotonic() - started}
            row["steps"].append(step)
            if not ended or not source.strip():
                row["status"] = "boundary_error"
                return row
            result = worker.execute(source)
            step["feedback"] = result
            turns.extend([{"role": "assistant", "content": source}, {"role": "user", "content": json.dumps(result)}])
            if result["done"] is not None and not result["exception"]:
                grade = worker.execute(stage["check"])
                if grade["exception"]:
                    row.update(status="incorrect", error=grade["exception"])
                    return row
                break
        else:
            return row
    row["status"] = "passed"
    return row


@exclusive_cpu()
def evaluate(args):
    manifest = json.loads((args.data / "manifest.json").read_text())
    path = args.data / f"{args.split}-tasks.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest["splits"][args.split]["sha256"]:
        raise ValueError("Frozen evaluation tasks have changed")
    if args.split == "test" and not args.final:
        raise ValueError("The untouched test requires --final after checkpoint selection")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if args.limit:
        if args.split == "test":
            raise ValueError("The final test cannot be reduced or cherry-picked")
        records = records[: args.limit]
    adapter_file = None
    if args.adapter:
        adapter_file = args.adapter / "adapter_model.safetensors" if args.adapter.is_dir() else args.adapter
    adapter_digest = hashlib.sha256(adapter_file.read_bytes()).hexdigest() if adapter_file else None
    args.output.mkdir(parents=True, exist_ok=False)
    command = [
        str(ROOT / "local/llama-b12038/llama-b12038/llama-server"),
        "--model",
        str(args.gguf_base.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        "8032",
        "--ctx-size",
        "4096",
        "--parallel",
        "1",
        "--threads",
        "8",
        "--threads-batch",
        "8",
        "--n-gpu-layers",
        "0",
    ]
    if args.adapter:
        command += ["--lora", str(args.adapter.resolve())]
    ready = threading.Event()
    rows = []
    with (args.output / "server.log").open("w") as log:
        server = None
        thread = None
        generate = None

        def capture():
            for line in server.stdout:
                log.write(line)
                log.flush()
                if "listening on http://127.0.0.1:8032" in line:
                    ready.set()

        if args.backend == "gguf":
            server = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            thread = threading.Thread(target=capture, daemon=True)
            thread.start()
        else:
            from model.train.cpu_inference import HFCompletion

            generate = HFCompletion(args.adapter, args.threads)
            ready.set()
        worker = None
        try:
            if not ready.wait(90):
                raise RuntimeError("Inference server did not become ready; inspect server.log")
            worker = Worker()
            for index, record in enumerate(records, 1):
                row = trial(worker, record, generate)
                rows.append(row)
                with (args.output / "responses.jsonl").open("a") as file:
                    file.write(json.dumps(row) + "\n")
                groups = {}
                for family in sorted({item["family"] for item in rows}):
                    group = [item for item in rows if item["family"] == family]
                    groups[family] = {"tasks": len(group), "passed": sum(item["status"] == "passed" for item in group)}
                passed = sum(item["status"] == "passed" for item in rows)
                summary = {
                    "label": args.label,
                    "split": args.split,
                    "complete": len(rows) == len(records),
                    "tasks": len(rows),
                    "passed": passed,
                    "success_rate": passed / len(rows),
                    "by_family": groups,
                    "statuses": dict(Counter(item["status"] for item in rows)),
                    "task_file_sha256": digest,
                    "adapter": str(args.adapter) if args.adapter else None,
                    "adapter_sha256": adapter_digest,
                    "backend": args.backend,
                    "gguf_base": str(args.gguf_base) if args.backend == "gguf" else None,
                    "scope": manifest["scope"],
                    "isolation": "network-disabled container; fresh PythonShell per task; no reference prior context",
                    "acceptance_met": args.split == "test" and len(rows) == 120 and passed >= 97,
                }
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(f"{args.label} {args.split} {index}/{len(records)}: {passed} passed; {record['family']}={row['status']}", flush=True)
        finally:
            if worker:
                worker.close()
            if server:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                thread.join(timeout=5)
    if adapter_file and hashlib.sha256(adapter_file.read_bytes()).hexdigest() != adapter_digest:
        raise RuntimeError("Adapter changed during evaluation")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/cpu-proof-v1")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backend", choices=("gguf", "hf"), default="gguf")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gguf-base", type=Path, default=ROOT / "local/Qwen3.5-0.8B-Q4_K_M.gguf")
    evaluate(parser.parse_args())
