"""Evaluate a GGUF LoRA adapter through the unchanged benchmark worker."""

import os
import sys
import json
import uuid
import argparse
import threading
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--label", default="Qwen3.5-0.8B-Q4_K_M-local-CPU-lora500")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--syntax-only", action="store_true")
    parser.add_argument("--serialization", choices=("chat", "raw"), default="raw")
    parser.add_argument("--syntax-data", type=Path)
    parser.add_argument("--audit-data", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if args.syntax_only and args.case:
        parser.error("--case cannot be combined with --syntax-only")
    if args.audit_data and (args.syntax_only or args.case or not args.audit_output):
        parser.error("--audit-data requires --audit-output and cannot be combined with --case or --syntax-only")
    if not args.adapter.is_file():
        raise FileNotFoundError(args.adapter)
    local = ROOT / "local"
    label = args.label
    command = [
        str(local / "llama-b12038/llama-b12038/llama-server"),
        "--model",
        str(local / "Qwen3.5-0.8B-Q4_K_M.gguf"),
        "--alias",
        label,
        "--lora",
        str(args.adapter.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        "8030",
        "--ctx-size",
        "8192",
        "--parallel",
        "1",
        "--threads",
        "8",
        "--threads-batch",
        "12",
        "--n-gpu-layers",
        "0",
        "--jinja",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--api-key",
        "local-only",
    ]
    log_dir = ROOT / "logs" / f"evaluation-{uuid.uuid4().hex[:8]}"
    log_dir.mkdir(parents=True)
    (log_dir / "inference.json").write_text(json.dumps({"command": command}, indent=2) + "\n")
    ready = threading.Event()
    with (log_dir / "inference.log").open("w") as log:
        server = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        def capture():
            for line in server.stdout:
                log.write(line)
                log.flush()
                if "listening on http://127.0.0.1:8030" in line:
                    ready.set()

        thread = threading.Thread(target=capture, daemon=True)
        thread.start()
        try:
            if not ready.wait(120):
                raise RuntimeError(f"Adapter server failed to start; see {log_dir / 'inference.log'}")
            settings = {"temperature": args.temperature, "top_p": 0.8, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
            benchmark = [
                "uv",
                "run",
                "--locked",
                "--project",
                str(ROOT),
                "python",
                "-u",
                "-m",
                "dataset.benchmark.main",
                "run",
                "--model",
                label,
                "--label",
                label,
                "--api",
                args.serialization,
                "--base-url",
                "http://127.0.0.1:8030/v1",
                "--api-key-env",
                "BENCHMARK_LOCAL_KEY",
                "--settings",
                json.dumps(settings),
                "--repeats",
                "1",
                "--no-build",
            ]
            for case in args.case or []:
                benchmark.extend(["--case", case])
            if args.syntax_only:
                benchmark = [
                    sys.executable,
                    "-u",
                    "-m",
                    "dataset.benchmark.syntax",
                    "--model",
                    label,
                    "--temperature",
                    str(args.temperature),
                    "--output",
                    str(log_dir / "syntax"),
                    "--serialization",
                    args.serialization,
                ]
                if args.syntax_data:
                    benchmark.extend(["--data", str(args.syntax_data.resolve())])
            if args.audit_data:
                benchmark = [
                    "uv",
                    "run",
                    "--locked",
                    "--project",
                    str(ROOT),
                    "python",
                    "-u",
                    "-m",
                    "dataset.benchmark.audit",
                    "--data",
                    str(args.audit_data.resolve()),
                    "--output",
                    str(args.audit_output.resolve()),
                    "--label",
                    label,
                ]
            result = subprocess.run(benchmark, cwd=ROOT.parents[1], env={**os.environ, "BENCHMARK_LOCAL_KEY": "local-only"}, check=False)
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            thread.join(timeout=5)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
