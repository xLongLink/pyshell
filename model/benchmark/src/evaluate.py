"""Evaluate a GGUF LoRA adapter through the unchanged benchmark worker."""

import os
import sys
import json
import uuid
import argparse
import subprocess
from pathlib import Path
from model.benchmark.src.inference_server import gguf_command, inference_server

ROOT = Path(__file__).resolve().parents[1]


def benchmark_command(args, log_dir):
    if args.syntax_only:
        command = [
            sys.executable,
            "-u",
            "-m",
            "model.benchmark.main",
            "syntax",
            "--model",
            args.label,
            "--temperature",
            str(args.temperature),
            "--output",
            str(log_dir / "syntax"),
            "--serialization",
            args.serialization,
        ]
        if args.syntax_data:
            command += ["--data", str(args.syntax_data.resolve())]
        return command
    prefix = ["uv", "run", "--locked", "--project", str(ROOT), "python", "-u", "-m"]
    if args.audit_data:
        return [
            *prefix,
            "model.benchmark.main",
            "audit",
            "--data",
            str(args.audit_data.resolve()),
            "--output",
            str(args.audit_output.resolve()),
            "--label",
            args.label,
        ]
    settings = {"temperature": args.temperature, "top_p": 0.8, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    command = [
        *prefix,
        "model.benchmark.main",
        "run",
        "--model",
        args.label,
        "--label",
        args.label,
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
        command += ["--case", case]
    return command


def main(argv=None):
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
    args = parser.parse_args(argv)
    if args.syntax_only and args.case:
        parser.error("--case cannot be combined with --syntax-only")
    if args.audit_data and (args.syntax_only or args.case or not args.audit_output):
        parser.error("--audit-data requires --audit-output and cannot be combined with --case or --syntax-only")
    if not args.adapter.is_file():
        raise FileNotFoundError(args.adapter)
    local = ROOT / "local"
    command = gguf_command(local, local / "Qwen3.5-0.8B-Q4_K_M.gguf", port=8030, context_size=8192, batch_threads=12)
    command += [
        "--alias",
        args.label,
        "--lora",
        str(args.adapter.resolve()),
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
    with inference_server(command, log_dir / "inference.log", port=8030, timeout=120):
        result = subprocess.run(
            benchmark_command(args, log_dir), cwd=ROOT.parents[1], env={**os.environ, "BENCHMARK_LOCAL_KEY": "local-only"}, check=False
        )
    return result.returncode
