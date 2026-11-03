"""Train the pilot, export its adapter, then run the held-out worker benchmark."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from model.train.src.checkpoint_evaluation import evaluation_command
from model.train.src.export import CONVERTER, export_command

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--name", default="adapter-500")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--serialization", choices=("chat", "raw"), default="raw")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--initialize-adapter", type=Path)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--evaluate-every", type=int, default=0)
    args = parser.parse_args(argv)
    if args.resume and args.initialize_adapter:
        parser.error("--resume and --initialize-adapter are mutually exclusive")
    data = (
        args.data
        or (
            ROOT.parents[1] / "dataset/data"
            if args.serialization == "raw"
            else ROOT / "data"
        )
    ).resolve()
    logs = ROOT / "logs" / args.name
    logs.mkdir(parents=True, exist_ok=True)
    adapter = ROOT / "artifacts" / args.name
    gguf = ROOT / "artifacts" / f"{args.name}.gguf"
    if not CONVERTER.is_file():
        raise FileNotFoundError(CONVERTER)
    commands = [
        (
            "training",
            [
                sys.executable,
                "-u",
                "-m",
                "model.train.main",
                "train",
                "--output",
                str(adapter),
                "--threads",
                str(args.threads),
                "--steps",
                str(args.steps),
                "--serialization",
                args.serialization,
                "--data",
                str(data),
                "--learning-rate",
                str(args.learning_rate),
                "--evaluate-every",
                str(args.evaluate_every),
                *(
                    ["--initialize-adapter", str(args.initialize_adapter.resolve())]
                    if args.initialize_adapter
                    else []
                ),
                *(["--resume", str(args.resume)] if args.resume else []),
            ],
        ),
        (
            "export",
            export_command(adapter, gguf),
        ),
        (
            "syntax",
            [
                sys.executable,
                "-u",
                "-m",
                "dataset.benchmark.main",
                "evaluate",
                "--adapter",
                str(gguf),
                "--serialization",
                args.serialization,
                "--temperature",
                "0",
                "--syntax-only",
                "--label",
                args.name,
                "--syntax-data",
                str(data / "validation.jsonl"),
            ],
        ),
        (
            "benchmark",
            evaluation_command(
                adapter, args.steps, adapter / "evaluations.jsonl", args.serialization
            ),
        ),
    ]
    for phase, command in commands:
        (logs / "pipeline.json").write_text(
            json.dumps(
                {"phase": phase, "status": "running", "command": command}, indent=2
            )
            + "\n"
        )
        print(f"Starting {phase}", flush=True)
        with (logs / f"{phase}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT.parents[1],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if returncode != 0 and not (
            phase in {"benchmark", "syntax"} and returncode == 1
        ):
            (logs / "pipeline.json").write_text(
                json.dumps(
                    {"phase": phase, "status": "failed", "returncode": returncode},
                    indent=2,
                )
                + "\n"
            )
            return returncode
    (logs / "pipeline.json").write_text(
        json.dumps(
            {"status": "completed", "benchmark_returncode": returncode}, indent=2
        )
        + "\n"
    )
    return returncode
