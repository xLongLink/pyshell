"""Run a bounded CPU pilot, selecting on development tasks before the frozen test."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dataset.benchmark.src.corpus import sha256, verify_training_data
from dataset.benchmark.src.proof_policy import development_ready
from model.train.src.export import export_command

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def run(args):
    if args.baseline_before_training and (
        not args.initialize_adapter or args.backend != "hf"
    ):
        raise ValueError(
            "The baseline comparison requires an initial adapter and the HF backend"
        )
    manifest = json.loads((args.data / "manifest.json").read_text())
    for split, metadata in manifest["splits"].items():
        actual = sha256(args.data / f"{split}-tasks.jsonl")
        if actual != metadata["sha256"]:
            raise ValueError(f"Frozen {split} tasks changed")
    training_data = args.training_data or args.data
    training_manifest = json.loads((training_data / "manifest.json").read_text())
    if (
        args.training_data
        and training_manifest["evaluation_test_sha256"]
        != manifest["splits"]["test"]["sha256"]
    ):
        raise ValueError("Curriculum does not identify the frozen final test")
    verify_training_data(training_data, training_manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "training",
        "base": "Qwen/Qwen3.5-0.8B",
        "fresh_adapter": args.initialize_adapter is None,
        "initialized_from": str(args.initialize_adapter)
        if args.initialize_adapter
        else None,
        "training_manifest": training_manifest,
        "evaluation_backend": args.backend,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "threads": args.threads,
        "dataset_manifest": manifest,
        "selection": "dev only: >=90% overall and >=60% per family before final test",
    }
    state = args.output / "experiment.json"
    state.write_text(json.dumps(metadata, indent=2) + "\n")

    def command(phase, arguments):
        print(f"Starting {phase}: {' '.join(arguments)}", flush=True)
        metadata["status"] = phase
        state.write_text(json.dumps(metadata, indent=2) + "\n")
        with (args.output / f"{phase}.log").open("w") as log:
            subprocess.run(
                arguments, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True
            )

    if args.baseline_before_training:
        baseline = args.output / "baseline-dev"
        command(
            "baseline_development",
            [
                sys.executable,
                "-u",
                "-m",
                "dataset.benchmark.main",
                "cpu",
                "--data",
                str(args.data.resolve()),
                "--adapter",
                str(args.initialize_adapter.resolve()),
                "--backend",
                "hf",
                "--threads",
                str(args.threads),
                "--label",
                args.output.name + "-baseline",
                "--output",
                str(baseline.resolve()),
            ],
        )
        metadata["baseline_development"] = json.loads(
            (baseline / "summary.json").read_text()
        )

    adapter = args.output / "adapter"
    command(
        "training",
        [
            sys.executable,
            "-u",
            "-m",
            "model.train.main",
            "train",
            "--data",
            str(training_data.resolve()),
            "--output",
            str(adapter.resolve()),
            "--steps",
            str(args.steps),
            "--threads",
            str(args.threads),
            "--learning-rate",
            str(args.learning_rate),
            "--serialization",
            "raw",
            "--eval-samples",
            "12",
            *(
                ["--initialize-adapter", str(args.initialize_adapter.resolve())]
                if args.initialize_adapter
                else []
            ),
        ],
    )
    gguf = args.output / "adapter.gguf"
    if args.backend == "gguf":
        command("export", export_command(adapter.resolve(), gguf.resolve()))
    dev = args.output / "dev"
    evaluation = [
        sys.executable,
        "-u",
        "-m",
        "dataset.benchmark.main",
        "cpu",
        "--data",
        str(args.data.resolve()),
        "--adapter",
        str(adapter.resolve()) if args.backend == "hf" else str(gguf.resolve()),
        "--backend",
        args.backend,
        "--threads",
        str(args.threads),
        "--label",
        args.output.name,
    ]
    command("development", [*evaluation, "--output", str(dev.resolve())])
    summary = json.loads((dev / "summary.json").read_text())
    ready = development_ready(summary)
    metadata.update(
        status="development_complete", development=summary, ready_for_final=ready
    )
    if ready and args.final_if_ready:
        command(
            "final_test",
            [
                *evaluation,
                "--split",
                "test",
                "--final",
                "--output",
                str((args.output / "test").resolve()),
            ],
        )
        final = json.loads((args.output / "test/summary.json").read_text())
        metadata.update(
            status="complete", final=final, objective_met=final["acceptance_met"]
        )
    else:
        metadata["status"] = "needs_development" if not ready else "ready_for_final"
    state.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=REPO / "dataset/benchmark/artifacts/cpu-proof-v1"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--initialize-adapter", type=Path)
    parser.add_argument("--backend", choices=("gguf", "hf"), default="gguf")
    parser.add_argument("--final-if-ready", action="store_true")
    parser.add_argument("--baseline-before-training", action="store_true")
    run(parser.parse_args(argv))
