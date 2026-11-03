"""Run a bounded CPU pilot, selecting on development tasks before the frozen test."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def run(args):
    manifest = json.loads((args.data / "manifest.json").read_text())
    for split, metadata in manifest["splits"].items():
        actual = hashlib.sha256(
            (args.data / f"{split}-tasks.jsonl").read_bytes()
        ).hexdigest()
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
    for name, key in (
        ("train.jsonl", "training_records_sha256"),
        ("validation.jsonl", "validation_records_sha256"),
    ):
        if (
            hashlib.sha256((training_data / name).read_bytes()).hexdigest()
            != training_manifest[key]
        ):
            raise ValueError(f"Training corpus changed: {name}")
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

    adapter = args.output / "adapter"
    command(
        "training",
        [
            sys.executable,
            "-u",
            "-m",
            "model.train.train",
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
    command(
        "export",
        [
            sys.executable,
            str(ROOT / "artifacts/llama.cpp-b12038/convert_lora_to_gguf.py"),
            "--base",
            str(ROOT / "artifacts/base"),
            "--outfile",
            str(gguf.resolve()),
            str(adapter.resolve()),
        ],
    )
    dev = args.output / "dev"
    evaluation = [
        sys.executable,
        "-u",
        "-m",
        "dataset.benchmark.cpu_evaluate",
        "--data",
        str(args.data.resolve()),
        "--adapter",
        str(adapter.resolve()) if args.backend == "hf" else str(gguf.resolve()),
        "--backend",
        args.backend,
        "--label",
        args.output.name,
    ]
    command("development", [*evaluation, "--output", str(dev.resolve())])
    summary = json.loads((dev / "summary.json").read_text())
    ready = (
        summary["complete"]
        and summary["tasks"] == 60
        and summary["success_rate"] >= 0.9
        and all(
            group["passed"] / group["tasks"] >= 0.6
            for group in summary["by_family"].values()
        )
    )
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


if __name__ == "__main__":
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
    run(parser.parse_args())
