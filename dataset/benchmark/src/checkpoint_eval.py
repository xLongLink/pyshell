"""Evaluate a saved checkpoint and append comparable worker benchmark metrics."""

import sys
import json
import argparse
import subprocess
from pathlib import Path
from model.train.src.export import export_command
from model.train.src.processes import run_logged
from dataset.benchmark.src.cases import CASES
from dataset.benchmark.src.syntax import check_source

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT.parents[1] / "model/train"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--serialization", choices=("chat", "raw"), required=True)
    parser.add_argument("--gguf", type=Path)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    adapter = args.gguf.resolve() if args.gguf is not None else checkpoint / "adapter.gguf"
    row = {"step": args.step, "checkpoint": str(checkpoint), "status": "evaluation_error"}
    try:
        if args.gguf is not None:
            if not adapter.is_file():
                raise FileNotFoundError(adapter)
        else:
            with (checkpoint / "export.log").open("w") as log:
                subprocess.run(export_command(checkpoint, adapter), stdout=log, stderr=subprocess.STDOUT, check=True)
        name = checkpoint.parent.name if checkpoint.name.startswith("checkpoint-") else checkpoint.name
        label = f"{name}-step{args.step}"
        command = [
            sys.executable,
            "-u",
            "-m",
            "dataset.benchmark.main",
            "evaluate",
            "--adapter",
            str(adapter),
            "--serialization",
            args.serialization,
            "--temperature",
            "0",
            "--label",
            label,
        ]
        run_directory = None

        def observe(line):
            nonlocal run_directory
            if line.startswith("Results: "):
                run_directory = Path(line.removeprefix("Results: ").strip())

        returncode = run_logged(command, checkpoint / "evaluation.log", cwd=ROOT.parents[1], on_line=observe)
        if returncode not in (0, 1) or run_directory is None:
            raise RuntimeError(f"Evaluation failed ({returncode}); see {checkpoint / 'evaluation.log'}")
        summary = json.loads((run_directory / "summary.json").read_text())
        responses = valid = end_markers = 0
        for path in run_directory.glob("*/trace.jsonl"):
            with path.open() as trace:
                for line in trace:
                    event = json.loads(line)
                    if event["event"] == "model_response":
                        for message in event["messages"]:
                            code = "".join(part["content"] for part in message["parts"] if part["part_kind"] == "text")
                            responses += 1
                            valid += check_source(code) is None
                            end_markers += bool((message.get("provider_details") or {}).get("end_marker_seen"))
        row.update(
            status="completed" if summary["trials"] == len(CASES) else "partial",
            results=str(run_directory),
            cases=summary["trials"],
            passed=summary["passed"],
            task_success_rate=summary["success_rate"],
            generated_responses=responses,
            valid_python=valid,
            raw_python_rate=valid / responses if responses else None,
            end_markers=end_markers,
            first_text_seconds=summary["first_text_seconds"],
            complete_code_seconds=summary["complete_code_seconds"],
            mean_calls=summary["mean_calls"],
            statuses=summary["statuses"],
        )
        print(
            f"Checkpoint {args.step}: {summary['passed']}/{summary['trials']} tasks passed; {valid}/{responses} raw responses compile.",
            flush=True,
        )
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        print(f"Checkpoint {args.step} evaluation failed: {row['error']}", flush=True)
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a") as file:
        file.write(json.dumps(row) + "\n")
    if row["status"] == "evaluation_error":
        return 2
    return 0 if row["status"] == "completed" and row["passed"] == row["cases"] else 1
