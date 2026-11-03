"""Export a saved checkpoint and append comparable worker benchmark metrics."""

import sys
import json
import argparse
import subprocess
from pathlib import Path
from dataset.benchmark.syntax import check_source

ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = ROOT.parents[1] / "model/train"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--serialization", choices=("chat", "raw"), required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    adapter = checkpoint / "adapter.gguf"
    row = {"step": args.step, "checkpoint": str(checkpoint), "status": "evaluation_error"}
    try:
        with (checkpoint / "export.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(TRAIN_ROOT / "artifacts/llama.cpp-b12038/convert_lora_to_gguf.py"),
                    "--base",
                    str(TRAIN_ROOT / "artifacts/base"),
                    "--outfile",
                    str(adapter),
                    str(checkpoint),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        name = checkpoint.parent.name if checkpoint.name.startswith("checkpoint-") else checkpoint.name
        label = f"{name}-step{args.step}"
        command = [
            sys.executable,
            "-u",
            "-m",
            "dataset.benchmark.evaluate",
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
        with (checkpoint / "evaluation.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT.parents[1], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                    if line.startswith("Results: "):
                        run_directory = Path(line.removeprefix("Results: ").strip())
                returncode = process.wait()
            except BaseException:
                process.terminate()
                process.wait(timeout=30)
                raise
        if returncode not in (0, 1) or run_directory is None:
            raise RuntimeError(f"Evaluation failed ({returncode}); see {checkpoint / 'evaluation.log'}")
        summary = json.loads((run_directory / "summary.json").read_text())
        model_responses = []
        for path in run_directory.glob("*/trace.jsonl"):
            for line in path.read_text().splitlines():
                event = json.loads(line)
                if event["event"] == "model_response":
                    for message in event["messages"]:
                        code = "".join(part["content"] for part in message["parts"] if part["part_kind"] == "text")
                        model_responses.append((code, message.get("provider_details") or {}))
        valid = sum(check_source(code) is None for code, _ in model_responses)
        row.update(
            status="completed" if summary["trials"] == 10 else "partial",
            results=str(run_directory),
            cases=summary["trials"],
            passed=summary["passed"],
            task_success_rate=summary["success_rate"],
            generated_responses=len(model_responses),
            valid_python=valid,
            raw_python_rate=valid / len(model_responses) if model_responses else None,
            end_markers=sum(bool(details.get("end_marker_seen")) for _, details in model_responses),
            first_text_seconds=summary["first_text_seconds"],
            complete_code_seconds=summary["complete_code_seconds"],
            mean_calls=summary["mean_calls"],
            statuses=summary["statuses"],
        )
        print(
            f"Checkpoint {args.step}: {summary['passed']}/{summary['trials']} tasks passed; {valid}/{len(model_responses)} raw responses compile.",
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


if __name__ == "__main__":
    raise SystemExit(main())
