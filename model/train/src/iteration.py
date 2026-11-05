"""Run sequential development-selected CPU pilots, opening the final set at most once."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

from model.benchmark.src.corpus import read_jsonl
from model.benchmark.src.verification import verify
from model.train.src.experiment import run as experiment

REPO = Path(__file__).resolve().parents[3]


def comparison(before, after):
    previous = read_jsonl(before / "responses.jsonl")
    current = read_jsonl(after / "responses.jsonl")
    if [row["id"] for row in previous] != [row["id"] for row in current]:
        raise ValueError("Paired comparison requires identical ordered tasks")
    gained = sum(
        a["status"] != "passed" and b["status"] == "passed"
        for a, b in zip(previous, current, strict=True)
    )
    lost = sum(
        a["status"] == "passed" and b["status"] != "passed"
        for a, b in zip(previous, current, strict=True)
    )
    n = gained + lost
    p = (
        min(1.0, 2 * sum(math.comb(n, k) for k in range(min(gained, lost) + 1)) / 2**n)
        if n
        else 1.0
    )
    return {"paired_gains": gained, "paired_regressions": lost, "exact_mcnemar_p": p}


def selection_score(summary):
    return summary["passed"], min(
        group["passed"] / group["tasks"] for group in summary["by_family"].values()
    )


def run(args):
    diagnostic = json.loads(args.diagnostic.read_text())
    if not diagnostic["complete"]:
        raise ValueError("Training-set diagnostic did not complete")
    args.output.mkdir(parents=True, exist_ok=False)
    state_path = args.output / "loop.json"
    best = args.initialize_adapter
    best_dev = args.baseline_dev
    best_summary = json.loads((best_dev / "summary.json").read_text())
    state = {
        "status": "running",
        "objective_met": False,
        "training_diagnostic": diagnostic,
        "history": [],
        "final_policy": "Open the fixed final set at most once, after development qualifies. A failed final remains a failure.",
    }
    for index, budget in enumerate(args.budgets):
        output = args.output / f"iteration-{index + 1}-{budget}"
        state.update(
            active_iteration=index + 1,
            best_adapter=str(best),
            best_development=best_summary,
        )
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        result = experiment(
            SimpleNamespace(
                data=args.data,
                training_data=args.training_data,
                output=output,
                steps=budget,
                threads=args.threads,
                device=args.device,
                learning_rate=args.learning_rate / 2**index,
                initialize_adapter=best,
                final_if_ready=True,
            )
        )
        summary = result["development"]
        paired = comparison(best_dev, output / "dev")
        record = {
            "iteration": index + 1,
            "examples": budget,
            "initialized_from": str(best),
            "passed": summary["passed"],
            "tasks": summary["tasks"],
            "by_family": summary["by_family"],
            **paired,
            "output": str(output),
        }
        (output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
        state["history"].append(record)
        if selection_score(summary) > selection_score(best_summary):
            best, best_dev, best_summary = output / "adapter", output / "dev", summary
        print(json.dumps(record, indent=2), flush=True)
        if result["status"] == "complete":
            state.update(status="verifying_final", final_experiment=str(output))
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            try:
                verify(output, args.data)
            finally:
                verification = output / "test/verification.json"
                report = (
                    json.loads(verification.read_text())
                    if verification.exists()
                    else {}
                )
                state.update(
                    status="objective_verified"
                    if report.get("objective_verified")
                    else "final_failed",
                    objective_met=bool(report.get("objective_verified")),
                    verification=report,
                )
                state_path.write_text(json.dumps(state, indent=2) + "\n")
            return
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    state.update(
        status="needs_diagnosis", best_adapter=str(best), best_development=best_summary
    )
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(
        "Development did not qualify. Final holdout remains unopened; inspect this batch before selecting another intervention.",
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO / "model/benchmark/artifacts/cpu-transfer-v1",
    )
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--initialize-adapter", type=Path, required=True)
    parser.add_argument("--baseline-dev", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[800, 1200, 1600])
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args(argv)
    if any(budget < 1 for budget in args.budgets):
        parser.error("Training budgets must be positive")
    run(args)
