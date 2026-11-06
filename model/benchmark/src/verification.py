import json
import argparse
from pathlib import Path
from contextlib import closing
from model.benchmark.src.corpus import adapter_weights, load_tasks, sha256, read_jsonl
from model.benchmark.src.execution import Worker
from model.benchmark.src.proof_policy import MAX_CALLS, FINAL_TASKS, final_passed, development_ready

ROOT = Path(__file__).resolve().parents[1]


def replay(worker, record, prediction):
    result = worker.execute(record["setup"], "reset")
    if result["exception"]:
        raise RuntimeError(f"Fixture failed: {record['id']}")
    checks = [record["check"]]
    if record["followup"]:
        checks.append(record["followup"]["check"])
    stage = 0
    calls = 0
    for step in prediction["steps"]:
        if step["stage"] != stage:
            raise ValueError(f"Invalid stage sequence: {record['id']}")
        calls += 1
        if calls > MAX_CALLS:
            raise ValueError(f"Call budget exceeded: {record['id']}")
        if not step["end_marker_seen"] or not step["source"].strip():
            return "boundary_error"
        result = worker.execute(step["source"])
        if result["done"] is not None and not result["exception"]:
            grade = worker.execute(checks[stage])
            if grade["exception"]:
                return "incorrect"
            calls = 0
            stage += 1
            if stage == len(checks):
                return "passed"
            worker.execute("saved_result = result")
    return "call_limit"


def verify(output, data, development):
    summary = json.loads((output / "summary.json").read_text())
    dev = json.loads((development / "summary.json").read_text())
    if not summary["complete"] or summary["split"] != "test":
        raise ValueError("The final evaluation has not finished")
    if dev["split"] != "dev":
        raise ValueError("Development results must use the dev split")
    if not development_ready(dev):
        raise ValueError("Development gate was not met")
    _, dev_digest, _ = load_tasks(data, "dev")
    if dev_digest != dev["task_file_sha256"]:
        raise ValueError("Development tasks differ from the frozen manifest")
    if dev["adapter_sha256"] != summary["adapter_sha256"]:
        raise ValueError("Development and final results used different adapters")
    if any(dev[key] != summary[key] for key in ("base_model", "base_revision", "adapter_config_sha256", "run_sha256", "inference")):
        raise ValueError("Development and final model identity or inference settings differ")
    _, task_digest, tasks = load_tasks(data, "test", final=True)
    if task_digest != summary["task_file_sha256"]:
        raise ValueError("Final tasks differ from the frozen manifest")
    adapter = adapter_weights(Path(summary["adapter"]))
    adapter_digest = sha256(adapter)
    if adapter_digest != summary["adapter_sha256"]:
        raise ValueError("Adapter differs from the one evaluated")
    if sha256(adapter.parent / "adapter_config.json") != summary["adapter_config_sha256"]:
        raise ValueError("Adapter configuration differs from the one evaluated")
    if summary["run_sha256"] is not None and sha256(adapter.parent / "run.json") != summary["run_sha256"]:
        raise ValueError("Artifact provenance differs from the one evaluated")
    predictions = read_jsonl(output / "responses.jsonl")
    if len(tasks) != FINAL_TASKS or [row["id"] for row in predictions] != [row["id"] for row in tasks]:
        raise ValueError("Final predictions do not cover all 120 tasks exactly once")
    passed = 0
    mismatches = []
    with closing(Worker()) as worker:
        for task, prediction in zip(tasks, predictions, strict=True):
            status = replay(worker, task, prediction)
            passed += status == "passed"
            if status != prediction["status"]:
                mismatches.append({"id": task["id"], "original": prediction["status"], "replay": status})
    verified = not mismatches and passed == summary["passed"] and final_passed(len(tasks), passed)
    report = {
        "tasks": len(tasks),
        "original_passed": summary["passed"],
        "replayed_passed": passed,
        "success_rate": passed / len(tasks),
        "status_mismatches": mismatches,
        "adapter_sha256": adapter_digest,
        "adapter_config_sha256": summary["adapter_config_sha256"],
        "base_model": summary["base_model"],
        "base_revision": summary["base_revision"],
        "run_sha256": summary["run_sha256"],
        "inference": summary["inference"],
        "test_tasks_sha256": task_digest,
        "backend": summary["backend"],
        "objective_verified": verified,
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not verified:
        raise RuntimeError("The evaluation did not meet its independent replay checks")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/cpu-proof-v1")
    args = parser.parse_args(argv)
    verify(args.output, args.data, args.development)
