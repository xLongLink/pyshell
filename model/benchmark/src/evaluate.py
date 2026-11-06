import json
from collections import Counter
from contextlib import closing

from model.benchmark.src import settings
from model.benchmark.src.artifact import load_artifact
from model.benchmark.src.corpus import load_tasks, read_jsonl, resolve_data, sha256
from model.benchmark.src.execution import Worker
from model.benchmark.src.proof_policy import MAX_CALLS, final_passed
from model.benchmark.src.proof_trial import trial


def inference_settings(config):
    return {
        "device": config.device,
        "dtype": "float32",
        "threads": config.threads,
        "max_new_tokens": config.tokens,
        "max_calls_per_stage": MAX_CALLS,
        "do_sample": False,
        "tf32": False,
    }


def evaluate(artifact, config, output, *, data, split="dev", final=False):
    manifest, digest, records = load_tasks(data, split, final=final)
    if not records:
        raise ValueError("Evaluation tasks must not be empty")
    artifact.verify()
    from model.benchmark.src.inference import HFCompletion

    output.mkdir(parents=True, exist_ok=False)
    groups = {}
    statuses = Counter()
    generate = HFCompletion(
        artifact.path, config.threads, config.device, base_model=artifact.base, revision=artifact.revision, max_new_tokens=config.tokens
    )
    with closing(Worker()) as worker, (output / "responses.jsonl").open("a") as file:
        for index, record in enumerate(records, 1):
            row = trial(worker, record, generate)
            file.write(json.dumps(row) + "\n")
            file.flush()
            group = groups.setdefault(row["family"], {"tasks": 0, "passed": 0})
            group["tasks"] += 1
            group["passed"] += row["status"] == "passed"
            statuses[row["status"]] += 1
            summary = {
                "label": output.name,
                "split": split,
                "complete": False,
                "tasks": index,
                "passed": statuses["passed"],
                "success_rate": statuses["passed"] / index,
                "by_family": dict(sorted(groups.items())),
                "statuses": dict(statuses),
                "task_file_sha256": digest,
                "adapter": str(artifact.path),
                "adapter_sha256": artifact.weights_sha256,
                "adapter_config_sha256": artifact.config_sha256,
                "base_model": artifact.base,
                "base_revision": artifact.revision,
                "run_sha256": artifact.run_sha256,
                "inference": inference_settings(config),
                "backend": "hf",
                "runtime": generate.runtime,
                "scope": manifest["scope"],
                "isolation": "network-disabled container; fresh PythonShell per task; no reference prior context",
                "acceptance_met": False,
            }
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"{output.name} {split} {index}/{len(records)}: {summary['passed']} passed; {row['family']}={row['status']}", flush=True)
    artifact.verify()
    if sha256(data / f"{split}-tasks.jsonl") != digest:
        raise RuntimeError("Frozen tasks changed during evaluation")
    summary.update(complete=True, acceptance_met=split == "test" and final_passed(len(records), summary["passed"]))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def comparison(baseline, candidate):
    before = json.loads((baseline / "summary.json").read_text())
    after = json.loads((candidate / "summary.json").read_text())
    if not before["complete"] or not after["complete"] or before["split"] != "dev" or after["split"] != "dev":
        raise ValueError("Comparison requires complete development results")
    if before["task_file_sha256"] != after["task_file_sha256"] or before["inference"] != after["inference"]:
        raise ValueError("Comparison requires identical tasks and inference settings")
    previous = read_jsonl(baseline / "responses.jsonl")
    current = read_jsonl(candidate / "responses.jsonl")
    if (
        len(previous) != before["tasks"]
        or len(current) != after["tasks"]
        or [(r["id"], r["family"]) for r in previous] != [(r["id"], r["family"]) for r in current]
    ):
        raise ValueError("Comparison requires identical ordered task coverage")
    gains = [b["id"] for a, b in zip(previous, current, strict=True) if a["status"] != "passed" and b["status"] == "passed"]
    regressions = [b["id"] for a, b in zip(previous, current, strict=True) if a["status"] == "passed" and b["status"] != "passed"]
    fields = ("passed", "success_rate", "adapter_sha256", "adapter_config_sha256", "base_model", "base_revision", "run_sha256")
    return {
        "split": "dev",
        "tasks": after["tasks"],
        "task_file_sha256": after["task_file_sha256"],
        "baseline": {key: before[key] for key in fields},
        "candidate": {key: after[key] for key in fields},
        "improved": after["passed"] > before["passed"],
        "delta": after["success_rate"] - before["success_rate"],
        "gains": gains,
        "regressions": regressions,
        "by_family": {
            family: {
                "tasks": group["tasks"],
                "baseline_passed": before["by_family"][family]["passed"],
                "candidate_passed": group["passed"],
                "delta": group["passed"] - before["by_family"][family]["passed"],
                "baseline_success_rate": before["by_family"][family]["passed"] / group["tasks"],
                "candidate_success_rate": group["passed"] / group["tasks"],
            }
            for family, group in after["by_family"].items()
        },
    }


def run(config):
    if config.output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark results: {config.output}")
    candidate = load_artifact(config.candidate, candidate=True)
    baseline = load_artifact(config.baseline)
    data = resolve_data(config, "dev")
    load_tasks(data, "dev")
    config.output.mkdir(parents=True)
    (config.output / "settings.json").write_text(config.model_dump_json(indent=2) + "\n")
    evaluate(baseline, config, config.output / "baseline", data=data)
    evaluate(candidate, config, config.output / "candidate", data=data)
    report = comparison(config.output / "baseline", config.output / "candidate")
    (config.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    run(settings.SETTINGS)
