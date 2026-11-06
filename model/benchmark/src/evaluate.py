import json
import argparse
from pathlib import Path
from contextlib import closing
from collections import Counter
from model.benchmark.src.corpus import adapter_weights, sha256, load_tasks
from model.benchmark.src.execution import Worker
from model.benchmark.src.proof_trial import trial
from model.benchmark.src.proof_policy import final_passed

ROOT = Path(__file__).resolve().parents[1]


def evaluate(args):
    manifest, digest, records = load_tasks(args.data, args.split, final=args.final, limit=args.limit)
    from model.benchmark.src.inference import HFCompletion

    adapter_file = adapter_weights(args.adapter) if args.adapter else None
    adapter_digest = sha256(adapter_file) if adapter_file else None
    args.output.mkdir(parents=True, exist_ok=False)
    groups = {}
    statuses = Counter()
    generate = HFCompletion(args.adapter, args.threads, args.device, base_model=args.base_model)
    with closing(Worker()) as worker, (args.output / "responses.jsonl").open("a") as file:
        for index, record in enumerate(records, 1):
            row = trial(worker, record, generate)
            file.write(json.dumps(row) + "\n")
            file.flush()
            group = groups.setdefault(row["family"], {"tasks": 0, "passed": 0})
            group["tasks"] += 1
            group["passed"] += row["status"] == "passed"
            statuses[row["status"]] += 1
            passed = statuses["passed"]
            summary = {
                "label": args.label,
                "split": args.split,
                "complete": index == len(records),
                "tasks": index,
                "passed": passed,
                "success_rate": passed / index,
                "by_family": dict(sorted(groups.items())),
                "statuses": dict(statuses),
                "task_file_sha256": digest,
                "adapter": str(args.adapter) if args.adapter else None,
                "adapter_sha256": adapter_digest,
                "backend": "hf",
                "device": args.device,
                "dtype": "float32",
                "runtime": generate.runtime,
                "scope": manifest["scope"],
                "isolation": "network-disabled container; fresh PythonShell per task; no reference prior context",
                "acceptance_met": args.split == "test" and final_passed(index, passed),
            }
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"{args.label} {args.split} {index}/{len(records)}: {passed} passed; {record['family']}={row['status']}", flush=True)
    if adapter_file and sha256(adapter_file) != adapter_digest:
        raise RuntimeError("Adapter changed during evaluation")
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/cpu-proof-v1")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--base-model", type=Path, default=ROOT / "local/base")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate(parser.parse_args(argv))
