"""Build a broader training-only curriculum without reading final-test tasks."""

import argparse
import contextlib
import hashlib
import io
import json
import random
import statistics
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.benchmark.cpu_tasks import FAMILIES, task
from dataset.benchmark.src.serialization import serialize

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
NAMES = (
    "records",
    "readings",
    "samples",
    "payload",
    "workspace",
    "observations",
    "data",
    "table",
    "values",
    "existing",
    "series",
    "entries",
)
FOCUS = {
    "grouped_revenue",
    "left_join",
    "missing_values",
    "recover_execution",
    "filtered_means",
}


def feedback(source, namespace):
    output, errors = io.StringIO(), io.StringIO()
    done = []
    namespace["done"] = lambda message: done.append(message)
    error = None
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
        try:
            exec(compile(source, "<agent>", "exec"), namespace)
        except Exception:
            error = traceback.format_exc()
    return {
        "stdout": output.getvalue(),
        "stderr": errors.getvalue(),
        "exception": error,
        "done": done[-1] if done and error is None else None,
    }


def prepare(output, evaluation):
    frozen = json.loads((evaluation / "manifest.json").read_text())
    rng = random.Random(440713)
    rows = []
    task_counts = Counter()
    failures = 0
    for family_index, family in enumerate(FAMILIES):
        for index in range(50 if family in FOCUS else 25):
            record = task(family, index, 440713 + family_index * 10000019, "train")
            old_name = (
                record["setup"].split(" = ")[0]
                if family != "recover_execution"
                else None
            )
            if family == "recover_execution":
                old_name = record["inspect"][5:-6]
            elif family == "missing_input":
                old_name = old_name.removesuffix("_other")
            name = f"{rng.choice(NAMES)}_{rng.randrange(1000, 99999)}"
            encoded = json.dumps(record).replace(old_name, name)
            record = json.loads(encoded)
            if family == "grouped_revenue":
                record["code"] = record["code"].replace(
                    "result = " + name + ".loc[", "filtered = " + name + ".loc["
                )
                marker = "].groupby("
                record["code"] = record["code"].replace(
                    marker, "]\nresult = filtered.groupby("
                )
            namespace = {"np": np, "pd": pd, "statistics": statistics, "json": json}
            exec(record["setup"], namespace)
            turns = [{"role": "user", "content": record["instruction"]}]

            def add(source, *, supervise=True, expect_error=False):
                nonlocal failures
                if supervise:
                    rows.append(
                        {
                            "id": f"curriculum-{family}-{index:04d}-{len(turns)}",
                            "family": family,
                            "prompt": serialize(turns),
                            "completion": source,
                        }
                    )
                result = feedback(source, namespace)
                if bool(result["exception"]) != expect_error:
                    raise AssertionError((family, source, result))
                failures += expect_error
                turns.extend(
                    [
                        {"role": "assistant", "content": source},
                        {"role": "user", "content": json.dumps(result)},
                    ]
                )

            add(record["inspect"])
            if family == "recover_execution" and index % 2 == 0:
                add(f"result = sum({name}.read())", expect_error=True)
                add(f"{name}.recover()\nresult = sum({name}.read())\ndone(str(result))")
            else:
                if (
                    family in {"left_join", "missing_values", "grouped_revenue"}
                    and index % 3 == 0
                ):
                    add(
                        f"print({name}['column_that_is_not_present'])",
                        supervise=False,
                        expect_error=True,
                    )
                add(record["code"])
            exec(record["check"], namespace)
            if record["followup"]:
                namespace["saved_result"] = namespace["result"]
                turns.append(
                    {"role": "user", "content": record["followup"]["instruction"]}
                )
                add(record["followup"]["code"] + "\ndone(str(result))")
                exec(record["followup"]["check"], namespace)
            task_counts[family] += 1
    output.mkdir(parents=True, exist_ok=False)
    train = output / "train.jsonl"
    train.write_text("".join(json.dumps(row) + "\n" for row in rows))
    validation = output / "validation.jsonl"
    validation.write_bytes((evaluation / "validation.jsonl").read_bytes())
    manifest = {
        "training_only": True,
        "tasks_by_family": dict(task_counts),
        "training_records": len(rows),
        "verified_error_trajectories": failures,
        "variable_prefixes": NAMES,
        "evaluation_test_sha256": frozen["splits"]["test"]["sha256"],
        "training_records_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "validation_records_sha256": hashlib.sha256(
            validation.read_bytes()
        ).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/cpu-proof-curriculum-v2"
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=REPO / "dataset/benchmark/artifacts/cpu-proof-v1",
    )
    args = parser.parse_args()
    prepare(args.output, args.evaluation)
