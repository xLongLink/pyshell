"""Freeze new binding evaluation instances before checkpoint selection."""

import argparse
import json
import re
import secrets
from collections import Counter
from contextlib import closing
from pathlib import Path

from model.benchmark.src.corpus import sha256, write_jsonl
from model.benchmark.src.execution import Worker
from model.benchmark.src.proof_policy import acceptance_metadata
from model.benchmark.src.reference import ReferenceSession
from model.benchmark.src.task_types import CONTRACT, FAMILIES
from model.train.src.generation import binding
from model.train.src.generation.transfer import make_task as transfer_task


def evaluation_task(family, pair, variant, seed, split):
    if split not in ("dev", "test"):
        raise ValueError("Evaluation split must be dev or test")
    if family in binding.REQUESTS:
        return binding.make_task(family, pair, variant, seed, split)
    record = transfer_task(
        family, pair, variant, seed, split, 6 if split == "dev" else 8
    )
    name = re.search(r"\b[a-z]+_[0-9]+\b", record["instruction"])[0]
    other = name + "_other"
    aggregate = "mean" if variant else "sum"
    requests = {
        "statistics": f"Use {name} to return "
        + (
            "only its median as one scalar."
            if variant
            else "a dictionary keyed count, mean, median, population_std; use population standard deviation."
        ),
        "grouped_revenue": f"Inspect the columns and their live values in {name} to identify group labels, numeric amounts and lifecycle states. Remove cancelled rows and return the {aggregate} amount per group in a dictionary.",
        "monthly_totals": f"Derive YYYY-MM keys from the date fields in {name} and sum amount per key. "
        + (
            "Restrict output to observed 2031-01 and 2031-02 keys."
            if variant
            else "Keep every observed month and no absent months."
        ),
        "missing_values": f"For the sensors of {name}, return the {'sum' if variant else 'average'} of non-null reading values, keyed by sensor. Include all-null sensors with None, not zero.",
        "left_join": f"Resolve names for orders {name} through customers {other} using customer_id. Return ordered dictionaries containing order_id and customer_name. "
        + (
            "Omit unmatched orders."
            if variant
            else "Keep unmatched orders with customer_name=None."
        ),
        "recover_execution": f"Fetch numbers with {name}.read() and return their {'arithmetic mean' if variant else 'sum'}. A RuntimeError means recover() on the same object, then retry read(); recover() is not the numeric payload.",
        "stateful_followup": f"Preserve the source {name}. Store each value raised to power {2 + variant} in a new list result, retaining that list for a subsequent in-place update.",
        "missing_input": f"Look up exactly {name} in the workspace, not a similar name. If available, sum it; otherwise leave result=None and status='missing_data'.",
        "linear_system": f"Solve the existing system {name} @ x = {other} and verify its residual. "
        + (
            "Return the sum of x's components as a scalar."
            if variant
            else "Return x as a plain Python list."
        ),
        "filtered_means": f"Use records in {name} with active=True only. Return the {'sum' if variant else 'arithmetic mean'} of value for each group as a dictionary, excluding groups without active rows.",
    }
    wrapper = (
        "Complete this operation on the current workspace: "
        if split == "dev"
        else "Carry out the following request using the supplied live objects: "
    )
    record["instruction"] = wrapper + requests[family] + "\n" + CONTRACT
    record["template_id"] = f"binding-{family}-{split}"
    record["id"] = f"{split}-binding-{family}-{pair:04d}-v{variant}"
    record["contrast_group"] = f"{split}-binding-{family}-{pair:04d}"
    if record["followup"]:
        original = record["followup"]["instruction"]
        record["followup"]["instruction"] = (
            "Next, update the very same result list, keeping its identity: " + original
        )
    return record


def prepare(output, seed=None):
    output.mkdir(parents=True, exist_ok=False)
    seed = secrets.randbits(48) if seed is None else seed
    metadata, setups = {}, {}
    validation = []
    with closing(Worker()) as worker:
        for split, count, offset in (("dev", 5, 0), ("test", 10, 1000000007)):
            records = []
            for family_index, family in enumerate(FAMILIES):
                for index in range(count):
                    record = evaluation_task(
                        family,
                        index // 2,
                        index % 2,
                        seed + offset + family_index * 10000019,
                        split,
                    )
                    session = ReferenceSession(
                        record, worker, capture_rows=split == "dev"
                    )
                    session.add(record["inspect"], record["id"] + "-inspect")
                    session.add(record["code"], record["id"] + "-solve")
                    session.check(record["check"])
                    if record["followup"]:
                        session.followup(record["followup"], record["id"] + "-followup")
                    if split == "dev":
                        validation.extend(session.rows)
                    if "negative_control" in record:
                        binding.verify_negative_control(record, worker)
                    record.pop("expected", None)
                    records.append(record)
            path = output / f"{split}-tasks.jsonl"
            write_jsonl(path, records)
            metadata[split] = {
                "tasks": len(records),
                "sha256": sha256(path),
                "families": dict(Counter(r["family"] for r in records)),
                "template_ids": sorted({r["template_id"] for r in records}),
            }
            setups[split] = {r["setup"] for r in records}
    if setups["dev"] & setups["test"]:
        raise ValueError("Development and final workspaces overlap")
    write_jsonl(output / "validation.jsonl", validation)
    source_root = Path(__file__).resolve().parents[4]
    snapshot = output / "sources"
    snapshot.mkdir()
    sources = {}
    for directory in (
        "model/train/src/generation",
        "model/benchmark/src",
        "model/protocol/src/pyshell",
    ):
        for source in sorted((source_root / directory).glob("*.py")):
            relative = source.relative_to(source_root)
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            sources[str(relative)] = sha256(target)
    manifest = {
        "experiment": "gpu-generalization-v2",
        "seed": seed,
        "scope": "New instances and API/schema combinations across twelve supported families; fresh evaluation wording; not unseen families or arbitrary Python",
        "acceptance": acceptance_metadata(),
        "splits": metadata,
        "selection_split": "dev",
        "development_each_family_minimum": 0.8,
        "validation_records_sha256": sha256(output / "validation.jsonl"),
        "source_sha256": sources,
        "old_test_reused": False,
        "reference_validation": "All reference trajectories and target-family negative controls executed in isolated containers before model selection",
        "independence": "Expected numeric results use separate NumPy/pandas calculations from emitted reference scripts; final replay uses the benchmark verifier. No external human review is claimed.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    receipt = {
        "experiment": manifest["experiment"],
        "splits": metadata,
        "manifest_sha256": sha256(output / "manifest.json"),
    }
    print(json.dumps(receipt, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare(args.output)
