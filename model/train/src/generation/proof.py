"""Freeze independent execution tasks for a CPU fine-tuning experiment."""

import argparse
import json
import random
import secrets
from dataclasses import asdict
from pathlib import Path

from model.benchmark.src.corpus import sha256 as digest
from model.benchmark.src.corpus import write_jsonl
from model.benchmark.src.proof_policy import acceptance_metadata
from model.benchmark.src.reference import ReferenceSession
from model.benchmark.src.task_families import BUILDERS, FixtureSeed
from model.benchmark.src.task_types import CONTRACT, FAMILIES, TaskFixture, TaskRecord

ROOT = Path(__file__).resolve().parents[3] / "benchmark"


def task(family, index, seed, split, *, name=None):
    return asdict(build_task(family, index, seed, split, name=name).record)


def build_task(family, index, seed, split, *, name=None):
    builder = BUILDERS[family]
    rng = random.Random(seed + index * 104729)
    prefix = {"train": "source", "dev": "workspace", "test": "input"}[split]
    default_name = f"{prefix}_{rng.randrange(100000, 999999)}"
    name = default_name if name is None else name
    other = name + "_other"
    count = rng.randint(4, 10 if split == "train" else 14)
    values = rng.sample(range(-30, 81) if split == "train" else range(-300, 501), count)
    labels = [rng.choice(["copper", "silver", "gold"]) for _ in values]
    content = builder(FixtureSeed(rng, name, values, labels))
    instruction = (
        content.request if split == "train" and index % 3 else content.alternate
    )
    if split == "test":
        instruction = {
            "statistics": f"I need four descriptive measures of {name}. Put its length under count, arithmetic average under mean, middle-value statistic under median, and population standard deviation under population_std in one dict.",
            "grouped_revenue": f"Inspect frame {name}: identify its numeric measurement column, group-label column, and lifecycle-state column. Sum measurements per group, leaving out rows in the cancelled state. Return a normal dict without mutating the frame.",
            "monthly_totals": f"Turn the dated transactions in {name} into a mapping of calendar months to total amounts. Each date is YYYY-MM-DD; use YYYY-MM keys, and do not fabricate absent months.",
            "missing_values": f"Produce a sensor-to-average-reading mapping from {name}. Null readings contribute nothing to either sum or count. A sensor with no usable readings still needs an entry with value None.",
            "left_join": f"Match every order in {name} to its customer in {other} using customer_id. Keep the original order sequence and unmatched orders. Emit list entries with exactly order_id and customer_name; use Python None when a name is missing.",
            "inspect_callable": content.alternate.replace(
                "Use the existing converter", "Discover the call signature for"
            ).replace("Read its help to discover", "Consult its documentation for"),
            "recover_execution": f"The existing object {name} is a reader that may start disconnected. Sum what read() returns. When disconnection occurs, invoke recover() on this same object and read again; a recovery return value is not the data.",
            "stateful_followup": f"Make a list of the squared numbers from {name}, assign it to result, and finish this request. Keep this list object alive for the next instruction.",
            "missing_input": f"A calculation requires the exact input named {name}. Look for it in the workspace. If it is unavailable, leave result as None and status as 'missing_data'; otherwise sum it. Never use a different object as a substitute.",
            "linear_system": f"Determine a vector x such that multiplying matrix {name} by x yields vector {other}. Use the provided arrays, store x as an ordinary list, and confirm the multiplication reproduces the right-hand side.",
            "filtered_means": f"Records in {name} carry a group, a value, and an active flag. Report average value separately for each group using active records only. Return a dictionary with no inactive-only groups.",
            "joined_totals": f"Compute spending per group from {name}. Each row supplies item and quantity; dictionary {other} supplies item prices. Add quantity times price to the row's group, but discard rows whose item has no listed price.",
        }[family]
    record = TaskRecord(
        id=f"{split}-{family}-{index:04d}",
        family=family,
        setup=content.setup,
        instruction=instruction + "\n" + CONTRACT,
        inspect=content.inspection,
        code=content.code + "\ndone(str(result))",
        check=content.check,
        followup=content.followup,
    )
    return TaskFixture(record, values, content.converter, content.extra)


def supervision(record):
    session = ReferenceSession(record)
    for index, source in enumerate((record["inspect"], record["code"])):
        session.add(source, f"{record['id']}-{index}")
    session.check(record["check"])
    if record["followup"]:
        session.followup(record["followup"], record["id"] + "-followup")
    return session.rows


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    test_seed = secrets.randbits(48)
    splits = {}
    for split, per_family, seed in (
        ("train", 30, 713),
        ("dev", 5, 9713),
        ("test", 10, test_seed),
    ):
        records = [
            task(family, index, seed + family_index * 10000019, split)
            for family_index, family in enumerate(FAMILIES)
            for index in range(per_family)
        ]
        path = output / f"{split}-tasks.jsonl"
        write_jsonl(path, records)
        splits[split] = {"tasks": len(records), "sha256": digest(path)}
        if split in {"train", "dev"}:
            rows = [row for record in records for row in supervision(record)]
            name = "train.jsonl" if split == "train" else "validation.jsonl"
            write_jsonl(output / name, rows)
    manifest = {
        "experiment": "cpu-execution-proof-v1",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "test_seed": test_seed,
        "scope": "unseen task instances and wording across 12 supported families; not unseen task families or UI tasks",
        "acceptance": acceptance_metadata(),
        "selection_split": "dev",
        "test_policy": "Do not inspect test predictions or tune after the final test. A failed final test remains a reported failure.",
        "splits": splits,
        "training_records_sha256": digest(output / "train.jsonl"),
        "validation_records_sha256": digest(output / "validation.jsonl"),
        "generator_sha256": digest(Path(__file__)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/cpu-proof-v1")
    prepare(parser.parse_args(argv).output)
