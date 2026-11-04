"""Construct API and schema binding tasks without split-dependent semantics."""

import argparse
import json
import random
from collections import Counter
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from model.benchmark.src.corpus import sha256, write_jsonl
from model.benchmark.src.execution import Worker
from model.benchmark.src.reference import ReferenceSession
from model.benchmark.src.task_types import CONTRACT

KEYWORDS = (
    ("scale", "offset"),
    ("factor", "bias"),
    ("gain", "shift"),
    ("multiplier", "translation"),
    ("slope", "intercept"),
    ("weight", "adjustment"),
)
FIELDS = (
    "code",
    "sku",
    "product",
    "item_key",
    "article",
    "reference",
    "segment",
    "team",
    "bucket",
    "region",
    "units",
    "count",
    "quantity",
    "volume",
)
REQUESTS = {
    "inspect_callable": {
        "train": "Read the help for {other}.convert. Apply {operation} to the numbers in {name} by calling that existing method with its documented keywords.",
        "dev": "Discover the signature and semantics of {other}.convert, then invoke it on {name} to perform {operation}. Do not substitute your own converter.",
        "test": "Use the live method {other}.convert, not replacement arithmetic, on {name}. Inspect its documentation to bind arguments for this transformation: {operation}.",
    },
    "joined_totals": {
        "train": "Inspect {name} and {other}. {roles} Compute {aggregate} per group, excluding records with no matching lookup key.",
        "dev": "Build a group-indexed mapping of {aggregate} by combining {name} with {other}. {roles} Unlisted items must not contribute.",
        "test": "For the live records {name}, obtain {aggregate} for each group using lookup {other}; skip unknown items. {roles}",
    },
}


def make_task(family, pair, variant, seed, split):
    if (
        family not in REQUESTS
        or split not in ("train", "dev", "test")
        or variant not in (0, 1)
    ):
        raise ValueError("Unsupported binding task")
    rng = random.Random(seed + pair * 65537)
    name, other = rng.sample(
        [
            f"{prefix}_{rng.randrange(1000, 9999999)}"
            for prefix in (
                "batch",
                "records",
                "samples",
                "inventory",
                "payload",
                "lookup",
            )
        ],
        2,
    )
    record = {
        "id": f"{split}-binding-{family}-{pair:04d}-v{variant}",
        "family": family,
        "template_id": f"binding-{family}-{split}",
        "contrast_group": f"{split}-binding-{family}-{pair:04d}",
        "variant": variant,
        "followup": None,
        "recovery_code": None,
    }
    if family == "inspect_callable":
        multiplier, shift = KEYWORDS[
            (pair + random.Random(seed).randrange(len(KEYWORDS))) % len(KEYWORDS)
        ]
        before = bool((pair + pair // len(KEYWORDS) + seed) % 2)
        values = rng.sample(range(-90, 100), 7)
        gain, offset = (
            rng.randint(2, 7) + variant * 2,
            rng.choice([-9, -5, 3, 8]) - variant * 3,
        )
        expression = (
            f"(value + {shift}) * {multiplier}"
            if before
            else f"value * {multiplier} + {shift}"
        )
        parameters = [multiplier, shift]
        rng.shuffle(parameters)
        setup = (
            f"{name} = {values!r}\nclass Converter:\n"
            "    def __init__(self):\n        self.calls = []\n"
            f"    def convert(self, values, *, {', '.join(parameters)}):\n"
            f'        """Return {expression} for each value. Options are keyword-only."""\n'
            f"        self.calls.append((list(values), {multiplier}, {shift}))\n"
            f"        return [{expression} for value in values]\n"
            f"{other} = Converter()\n_binding_original = {other}"
        )
        operation = (
            f"add {offset}, then multiply by {gain}"
            if before
            else f"multiply by {gain}, then add {offset}"
        )
        expected = (
            (np.asarray(values) + offset) * gain
            if before
            else np.asarray(values) * gain + offset
        ).tolist()
        code = (
            f"result = {other}.convert({name}, {multiplier}={gain}, {shift}={offset})"
        )
        check = (
            f"assert {other} is _binding_original\nassert isinstance(result, list)\n"
            f"assert np.allclose(result, {expected!r}, rtol=0, atol=1e-7)\n"
            f"assert ({values!r}, {gain}, {offset}) in _binding_original.calls"
        )
        record.update(
            setup=setup,
            inspect=f"print({name})\nhelp({other}.convert)",
            instruction=REQUESTS[family][split].format(
                name=name, other=other, operation=operation
            ),
            check=check,
            binding_axes={
                "keywords": parameters,
                "order": "add_then_multiply" if before else "multiply_then_add",
            },
            negative_control=f"result = {expected!r}\ndone(str(result))",
        )
    else:
        item, group, quantity, container, price = rng.sample(FIELDS, 5)
        shape = (pair + seed) % 4
        nested_rows, nested_prices = bool(shape & 1), bool(shape & 2)
        keys = rng.sample(range(100, 900), 4)
        prices = {key: rng.randint(2, 20) for key in keys[:3]}
        flat = [
            {
                "item": keys[i % 4],
                "group": ["red", "blue", "green"][i % 3],
                "quantity": rng.choice([-3, 0, 2, 5, 8]),
            }
            for i in range(12)
        ]
        flat[0]["quantity"] = 5
        flat[0]["group"] = "anchor"
        rows = [
            {
                item: row["item"],
                group: row["group"],
                quantity: row["quantity"],
                "unused": 999,
            }
            for row in flat
        ]
        if nested_rows:
            rows = [{container: row, "unused": -999} for row in rows]
        lookup = {
            key: {price: value, "unused": -777} if nested_prices else value
            for key, value in prices.items()
        }
        path = f"entry[{container!r}]" if nested_rows else "entry"
        price_expression = f"{other}[row[{item!r}]]" + (
            f"[{price!r}]" if nested_prices else ""
        )
        term = f"row[{quantity!r}]" + (f" * {price_expression}" if not variant else "")
        code = (
            f"result = {{}}\nfor entry in {name}:\n    row = {path}\n"
            f"    if row[{item!r}] in {other}:\n        key = row[{group!r}]\n"
            f"        result[key] = result.get(key, 0) + {term}"
        )
        frame = pd.DataFrame(flat)
        frame = frame[frame["item"].isin(prices)]
        frame["contribution"] = (
            frame["quantity"]
            if variant
            else frame["quantity"] * frame["item"].map(prices)
        )
        expected = frame.groupby("group")["contribution"].sum().to_dict()
        roles = (
            f"Each record's fields are {'inside ' + repr(container) if nested_rows else 'at the top level'}: "
            f"item key {item!r}, group {group!r}, quantity {quantity!r}. "
            f"Lookup keys are item identifiers; prices are {'in field ' + repr(price) + ' of each lookup value' if nested_prices else 'the lookup values'}. Ignore unused fields."
        )
        record.update(
            setup=f"{name} = {rows!r}\n{other} = {lookup!r}",
            inspect=f"print({name})\nprint({other})",
            instruction=REQUESTS[family][split].format(
                name=name,
                other=other,
                roles=roles,
                aggregate="total quantity, not spending"
                if variant
                else "total spending (quantity times unit price)",
            ),
            check=f"assert isinstance(result, dict)\nassert result == {expected!r}",
            binding_axes={
                "nested_rows": nested_rows,
                "nested_prices": nested_prices,
                "fields": [item, group, quantity, container, price],
            },
            negative_control="result = {}\ndone(str(result))",
        )
    record["code"] = code + "\ndone(str(result))"
    record["instruction"] += "\n" + CONTRACT
    return record


def prepare_training(output, pairs, seed):
    if pairs < 1:
        raise ValueError("pairs must be positive")
    output.mkdir(parents=True, exist_ok=False)
    records, rows = [], []
    with closing(Worker()) as worker:
        for family_index, family in enumerate(REQUESTS):
            for pair in range(pairs):
                for variant in (0, 1):
                    record = make_task(
                        family, pair, variant, seed + family_index * 10000019, "train"
                    )
                    session = ReferenceSession(record, worker)
                    session.add(record["inspect"], record["id"] + "-inspect")
                    session.add(record["code"], record["id"] + "-solve")
                    session.check(record["check"])
                    rows.extend(session.rows)
                    assert worker.execute(record["setup"], "reset")["exception"] is None
                    assert (
                        worker.execute(record["negative_control"])["exception"] is None
                    )
                    if worker.execute(record["check"])["exception"] is None:
                        raise AssertionError("Negative control unexpectedly passed")
                    records.append(record)
    write_jsonl(output / "train-tasks.jsonl", records)
    write_jsonl(output / "train.jsonl", rows)
    manifest = {
        "training_only": True,
        "seed": seed,
        "pairs_per_family": pairs,
        "training_records": len(rows),
        "tasks_by_family": dict(Counter(r["family"] for r in records)),
        "training_records_sha256": sha256(output / "train.jsonl"),
        "tasks_sha256": sha256(output / "train-tasks.jsonl"),
        "generator_sha256": sha256(Path(__file__)),
        "feedback": "network-isolated PythonShell; every reference and negative control executed",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args(argv)
    print(json.dumps(prepare_training(args.output, args.pairs, args.seed), indent=2))
