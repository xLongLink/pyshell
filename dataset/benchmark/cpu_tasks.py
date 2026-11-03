"""Freeze independent execution tasks for a CPU fine-tuning experiment."""

import json
import numpy as np
import pandas as pd
import random
import hashlib
import secrets
import argparse
import statistics
from pathlib import Path
from dataclasses import asdict, dataclass
from dataset.benchmark.src.reference import feedback
from dataset.benchmark.src.serialization import serialize

ROOT = Path(__file__).resolve().parent
FAMILIES = (
    "statistics",
    "grouped_revenue",
    "monthly_totals",
    "missing_values",
    "left_join",
    "inspect_callable",
    "recover_execution",
    "stateful_followup",
    "missing_input",
    "linear_system",
    "filtered_means",
    "joined_totals",
)
CONTRACT = (
    "Use the existing Python workspace, not invented input data. "
    "You may execute Python and inspect its output. np, pd, statistics, and json are available. "
    "Save the requested answer in result and call done(str(result)) when finished."
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class TaskRecord:
    id: str
    family: str
    setup: str
    instruction: str
    inspect: str
    code: str
    check: str
    followup: dict | None


@dataclass
class ConverterSpec:
    gain: int
    offset: int
    multiplier: str
    shift: str


@dataclass
class TaskFixture:
    record: TaskRecord
    values: list[int]
    converter: ConverterSpec | None = None
    extra: int | None = None


def task(family, index, seed, split):
    return asdict(build_task(family, index, seed, split).record)


def build_task(family, index, seed, split):
    rng = random.Random(seed + index * 104729)
    prefix = {"train": "source", "dev": "workspace", "test": "input"}[split]
    name = f"{prefix}_{rng.randrange(100000, 999999)}"
    other = name + "_other"
    count = rng.randint(4, 10 if split == "train" else 14)
    values = rng.sample(range(-30, 81) if split == "train" else range(-300, 501), count)
    labels = [rng.choice(["copper", "silver", "gold"]) for _ in values]
    setup = f"{name} = {values!r}"
    inspect = f"print(type({name}).__name__)\nprint({name})"
    followup = None
    converter = None
    extra = None
    if family == "statistics":
        request = f"For the numeric list {name}, return a dict with count, mean, median, and population_std."
        alternate = f"Summarize {name} into result: keys count, mean, median, population_std (population standard deviation)."
        code = f"result = dict(count=len({name}), mean=statistics.mean({name}), median=statistics.median({name}), population_std=statistics.pstdev({name}))"
        expected = dict(
            count=count, mean=statistics.mean(values), median=statistics.median(values), population_std=statistics.pstdev(values)
        )
        check = f"assert set(result) == set({expected!r})\nassert all(abs(result[k] - v) < 1e-7 for k, v in {expected!r}.items())"
    elif family == "grouped_revenue":
        states = [rng.choice(["paid", "pending", "cancelled"]) for _ in values]
        group, amount, status = rng.sample(["region", "sector", "category", "zone", "value", "total", "state", "status"], 3)
        rows = [{group: key, amount: value, status: state} for key, value, state in zip(labels, values, states)]
        setup = f"{name} = pd.DataFrame({rows!r})"
        request = f"In DataFrame {name}, exclude rows whose {status!r} is 'cancelled'. Sum {amount!r} by {group!r} into a plain dict."
        alternate = f"Aggregate {name}: group column {group!r}, sum column {amount!r}; ignore cancelled entries in {status!r}. The result must be a dict."
        code = f"result = {name}.loc[{name}[{status!r}] != 'cancelled'].groupby({group!r})[{amount!r}].sum().to_dict()"
        expected = {}
        for row in rows:
            if row[status] != "cancelled":
                expected[row[group]] = expected.get(row[group], 0) + row[amount]
        check = f"assert result == {expected!r}\nassert len({name}) == {len(rows)}"
    elif family == "monthly_totals":
        rows = [{"date": f"2031-{rng.randint(1, 4):02d}-{rng.randint(1, 28):02d}", "amount": value} for value in values]
        setup = f"{name} = {rows!r}"
        request = f"Group the list of records {name} by the YYYY-MM prefix of date and total amount. Return a dict, including only observed months."
        alternate = f"Calculate monthly amount totals from {name}; use date[:7] as each dict key and omit months with no records."
        code = f"result = {{}}\nfor row in {name}:\n    month = row['date'][:7]\n    result[month] = result.get(month, 0) + row['amount']"
        expected = {}
        for row in rows:
            key = row["date"][:7]
            expected[key] = expected.get(key, 0) + row["amount"]
        check = f"assert result == {expected!r}"
    elif family == "missing_values":
        rows = [{"sensor": key, "reading": value if rng.random() > 0.35 else None} for key, value in zip(labels, values)]
        rows += [{"sensor": "empty", "reading": None}]
        setup = f"{name} = pd.DataFrame({rows!r})"
        request = (
            f"For DataFrame {name}, average reading by sensor, ignoring missing readings. Keep all-missing sensors as None in a plain dict."
        )
        alternate = f"Return per-sensor means of the reading column in {name}. Missing observations are excluded; groups with no numbers must map to None."
        code = f"means = {name}.groupby('sensor')['reading'].mean()\nresult = {{key: None if pd.isna(value) else float(value) for key, value in means.items()}}"
        expected = {}
        for key in {row["sensor"] for row in rows}:
            valid = [row["reading"] for row in rows if row["sensor"] == key and row["reading"] is not None]
            expected[key] = sum(valid) / len(valid) if valid else None
        check = f"expected = {expected!r}\nassert set(result) == set(expected)\nassert all(result[k] is None if v is None else result[k] is not None and abs(result[k] - v) < 1e-7 for k, v in expected.items())"
    elif family == "left_join":
        keys = rng.sample(range(10, 90), 4)
        orders = [{"order_id": i + 101, "customer_id": rng.choice(keys)} for i in range(count)]
        customers = [{"customer_id": key, "name": label} for key, label in zip(keys[:3], ["Nia", "Ivo", "Mara"])]
        setup = f"{name} = pd.DataFrame({orders!r})\n{other} = pd.DataFrame({customers!r})"
        inspect += f"\nprint({other})"
        request = f"Left join orders DataFrame {name} with customers DataFrame {other} on customer_id. Return order_id and customer_name records in original order; unmatched names must be None."
        alternate = f"Enrich {name} with names from {other} using customer_id. Preserve every order and its order. result must be a list of dicts with order_id and customer_name, using None for unknown customers."
        code = f"joined = {name}.merge({other}, on='customer_id', how='left', sort=False)\nresult = [{{'order_id': int(row.order_id), 'customer_name': None if pd.isna(row.name) else row.name}} for row in joined.itertuples()]"
        lookup = {row["customer_id"]: row["name"] for row in customers}
        expected = [{"order_id": row["order_id"], "customer_name": lookup.get(row["customer_id"])} for row in orders]
        check = f"assert result == {expected!r}\nassert len({name}) == {count}"
    elif family == "inspect_callable":
        gain, offset = rng.randint(2, 8), rng.randint(-9, 9)
        multiplier, shift = rng.choice([("scale", "offset"), ("multiplier", "shift"), ("factor", "bias")])
        converter = ConverterSpec(gain, offset, multiplier, shift)
        setup += f'\nclass Converter:\n    def convert(self, values, *, {multiplier}, {shift}):\n        """Return value * {multiplier} + {shift} for each item. Both options are keyword-only."""\n        return [value * {multiplier} + {shift} for value in values]\n{other} = Converter()'
        inspect = f"help({other}.convert)"
        request = f"Inspect {other}.convert before calling it on {name}. Multiply each item by {gain}, then add {offset}. Use the callable's documented keyword arguments."
        alternate = f"Use the existing converter {other}.convert to transform {name} with gain {gain} and additive shift {offset}. Read its help to discover the keyword-only parameter names."
        code = f"result = {other}.convert({name}, {multiplier}={gain}, {shift}={offset})"
        check = f"assert result == {[value * gain + offset for value in values]!r}"
    elif family == "recover_execution":
        setup = f'class Reader:\n    def __init__(self):\n        self.ready = False\n        self.attempts = 0\n    def read(self):\n        """Read numbers; on disconnection call recover(), then retry read()."""\n        self.attempts += 1\n        if not self.ready:\n            raise RuntimeError(\'Disconnected; recover and retry\')\n        return {values!r}\n    def recover(self):\n        self.ready = True\n{name} = Reader()'
        inspect = f"help({name}.read)"
        request = f"Read values from {name}.read() and sum them. If reading fails, call {name}.recover() and retry. Do not replace or recreate the existing reader."
        alternate = f"Obtain a numeric total from the existing reader {name}. A disconnected read() must be followed by recover() and another read(), not by invented data."
        code = f"try:\n    result = sum({name}.read())\nexcept RuntimeError:\n    {name}.recover()\n    result = sum({name}.read())"
        check = f"assert result == {sum(values)}\nassert {name}.ready\nassert {name}.attempts == 2"
    elif family == "stateful_followup":
        extra = rng.randint(2, 30)
        request = f"Square each value in {name} and save the list in result. Keep that same result object available for a follow-up."
        alternate = f"Create result containing the elementwise squares of {name}. A later request will modify the same list in place."
        code = f"result = [value ** 2 for value in {name}]"
        check = f"assert result == {[value**2 for value in values]!r}"
        followup = {
            "instruction": f"Append the square of {extra} to the existing result list exactly once, in place. Do not rebuild it or change the original input.",
            "code": f"result.append({extra} ** 2)",
            "check": f"assert result is saved_result\nassert result == {[value**2 for value in values] + [extra**2]!r}\nassert {name} == {values!r}",
        }
    elif family == "missing_input":
        setup = f"{other} = {values!r}"
        inspect = "print(sorted(key for key in globals() if not key.startswith('_')))"
        request = f"Sum the existing list {name}. If that exact variable is absent, set result = None and status = 'missing_data'. Do not substitute other workspace data."
        alternate = f"Check whether {name} exists before calculating its sum. If unavailable, report result=None and status='missing_data'; another list is not an acceptable replacement."
        code = f"if {name!r} in globals():\n    result = sum({name})\nelse:\n    result = None\n    status = 'missing_data'"
        check = "assert result is None\nassert status == 'missing_data'"
    elif family == "linear_system":
        matrix = [[rng.randint(2, 8), 1], [1, rng.randint(2, 8)]]
        solution = rng.sample(range(-20, 21), 2)
        rhs = [sum(a * b for a, b in zip(row, solution)) for row in matrix]
        setup = f"{name} = np.array({matrix!r}, dtype=float)\n{other} = np.array({rhs!r}, dtype=float)"
        inspect = f"print({name}.shape, {other}.shape)"
        request = f"Solve {name} @ x = {other} using the existing NumPy arrays. Save x as a plain Python list and verify the residual."
        alternate = f"Find the solution vector of the linear system with coefficient matrix {name} and right-hand side {other}. result must be a Python list, and its residual must be checked."
        code = f"result = np.linalg.solve({name}, {other}).tolist()\nassert np.allclose({name} @ np.array(result), {other})"
        check = f"assert isinstance(result, list)\nassert np.allclose(result, {solution!r}, atol=1e-7)"
    elif family == "filtered_means":
        rows = [{"group": key, "value": value, "active": rng.choice([True, False])} for key, value in zip(labels, values)]
        setup = f"{name} = {rows!r}"
        request = f"For the list of dicts {name}, keep active records and compute mean value by group. Return a dict; omit groups with no active records."
        alternate = f"Calculate per-group averages from {name}, counting only records where active is true. Do not include inactive-only groups in the output dict."
        code = f"totals = {{}}\ncounts = {{}}\nfor row in {name}:\n    if row['active']:\n        key = row['group']\n        totals[key] = totals.get(key, 0) + row['value']\n        counts[key] = counts.get(key, 0) + 1\nresult = {{key: totals[key] / counts[key] for key in totals}}"
        expected = {}
        for key in set(labels):
            valid = [row["value"] for row in rows if row["active"] and row["group"] == key]
            if valid:
                expected[key] = sum(valid) / len(valid)
        check = f"assert set(result) == set({expected!r})\nassert all(abs(result[k] - v) < 1e-7 for k, v in {expected!r}.items())"
    else:
        prices = {str(i): rng.randint(2, 30) for i in range(4)}
        rows = [{"item": str(rng.randint(0, 5)), "group": label, "quantity": rng.randint(1, 8)} for label in labels]
        setup = f"{name} = {rows!r}\n{other} = {prices!r}"
        inspect += f"\nprint({other})"
        request = f"Orders {name} are a list of dicts; {other} maps item IDs to unit prices. Total quantity * unit price by group, skipping unknown items. Return a dict."
        alternate = f"Combine records in {name} with the price lookup {other}. For known item keys only, sum quantity times price per group into result."
        code = f"result = {{}}\nfor row in {name}:\n    if row['item'] in {other}:\n        key = row['group']\n        result[key] = result.get(key, 0) + row['quantity'] * {other}[row['item']]"
        expected = {}
        for row in rows:
            if row["item"] in prices:
                expected[row["group"]] = expected.get(row["group"], 0) + row["quantity"] * prices[row["item"]]
        check = f"assert result == {expected!r}"
    instruction = request if split == "train" and index % 3 else alternate
    if split == "test":
        instruction = {
            "statistics": f"I need four descriptive measures of {name}. Put its length under count, arithmetic average under mean, middle-value statistic under median, and population standard deviation under population_std in one dict.",
            "grouped_revenue": f"Inspect frame {name}: identify its numeric measurement column, group-label column, and lifecycle-state column. Sum measurements per group, leaving out rows in the cancelled state. Return a normal dict without mutating the frame.",
            "monthly_totals": f"Turn the dated transactions in {name} into a mapping of calendar months to total amounts. Each date is YYYY-MM-DD; use YYYY-MM keys, and do not fabricate absent months.",
            "missing_values": f"Produce a sensor-to-average-reading mapping from {name}. Null readings contribute nothing to either sum or count. A sensor with no usable readings still needs an entry with value None.",
            "left_join": f"Match every order in {name} to its customer in {other} using customer_id. Keep the original order sequence and unmatched orders. Emit list entries with exactly order_id and customer_name; use Python None when a name is missing.",
            "inspect_callable": alternate.replace("Use the existing converter", "Discover the call signature for").replace(
                "Read its help to discover", "Consult its documentation for"
            ),
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
        setup=setup,
        instruction=instruction + "\n" + CONTRACT,
        inspect=inspect,
        code=code + "\ndone(str(result))",
        check=check,
        followup=followup,
    )
    return TaskFixture(record, values, converter, extra)


def supervision(record):
    namespace = {"np": np, "pd": pd, "statistics": statistics, "json": json}
    exec(record["setup"], namespace)
    turns = [{"role": "user", "content": record["instruction"]}]
    rows = []
    actions = [record["inspect"], record["code"]]
    for index, source in enumerate(actions):
        rows.append({"id": f"{record['id']}-{index}", "family": record["family"], "prompt": serialize(turns), "completion": source})
        result = feedback(source, namespace)
        if result["exception"]:
            raise AssertionError((record["id"], result["exception"]))
        turns += [{"role": "assistant", "content": source}, {"role": "user", "content": json.dumps(result)}]
    exec(record["check"], namespace)
    if record["followup"]:
        namespace["saved_result"] = namespace["result"]
        turns.append({"role": "user", "content": record["followup"]["instruction"]})
        source = record["followup"]["code"] + "\ndone(str(result))"
        rows.append({"id": record["id"] + "-followup", "family": record["family"], "prompt": serialize(turns), "completion": source})
        result = feedback(source, namespace)
        if result["exception"]:
            raise AssertionError((record["id"], result["exception"]))
        exec(record["followup"]["check"], namespace)
    return rows


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    test_seed = secrets.randbits(48)
    splits = {}
    for split, per_family, seed in (("train", 30, 713), ("dev", 5, 9713), ("test", 10, test_seed)):
        records = [
            task(family, index, seed + family_index * 10000019, split)
            for family_index, family in enumerate(FAMILIES)
            for index in range(per_family)
        ]
        path = output / f"{split}-tasks.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        splits[split] = {"tasks": len(records), "sha256": digest(path)}
        if split == "train":
            rows = [row for record in records for row in supervision(record)]
            (output / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        elif split == "dev":
            rows = [row for record in records for row in supervision(record)]
            (output / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "experiment": "cpu-execution-proof-v1",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "test_seed": test_seed,
        "scope": "unseen task instances and wording across 12 supported families; not unseen task families or UI tasks",
        "acceptance": {"test_tasks": 120, "required_passes": 97, "strictly_greater_than": 0.8, "max_calls_per_stage": 4},
        "selection_split": "dev",
        "test_policy": "Do not inspect test predictions or tune after the final test. A failed final test remains a reported failure.",
        "splits": splits,
        "training_records_sha256": digest(output / "train.jsonl"),
        "validation_records_sha256": digest(output / "validation.jsonl"),
        "generator_sha256": digest(Path(__file__)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/cpu-proof-v1")
    prepare(parser.parse_args().output)
