"""Build a template-held-out, contrastive successor to the first CPU proof."""

import argparse
import hashlib
import json
import random
import secrets
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from model.benchmark.src.corpus import sha256, write_jsonl
from model.benchmark.src.proof_policy import acceptance_metadata
from model.benchmark.src.reference import ReferenceSession
from model.benchmark.src.task_families import FAMILIES
from model.benchmark.src.task_types import CONTRACT
from model.benchmark.src.transfer_families import BUILDERS, BuildContext
from model.train.src.generation.construction import TaskRecord
from model.train.src.generation.proof import build_task
from model.train.src.generation.publication import seal_corpus

ROOT = Path(__file__).resolve().parents[3] / "benchmark"
TEMPLATE_PARTITIONS = {
    "train": range(6),
    "dev": range(6, 8),
    "test": range(8, 10),
}
NAMES = (
    "input",
    "samples",
    "readings",
    "records",
    "payload",
    "observations",
    "workspace",
    "batch",
    "entries",
    "values",
    "items",
    "source",
)
TEMPLATES = {
    "statistics": [
        "For {name}, calculate {stat_request}.",
        "Use list {name} to report {stat_request}.",
        "Summarize the numbers in {name}: I want {stat_request}.",
        "Inspect {name} and return {stat_request}.",
        "The numeric observations are in {name}. Determine {stat_request}.",
        "Please derive {stat_request} from the existing {name} list.",
        "I have measurements named {name}; the answer should contain {stat_request}.",
        "Analyze the list currently stored as {name}, producing {stat_request}.",
        "From the actual entries of {name}, work out {stat_request} without substituting data.",
        "What are {stat_request} for {name}? Use its live values, including any negative ones.",
    ],
    "grouped_revenue": [
        "In frame {name}, {group_roles} Exclude cancelled rows and compute {group_aggregate} per group as a dict.",
        "Return a group-to-number mapping from {name}: discard cancelled entries, then find {group_aggregate}. {group_roles}",
        "Inspect {name}. For each group, report {group_aggregate} after removing the cancelled records. {group_roles}",
        "Using the live table {name}, calculate {group_aggregate} by group, not including cancelled records. {group_roles}",
        "I need {group_aggregate} for every group represented in the non-cancelled rows of {name}. {group_roles}",
        "Filter out cancelled records in {name} before grouping it. The requested statistic is {group_aggregate}. {group_roles}",
        "Make a dictionary from {name} giving {group_aggregate} separately for each group; cancelled rows must contribute nothing. {group_roles}",
        "Find {group_aggregate} within each group of {name}, considering only rows whose lifecycle is not cancelled. {group_roles}",
        "Ignore the cancelled transactions in {name}. Reduce the remaining measurements to {group_aggregate} per group. {group_roles}",
        "What is {group_aggregate} for each category in {name} once cancelled observations are excluded? {group_roles}",
    ],
    "monthly_totals": [
        "Total amount by date[:7] from list {name}. {months}",
        "Group the transactions in {name} by YYYY-MM and sum amount. {months}",
        "Return calendar-month amount totals for records {name}. {months}",
        "Aggregate amount for each month in {name}; dates use YYYY-MM-DD. {months}",
        "Make a YYYY-MM to summed-amount dictionary using {name}. {months}",
        "Add up the amounts in {name} separately for each month. {months}",
        "Convert the dated records {name} into monthly sums, with YYYY-MM keys. {months}",
        "Which amount total belongs to each calendar month in {name}? Return a mapping. {months}",
        "Use the actual transactions from {name} to produce a month-indexed breakdown of total amount. {months}",
        "Combine entries of {name} that fall in the same month and add their amounts together. {months}",
    ],
    "missing_values": [
        "For frame {name}, compute {reading_aggregate} of reading by sensor, excluding null readings. Keep empty sensors as None.",
        "Group {name} by sensor and return {reading_aggregate} reading per sensor; all-null groups must map to None.",
        "Calculate per-sensor {reading_aggregate} from {name}, ignoring missing reading values without dropping empty sensors.",
        "Inspect sensor and reading columns in {name}. Return {reading_aggregate} by sensor; no usable readings means None.",
        "I need a dict of {reading_aggregate} readings for the sensors in {name}. Nulls are not observations; preserve all-null sensors as None.",
        "Reduce reading values in {name} to their {reading_aggregate} for each sensor. Exclude nulls and retain None for empty groups.",
        "Each sensor in {name} needs an entry showing {reading_aggregate} of its non-null readings, or None when none exist.",
        "Produce sensor-level {reading_aggregate} readings from {name}, preserving sensors that have only null observations with value None.",
        "Summarize the valid readings for every sensor of {name} using {reading_aggregate}. An entirely missing group still belongs in the dict as None.",
        "Ignoring gaps in the reading column, find {reading_aggregate} per sensor from {name}; represent a sensor with zero usable measurements as None.",
    ],
    "left_join": [
        "Join orders {name} to customer lookup {other} using customer_id. {join_rule} Return order_id/customer_name dicts in original order.",
        "Attach customer names from {other} to {name} on customer_id. {join_rule} Output records with order_id and customer_name, preserving order sequence.",
        "Match the orders in {name} to names in {other}. {join_rule} Use customer_id, retain order sequence, and return order_id/customer_name records.",
        "Enrich {name} with customer names from {other} via customer_id. {join_rule} Use plain dict entries containing order_id and customer_name.",
        "For each order in {name}, look up its customer in {other}. {join_rule} Keep original order sequence and keys order_id/customer_name.",
        "Use customer_id to associate rows of {name} with customer records in {other}. {join_rule} Return ordered order_id/customer_name dictionaries.",
        "Build an ordered list of order_id and customer_name pairs as dictionaries from {name} and {other}, matching on customer_id. {join_rule}",
        "Resolve the customers for orders {name} using table {other}; customer_id is the link. {join_rule} Preserve order sequence and emit order_id/customer_name records.",
        "Give the customer name for the orders in {name} using the identifiers listed in {other}. {join_rule} The list must retain order sequence and use order_id/customer_name keys.",
        "Associate each qualifying order from {name} with its name in {other} by customer_id. {join_rule} Report ordered dictionaries with order_id and customer_name.",
    ],
    "inspect_callable": [
        "Inspect {other}.convert, then apply gain {gain} and shift {shift} to {name} using the documented keyword arguments.",
        "Use {other}.convert on {name}: multiply by {gain} and add {shift}. Read its help for the keyword-only parameter names.",
        "Call the converter {other}.convert on {name} with multiplier {gain} and additive offset {shift}, after inspecting its signature.",
        "Consult the documentation of {other}.convert and transform {name} with scale {gain}, then translation {shift}.",
        "Each value of {name} should be multiplied by {gain} and shifted by {shift}. Use {other}.convert and discover its keyword-only names.",
        "Read help for {other}.convert before using it to map {name} to value * {gain} + {shift}.",
        "Through the existing {other}.convert callable, obtain the affine transformation of {name} with gain {gain} and displacement {shift}; inspect its arguments first.",
        "I want {name} transformed by factor {gain}, followed by adding {shift}. Discover how {other}.convert accepts those options and invoke it.",
        "Determine the accepted keywords of {other}.convert and use that method, not a replacement, for {name} with gain {gain} and bias {shift}.",
        "Using the documented interface of {other}.convert, calculate value * {gain} + {shift} for the existing {name} values.",
    ],
    "recover_execution": [
        "Read {name}.read() and return {reader_aggregate}. If disconnected, call recover() on that same object and retry read().",
        "Obtain {reader_aggregate} from the numbers returned by {name}.read(). A RuntimeError requires {name}.recover(), then a fresh read().",
        "Inspect reader {name} and compute {reader_aggregate}. On disconnection, recover the original object and retry its read() method.",
        "Use the existing {name} reader to get {reader_aggregate}. If read() fails, recover() restores the connection but does not return the data; read again.",
        "Return {reader_aggregate} of the data from {name}.read(), handling a disconnected reader by calling recover() and repeating read().",
        "Get the numeric contents of {name} with read(), then calculate {reader_aggregate}. Repair a failed connection with recover() before retrying read().",
        "The existing object {name} supplies readings through read(); find {reader_aggregate}. Recover from a disconnected read and fetch the readings again instead of using recover() as data.",
        "Calculate {reader_aggregate} from the live reader {name}. A failed read must lead to recover() on the same instance, followed by another read().",
        "Fetch the numbers held by {name} and report {reader_aggregate}. If read() reports a disconnection, repair it with recover() and obtain the numbers by retrying read().",
        "Use {name} without replacing it: read its data and determine {reader_aggregate}. A connection failure needs recover() before the next read; recovery itself is not the payload.",
    ],
    "stateful_followup": [
        "Raise each item of {name} to power {power}; save a list in result for an in-place follow-up.",
        "Create result as the elementwise power-{power} values of {name}. Keep the list object for the next request.",
        "Map values in {name} to value ** {power} and leave the result list available for later modification.",
        "Store a list containing each {name} entry raised to exponent {power}. A subsequent request will mutate this same list.",
        "Compute power {power} for every number in {name}, returning a persistent result list.",
        "Build result from {name} by exponentiating each number to {power}; preserve that output object for a follow-up.",
        "Produce the elementwise {power}-th powers of {name} in a list called result, which must remain available afterward.",
        "Transform {name} into its power-{power} sequence and save that list for reuse by a later instruction.",
        "For every value currently in {name}, calculate value raised to {power}. Keep the resulting list in result so it can be changed in place next.",
        "Leave result holding the {power}-th powers of the values in {name}, as one persistent Python list.",
    ],
    "missing_input": [
        "Sum exact variable {name} if it exists; otherwise use result=None and status='missing_data'. Never substitute another object.",
        "Find {name} in the workspace. Return its sum if present; if absent, result must be None and status 'missing_data'.",
        "Only {name} is the requested input. Sum it when available, or set result=None and status='missing_data' without inventing data.",
        "Check for an existing list named {name}. An available list should be summed; a missing one requires None and status='missing_data'.",
        "Use {name}, not a similarly named list, to calculate a sum. If it is unavailable, leave result as None and status as 'missing_data'.",
        "Calculate the total of {name} only if that exact name is defined. If undefined, report result=None and status='missing_data'.",
        "Determine whether the requested {name} input is actually present before adding it up; missing input means result=None and status='missing_data'.",
        "The computation needs {name} specifically. Return its total when it exists, otherwise None with status='missing_data'; do not use a replacement.",
        "Do not assume {name} exists: verify its presence, then sum it if available. If not, return None in result and 'missing_data' in status.",
        "Work only with the exact requested object {name}. Its absence must produce result=None and status='missing_data'; its presence calls for summing its entries.",
    ],
    "linear_system": [
        "Solve {name} @ x = {other} and verify the residual. {linear_output}",
        "Use the existing coefficient matrix {name} and RHS {other} to solve for x; check A @ x equals b. {linear_output}",
        "Find x satisfying the linear system with matrix {name} and vector {other}, then verify it by multiplication. {linear_output}",
        "Compute and check the solution of {name} @ x = {other}. {linear_output}",
        "Determine the solution vector for matrix {name}, right-hand side {other}, and confirm its residual is zero. {linear_output}",
        "Use NumPy to solve the system defined by {name} and {other}; check the proposed solution. {linear_output}",
        "Find a vector whose product with {name} reproduces {other}, and validate that product. {linear_output}",
        "Resolve the equations represented by {name} and {other}, checking the residual before finishing. {linear_output}",
        "What x makes matrix {name} multiplied by x equal {other}? Confirm the equality numerically. {linear_output}",
        "Using the actual arrays, determine x from {name} @ x = {other} and verify that it reconstructs the RHS. {linear_output}",
    ],
    "filtered_means": [
        "Keep active records from list {name}; report {active_aggregate} of value by group as a dict, omitting inactive-only groups.",
        "From {name}, group active rows and compute {active_aggregate} of value. Ignore inactive rows and omit empty groups.",
        "Calculate {active_aggregate} per group in {name}, using value only when active is true. Return a plain dict.",
        "Inspect records {name}. For each group with active observations, find {active_aggregate} of their value fields.",
        "Give a group-to-statistic mapping for active entries of {name}; the requested statistic is {active_aggregate} of value.",
        "Filter {name} to active=True and compute grouped {active_aggregate} of value, without entries for groups lacking active data.",
        "Using only the active observations in {name}, determine {active_aggregate} of value for every represented group.",
        "Summarize value separately for groups in {name} with {active_aggregate}, excluding inactive records entirely.",
        "The active flag selects usable records in {name}. Reduce their value fields to {active_aggregate} for each group that remains.",
        "For records of {name} marked active, what is {active_aggregate} of value per group? Return no groups that have no active records.",
    ],
    "joined_totals": [
        "Use orders {name} and price lookup {other} to calculate {spending_output} by group. Exclude items missing from the lookup.",
        "Match item keys from {name} to {other}, skip unknown items, and return grouped {spending_output} as a dict.",
        "For known items only, combine {name} with {other} and compute {spending_output} per group.",
        "Inspect the order list {name} and price dictionary {other}. Return {spending_output} by group, ignoring unpriced items.",
        "Aggregate {spending_output} for each group in {name}, considering only orders whose item is a key in {other}.",
        "Look up item prices for {name} in {other}. Discard unknown keys; report {spending_output} separately by group.",
        "Restrict {name} to items represented in {other}, then give each group's {spending_output}.",
        "Produce a group-indexed dictionary of {spending_output} from {name}; records without an item price in {other} must not count.",
        "Which {spending_output} belongs to each group of orders {name} after items absent from price map {other} are removed?",
        "Use {other} to identify priced items among {name}, and combine those orders into {spending_output} for each group.",
    ],
}


def make_task(family, pair, variant, seed, split, slot) -> TaskRecord:
    rng = random.Random(seed + pair * 65537)
    name = f"{rng.choice(NAMES)}_{rng.randrange(10000, 999999)}"
    fixture = build_task(family, pair, seed, name=name)
    record = fixture.record
    ns = {"np": np, "pd": pd, "statistics": statistics, "json": json}
    exec(record["setup"], ns)
    build = BuildContext(
        fixture=fixture,
        namespace=ns,
        name=name,
        pair=pair,
        variant=variant,
        slot=slot,
        split=split,
        wording={},
    )
    solution = BUILDERS[family](build)
    instruction = TEMPLATES[family][slot].format(
        name=build.name, other=build.other, **build.wording
    )
    task = TaskRecord(
        {
            "id": f"{split}-{family}-{pair:04d}-v{variant}",
            "family": family,
            "setup": record["setup"],
            "instruction": instruction + "\n" + CONTRACT,
            "inspect": record["inspect"],
            "code": solution.code + "\ndone(str(result))",
            "check": solution.check,
            "followup": record["followup"],
            "template_id": f"{family}-t{slot}",
            "contrast_group": f"{split}-{family}-{pair:04d}",
            "variant": variant,
            "recovery_code": solution.recovery_code,
            "expected": solution.expected,
        },
        name=name,
    )
    task.extra = fixture.extra
    return task


def make_contrast_pair(family, pair, seed, split, slot):
    for attempt in range(32):
        variants = [
            make_task(family, pair, variant, seed + attempt * 1000000007, split, slot)
            for variant in (0, 1)
        ]
        if variants[0]["expected"] != variants[1]["expected"]:
            return variants
    raise ValueError(f"Could not construct a nontrivial contrast for {family}")


def supervision(
    record,
    include_inspection,
    *,
    workspace=None,
    retry=None,
    id_prefix=None,
    capture_rows=True,
):
    session = ReferenceSession(record, workspace, capture_rows=capture_rows)
    if retry is None:
        retry = bool(record["recovery_code"] and record["variant"])
    id_prefix = record["id"] if id_prefix is None else id_prefix

    def add(source, supervise=True, expect_error=False):
        record_id = id_prefix + f"-s{len(session.turns)}" if supervise else None
        session.add(source, record_id, expect_error=expect_error)

    add(record["inspect"], include_inspection)
    if retry:
        add(f"values = {record.name}.read()", expect_error=True)
        add(record["recovery_code"])
    else:
        add(record["code"])
    session.check(record["check"])
    if record["followup"]:
        session.followup(record["followup"], id_prefix + f"-s{len(session.turns) + 1}")
    return session.rows


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    test_seed = secrets.randbits(48)
    splits = {}
    setup_sets = {}
    train_rows = []
    validation_rows = []
    for split, seed, count in (
        ("train", 731971, 12),
        ("dev", 813797, 5),
        ("test", test_seed, 10),
    ):
        slots = TEMPLATE_PARTITIONS[split]
        records = []
        contrast_pairs = 0
        same_workspace_pairs = 0
        for family_index, family in enumerate(FAMILIES):
            for pair in range((count + 1) // 2):
                slot = slots[pair % len(slots)]
                variants = make_contrast_pair(
                    family, pair, seed + family_index * 10000019, split, slot
                )
                if pair * 2 + 1 < count:
                    contrast_pairs += 1
                    same_workspace_pairs += variants[0]["setup"] == variants[1]["setup"]
                for variant, record in enumerate(variants):
                    if pair * 2 + variant >= count:
                        continue
                    rows = supervision(
                        record,
                        include_inspection=split != "train" or pair == 0,
                        capture_rows=split != "test",
                    )
                    if split == "train":
                        train_rows.extend(rows)
                    elif split == "dev":
                        validation_rows.extend(rows)
                    record.pop("expected")
                    records.append(record)
        path = output / f"{split}-tasks.jsonl"
        write_jsonl(path, records)
        splits[split] = {
            "tasks": len(records),
            "sha256": sha256(path),
            "template_ids": sorted({record["template_id"] for record in records}),
            "families": dict(Counter(record["family"] for record in records)),
            "contrast_pairs": contrast_pairs,
            "same_workspace_contrasts": same_workspace_pairs,
        }
        setup_sets[split] = {record["setup"] for record in records}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert not setup_sets[left] & setup_sets[right]
        assert not set(splits[left]["template_ids"]) & set(
            splits[right]["template_ids"]
        )
    write_jsonl(output / "train.jsonl", train_rows)
    write_jsonl(output / "validation.jsonl", validation_rows)
    manifest = {
        "experiment": "cpu-template-transfer-v1",
        "scope": "unseen instruction templates and task instances across 12 supported families, including paired contrasting operations; not unseen families or UI tasks",
        "acceptance": acceptance_metadata(),
        "test_seed": test_seed,
        "splits": splits,
        "training_records": len(train_rows),
        "contrasting_pairs": sum(
            metadata["contrast_pairs"] for metadata in splits.values()
        ),
        "selection_split": "dev",
        "old_test_reused": False,
        "training_records_sha256": sha256(output / "train.jsonl"),
        "validation_records_sha256": sha256(output / "validation.jsonl"),
        "generator_sha256": sha256(Path(__file__)),
        "instruction_template_sha256": hashlib.sha256(
            json.dumps(TEMPLATES, sort_keys=True).encode()
        ).hexdigest(),
    }
    (output / "generator-frozen.py").write_bytes(Path(__file__).read_bytes())
    seal_corpus(output, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/cpu-transfer-v1"
    )
    prepare(parser.parse_args(argv).output)
