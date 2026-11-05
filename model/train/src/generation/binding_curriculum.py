"""Teach live object-role binding and recovery using training-only tasks."""

import argparse
import ast
import json
import re
from contextlib import closing
from pathlib import Path

from model.benchmark.src.corpus import (
    read_jsonl,
    sha256,
    verify_training_data,
)
from model.benchmark.src.execution import Worker
from model.benchmark.src.reference import ReferenceSession
from model.train.src.generation.binding import make_task
from model.train.src.generation.binding_language import (
    diversify_binding,
    diversify_support,
)
from model.train.src.generation.publication import publish_curriculum
from model.train.src.generation.transfer import make_contrast_pair, supervision


def binding_record(family, pair, variant, seed):
    record = make_task(family, pair, variant, seed, "train")
    assignments = [
        node for node in ast.parse(record["setup"]).body if isinstance(node, ast.Assign)
    ]
    name, other = [node.targets[0].id for node in assignments[:2]]
    if family == "joined_totals":
        roles = [
            f"The order-record list is {name}; the item-price lookup is {other}.",
            f"Price lookup {other} provides prices for the order records in {name}.",
            f"Read orders from the list {name}, and resolve item prices through dictionary {other}.",
            f"Keep these roles distinct: {name} holds order rows; {other} maps item identifiers to prices.",
        ]
        aliases = f"orders = {name}\nprices = {other}\n"
        replacements = {name: "orders", other: "prices"}
        errors = [
            f"for row in {other}:\n    print(row['item'])",
            f"print({name}[0]['item'])",
        ]
    else:
        roles = [f"The input numbers are {name}; the existing converter is {other}."]
        aliases = f"values = {name}\nconverter = {other}\n"
        replacements = {name: "values", other: "converter"}
        errors = [
            f"{other}.convert({name}, unsupported_gain=2, unsupported_shift=3)",
            f"{other}.convert({name}, 2, 3)",
        ]
    record["instruction"] = roles[pair % len(roles)] + "\n" + record["instruction"]
    record["inspect"] = (
        f"print(type({name}).__name__)\nprint({name})\nprint(type({other}).__name__)\n"
        + (
            f"help({other}.convert)"
            if family == "inspect_callable"
            else f"print({other})"
        )
    )
    code = record["code"]
    for source, target in replacements.items():
        code = re.sub(r"\b" + re.escape(source) + r"\b", target, code)
    record["code"] = aliases + code
    record["id"] = "roles-" + record["id"]
    record["template_id"] = f"roles-{family}-train-{pair % len(roles)}"
    record["contrast_group"] = "roles-" + record["contrast_group"]
    return record, errors


def trajectories(record, errors, worker):
    rows = []
    session = ReferenceSession(record, worker)
    session.add(record["inspect"], record["id"] + "-inspect")
    session.add(record["code"], record["id"] + "-solve")
    session.check(record["check"])
    rows.extend(session.rows)
    for index, error in enumerate(errors):
        session = ReferenceSession(record, worker)
        session.add(record["inspect"])
        session.add(error, expect_error=True)
        session.add(record["code"], record["id"] + f"-recover-{index}")
        session.check(record["check"])
        rows.extend(session.rows)
    return rows


def prepare(
    output,
    replay,
    evaluation,
    pairs=64,
    support_pairs=32,
    seed=20261109,
    language=False,
):
    if pairs < 1 or support_pairs < 1:
        raise ValueError("Pair budgets must be positive")
    frozen = json.loads((evaluation / "manifest.json").read_text())
    verify_training_data(replay, json.loads((replay / "manifest.json").read_text()))
    rows = read_jsonl(replay / "train.jsonl")
    records = read_jsonl(replay / "train-tasks.jsonl")
    output.mkdir(parents=True, exist_ok=False)
    with closing(Worker()) as worker:
        for family_index, family in enumerate(("inspect_callable", "joined_totals")):
            for pair in range(pairs):
                for variant in (0, 1):
                    record, errors = binding_record(
                        family, pair, variant, seed + family_index * 10000019
                    )
                    if language:
                        record = diversify_binding(record, pair)
                    rows.extend(trajectories(record, errors, worker))
                    records.append(record)
        for family_index, family in enumerate(
            ("left_join", "stateful_followup", "missing_input")
        ):
            for pair in range(support_pairs):
                for record in make_contrast_pair(
                    family,
                    pair,
                    seed + 200000003 + family_index * 10000019,
                    "train",
                    pair % 6,
                ):
                    record["id"] = "roles-support-" + record["id"]
                    record["contrast_group"] = (
                        "roles-support-" + record["contrast_group"]
                    )
                    if language:
                        record = diversify_support(record, pair)
                    rows.extend(
                        supervision(record, include_inspection=True, workspace=worker)
                    )
                    records.append(
                        {
                            key: value
                            for key, value in record.items()
                            if key != "expected"
                        }
                    )
    return publish_curriculum(
        output,
        replay,
        evaluation,
        frozen,
        rows,
        records,
        generator_source=Path(__file__),
        metadata={
            "language_diversity": language,
            "seed": seed,
            "pairs_per_target_family": pairs,
            "support_pairs_per_family": support_pairs,
        },
        details={
            "language_generator_sha256": sha256(
                Path(__file__).with_name("binding_language.py")
            ),
            "feedback": "Real isolated execution; successful direct and two error-recovery trajectories per target task; erroneous scripts are context only, never supervised targets",
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--support-pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--language", action="store_true")
    args = parser.parse_args(argv)
    prepare(
        args.output,
        args.replay,
        args.evaluation,
        args.pairs,
        args.support_pairs,
        args.seed,
        args.language,
    )
