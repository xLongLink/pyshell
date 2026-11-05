"""Consolidate join selection while replaying learned binding skills."""

import argparse
import json
import re
from contextlib import closing
from pathlib import Path

from model.benchmark.src.corpus import (
    read_jsonl,
    verify_training_data,
)
from model.benchmark.src.execution import Worker
from model.train.src.generation.publication import publish_curriculum
from model.train.src.generation.transfer import make_contrast_pair, supervision


def join_record(record):
    name = re.search(r"\b[a-z]+_[0-9]+\b", record["instruction"])[0]
    other = name + "_other"
    selection = (
        "orders.loc[orders['customer_id'].isin(customers['customer_id'])]"
        if record["variant"]
        else "orders"
    )
    record["code"] = (
        f"orders = {name}\ncustomers = {other}\nselected = {selection}\n"
        "joined = selected.merge(customers, on='customer_id', how='left', sort=False)\n"
        "result = [{'order_id': int(row.order_id), 'customer_name': None if pd.isna(row.name) else row.name} for row in joined.itertuples()]\n"
        "done(str(result))"
    )
    record["id"] = "join-consolidation-" + record["id"]
    record["contrast_group"] = "join-consolidation-" + record["contrast_group"]
    return record


def prepare(output, replay, evaluation, pairs=256, seed=20261119):
    if pairs < 1:
        raise ValueError("pairs must be positive")
    verify_training_data(replay, json.loads((replay / "manifest.json").read_text()))
    frozen = json.loads((evaluation / "manifest.json").read_text())
    rows = read_jsonl(replay / "train.jsonl")
    records = read_jsonl(replay / "train-tasks.jsonl")
    output.mkdir(parents=True, exist_ok=False)
    with closing(Worker()) as worker:
        for pair in range(pairs):
            for record in make_contrast_pair(
                "left_join", pair, seed, "train", pair % 6
            ):
                record = join_record(record)
                rows.extend(
                    supervision(record, include_inspection=True, workspace=worker)
                )
                records.append(
                    {key: value for key, value in record.items() if key != "expected"}
                )
    return publish_curriculum(
        output,
        replay,
        evaluation,
        frozen,
        rows,
        records,
        generator_source=Path(__file__),
        metadata={"seed": seed, "new_join_pairs": pairs},
        details={
            "intervention": "Keep-all versus matched-only selection is an explicit filter before a common merge; independent original list-of-records oracle verifies every result",
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare(args.output, args.replay, args.evaluation)
