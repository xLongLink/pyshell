import argparse
from pathlib import Path

from model.train.src.generation.publication import replay_curriculum
from model.train.src.generation.transfer import make_contrast_pair, supervision


def join_record(record):
    name, other = record.name, record.other
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

    def generate(worker):
        for pair in range(pairs):
            for record in make_contrast_pair(
                "left_join", pair, seed, "train", pair % 6
            ):
                record = join_record(record)
                yield record, supervision(
                    record, include_inspection=True, workspace=worker
                )

    return replay_curriculum(
        output,
        replay,
        evaluation,
        generate,
        generator_source=Path(__file__),
        metadata={
            "seed": seed,
            "new_join_pairs": pairs,
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
