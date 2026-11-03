"""Select deterministic, family-balanced final-script audit examples."""

import random
from collections import defaultdict


def balanced_sample(records, count, seed=713):
    groups = defaultdict(list)
    for record in records:
        if not record["inspect_only"]:
            groups[record["family"]].append(record)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    keys = sorted(groups)
    rng.shuffle(keys)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            raise ValueError(f"Only {len(selected)} eligible examples available")
    return selected
