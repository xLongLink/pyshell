import random

from model.benchmark.src.task_families import BUILDERS, FixtureSeed


def build_task(family, index, seed, *, name):
    builder = BUILDERS[family]
    rng = random.Random(seed + index * 104729)
    # Preserve the discarded name draw so existing datasets retain their values.
    rng.randrange(100000, 999999)
    count = rng.randint(4, 14)
    values = rng.sample(range(-300, 501), count)
    labels = [rng.choice(["copper", "silver", "gold"]) for _ in values]
    fixture_seed = FixtureSeed(rng, name, values, labels)
    return builder(fixture_seed)
