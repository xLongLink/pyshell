import random
from dataclasses import dataclass

from model.benchmark.src.task_types import ConverterSpec


@dataclass
class FixtureSeed:
    rng: random.Random
    name: str
    values: list[int]
    labels: list[str]

    @property
    def other(self):
        return self.name + "_other"

    @property
    def setup(self):
        return f"{self.name} = {self.values!r}"

    @property
    def inspection(self):
        return f"print(type({self.name}).__name__)\nprint({self.name})"


@dataclass
class FixtureContent:
    setup: str
    inspection: str
    converter: ConverterSpec | None = None
    extra: int | None = None


def statistics_task(seed):
    return FixtureContent(seed.setup, seed.inspection)


def grouped_revenue(seed):
    name, rng, values = seed.name, seed.rng, seed.values
    states = [rng.choice(["paid", "pending", "cancelled"]) for _ in values]
    group, amount, status = rng.sample(["region", "sector", "category", "zone", "value", "total", "state", "status"], 3)
    rows = [{group: key, amount: value, status: state} for key, value, state in zip(seed.labels, values, states)]
    setup = f"{name} = pd.DataFrame({rows!r})"
    return FixtureContent(setup, seed.inspection)


def monthly_totals(seed):
    name, rng = seed.name, seed.rng
    rows = [{"date": f"2031-{rng.randint(1, 4):02d}-{rng.randint(1, 28):02d}", "amount": value} for value in seed.values]
    return FixtureContent(f"{name} = {rows!r}", seed.inspection)


def missing_values(seed):
    name = seed.name
    rows = [{"sensor": key, "reading": value if seed.rng.random() > 0.35 else None} for key, value in zip(seed.labels, seed.values)]
    rows += [{"sensor": "empty", "reading": None}]
    return FixtureContent(f"{name} = pd.DataFrame({rows!r})", seed.inspection)


def left_join(seed):
    name, other, rng = seed.name, seed.other, seed.rng
    keys = rng.sample(range(10, 90), 4)
    orders = [{"order_id": i + 101, "customer_id": rng.choice(keys)} for i in range(len(seed.values))]
    customers = [{"customer_id": key, "name": label} for key, label in zip(keys[:3], ["Nia", "Ivo", "Mara"])]
    setup = f"{name} = pd.DataFrame({orders!r})\n{other} = pd.DataFrame({customers!r})"
    return FixtureContent(setup, seed.inspection + f"\nprint({other})")


def inspect_callable(seed):
    other, rng = seed.other, seed.rng
    gain, offset = rng.randint(2, 8), rng.randint(-9, 9)
    multiplier, shift = rng.choice([("scale", "offset"), ("multiplier", "shift"), ("factor", "bias")])
    converter = ConverterSpec(gain, offset, multiplier, shift)
    setup = (
        seed.setup
        + f'\nclass Converter:\n    def convert(self, values, *, {multiplier}, {shift}):\n        """Return value * {multiplier} + {shift} for each item. Both options are keyword-only."""\n        return [value * {multiplier} + {shift} for value in values]\n{other} = Converter()'
    )
    return FixtureContent(setup, f"help({other}.convert)", converter=converter)


def recover_execution(seed):
    name, values = seed.name, seed.values
    setup = f'class Reader:\n    def __init__(self):\n        self.ready = False\n        self.attempts = 0\n    def read(self):\n        """Read numbers; on disconnection call recover(), then retry read()."""\n        self.attempts += 1\n        if not self.ready:\n            raise RuntimeError(\'Disconnected; recover and retry\')\n        return {values!r}\n    def recover(self):\n        self.ready = True\n{name} = Reader()'
    return FixtureContent(setup, f"help({name}.read)")


def stateful_followup(seed):
    extra = seed.rng.randint(2, 30)
    return FixtureContent(seed.setup, seed.inspection, extra=extra)


def missing_input(seed):
    return FixtureContent(
        f"{seed.other} = {seed.values!r}",
        "print(sorted(key for key in globals() if not key.startswith('_')))",
    )


def linear_system(seed):
    name, other, rng = seed.name, seed.other, seed.rng
    matrix = [[rng.randint(2, 8), 1], [1, rng.randint(2, 8)]]
    solution = rng.sample(range(-20, 21), 2)
    rhs = [sum(a * b for a, b in zip(row, solution)) for row in matrix]
    setup = f"{name} = np.array({matrix!r}, dtype=float)\n{other} = np.array({rhs!r}, dtype=float)"
    return FixtureContent(setup, f"print({name}.shape, {other}.shape)")


def filtered_means(seed):
    rows = [{"group": key, "value": value, "active": seed.rng.choice([True, False])} for key, value in zip(seed.labels, seed.values)]
    return FixtureContent(f"{seed.name} = {rows!r}", seed.inspection)


def joined_totals(seed):
    name, other, rng = seed.name, seed.other, seed.rng
    prices = {str(i): rng.randint(2, 30) for i in range(4)}
    rows = [{"item": str(rng.randint(0, 5)), "group": label, "quantity": rng.randint(1, 8)} for label in seed.labels]
    setup = f"{name} = {rows!r}\n{other} = {prices!r}"
    return FixtureContent(setup, seed.inspection + f"\nprint({other})")


BUILDERS = {
    "statistics": statistics_task,
    "grouped_revenue": grouped_revenue,
    "monthly_totals": monthly_totals,
    "missing_values": missing_values,
    "left_join": left_join,
    "inspect_callable": inspect_callable,
    "recover_execution": recover_execution,
    "stateful_followup": stateful_followup,
    "missing_input": missing_input,
    "linear_system": linear_system,
    "filtered_means": filtered_means,
    "joined_totals": joined_totals,
}
