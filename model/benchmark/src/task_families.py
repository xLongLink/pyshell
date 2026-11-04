"""Deterministic fixture construction, without file or experiment ownership."""

import random
import statistics
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
    request: str
    alternate: str
    code: str
    check: str
    followup: dict | None = None
    converter: ConverterSpec | None = None
    extra: int | None = None


def statistics_task(seed):
    name, values = seed.name, seed.values
    request = f"For the numeric list {name}, return a dict with count, mean, median, and population_std."
    alternate = f"Summarize {name} into result: keys count, mean, median, population_std (population standard deviation)."
    code = f"result = dict(count=len({name}), mean=statistics.mean({name}), median=statistics.median({name}), population_std=statistics.pstdev({name}))"
    expected = dict(
        count=len(values), mean=statistics.mean(values), median=statistics.median(values), population_std=statistics.pstdev(values)
    )
    check = f"assert set(result) == set({expected!r})\nassert all(abs(result[k] - v) < 1e-7 for k, v in {expected!r}.items())"
    return FixtureContent(seed.setup, seed.inspection, request, alternate, code, check)


def grouped_revenue(seed):
    name, rng, values = seed.name, seed.rng, seed.values
    states = [rng.choice(["paid", "pending", "cancelled"]) for _ in values]
    group, amount, status = rng.sample(["region", "sector", "category", "zone", "value", "total", "state", "status"], 3)
    rows = [{group: key, amount: value, status: state} for key, value, state in zip(seed.labels, values, states)]
    setup = f"{name} = pd.DataFrame({rows!r})"
    request = f"In DataFrame {name}, exclude rows whose {status!r} is 'cancelled'. Sum {amount!r} by {group!r} into a plain dict."
    alternate = f"Aggregate {name}: group column {group!r}, sum column {amount!r}; ignore cancelled entries in {status!r}. The result must be a dict."
    code = f"result = {name}.loc[{name}[{status!r}] != 'cancelled'].groupby({group!r})[{amount!r}].sum().to_dict()"
    expected = {}
    for row in rows:
        if row[status] != "cancelled":
            expected[row[group]] = expected.get(row[group], 0) + row[amount]
    check = f"assert result == {expected!r}\nassert len({name}) == {len(rows)}"
    return FixtureContent(setup, seed.inspection, request, alternate, code, check)


def monthly_totals(seed):
    name, rng = seed.name, seed.rng
    rows = [{"date": f"2031-{rng.randint(1, 4):02d}-{rng.randint(1, 28):02d}", "amount": value} for value in seed.values]
    request = (
        f"Group the list of records {name} by the YYYY-MM prefix of date and total amount. Return a dict, including only observed months."
    )
    alternate = f"Calculate monthly amount totals from {name}; use date[:7] as each dict key and omit months with no records."
    code = f"result = {{}}\nfor row in {name}:\n    month = row['date'][:7]\n    result[month] = result.get(month, 0) + row['amount']"
    expected = {}
    for row in rows:
        key = row["date"][:7]
        expected[key] = expected.get(key, 0) + row["amount"]
    return FixtureContent(f"{name} = {rows!r}", seed.inspection, request, alternate, code, f"assert result == {expected!r}")


def missing_values(seed):
    name = seed.name
    rows = [{"sensor": key, "reading": value if seed.rng.random() > 0.35 else None} for key, value in zip(seed.labels, seed.values)]
    rows += [{"sensor": "empty", "reading": None}]
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
    return FixtureContent(f"{name} = pd.DataFrame({rows!r})", seed.inspection, request, alternate, code, check)


def left_join(seed):
    name, other, rng = seed.name, seed.other, seed.rng
    keys = rng.sample(range(10, 90), 4)
    orders = [{"order_id": i + 101, "customer_id": rng.choice(keys)} for i in range(len(seed.values))]
    customers = [{"customer_id": key, "name": label} for key, label in zip(keys[:3], ["Nia", "Ivo", "Mara"])]
    setup = f"{name} = pd.DataFrame({orders!r})\n{other} = pd.DataFrame({customers!r})"
    request = f"Left join orders DataFrame {name} with customers DataFrame {other} on customer_id. Return order_id and customer_name records in original order; unmatched names must be None."
    alternate = f"Enrich {name} with names from {other} using customer_id. Preserve every order and its order. result must be a list of dicts with order_id and customer_name, using None for unknown customers."
    code = f"joined = {name}.merge({other}, on='customer_id', how='left', sort=False)\nresult = [{{'order_id': int(row.order_id), 'customer_name': None if pd.isna(row.name) else row.name}} for row in joined.itertuples()]"
    lookup = {row["customer_id"]: row["name"] for row in customers}
    expected = [{"order_id": row["order_id"], "customer_name": lookup.get(row["customer_id"])} for row in orders]
    check = f"assert result == {expected!r}\nassert len({name}) == {len(seed.values)}"
    return FixtureContent(setup, seed.inspection + f"\nprint({other})", request, alternate, code, check)


def inspect_callable(seed):
    name, other, rng, values = seed.name, seed.other, seed.rng, seed.values
    gain, offset = rng.randint(2, 8), rng.randint(-9, 9)
    multiplier, shift = rng.choice([("scale", "offset"), ("multiplier", "shift"), ("factor", "bias")])
    converter = ConverterSpec(gain, offset, multiplier, shift)
    setup = (
        seed.setup
        + f'\nclass Converter:\n    def convert(self, values, *, {multiplier}, {shift}):\n        """Return value * {multiplier} + {shift} for each item. Both options are keyword-only."""\n        return [value * {multiplier} + {shift} for value in values]\n{other} = Converter()'
    )
    request = f"Inspect {other}.convert before calling it on {name}. Multiply each item by {gain}, then add {offset}. Use the callable's documented keyword arguments."
    alternate = f"Use the existing converter {other}.convert to transform {name} with gain {gain} and additive shift {offset}. Read its help to discover the keyword-only parameter names."
    code = f"result = {other}.convert({name}, {multiplier}={gain}, {shift}={offset})"
    check = f"assert result == {[value * gain + offset for value in values]!r}"
    return FixtureContent(setup, f"help({other}.convert)", request, alternate, code, check, converter=converter)


def recover_execution(seed):
    name, values = seed.name, seed.values
    setup = f'class Reader:\n    def __init__(self):\n        self.ready = False\n        self.attempts = 0\n    def read(self):\n        """Read numbers; on disconnection call recover(), then retry read()."""\n        self.attempts += 1\n        if not self.ready:\n            raise RuntimeError(\'Disconnected; recover and retry\')\n        return {values!r}\n    def recover(self):\n        self.ready = True\n{name} = Reader()'
    request = f"Read values from {name}.read() and sum them. If reading fails, call {name}.recover() and retry. Do not replace or recreate the existing reader."
    alternate = f"Obtain a numeric total from the existing reader {name}. A disconnected read() must be followed by recover() and another read(), not by invented data."
    code = f"try:\n    result = sum({name}.read())\nexcept RuntimeError:\n    {name}.recover()\n    result = sum({name}.read())"
    check = f"assert result == {sum(values)}\nassert {name}.ready\nassert {name}.attempts == 2"
    return FixtureContent(setup, f"help({name}.read)", request, alternate, code, check)


def stateful_followup(seed):
    name, values = seed.name, seed.values
    extra = seed.rng.randint(2, 30)
    request = f"Square each value in {name} and save the list in result. Keep that same result object available for a follow-up."
    alternate = f"Create result containing the elementwise squares of {name}. A later request will modify the same list in place."
    code = f"result = [value ** 2 for value in {name}]"
    check = f"assert result == {[value**2 for value in values]!r}"
    followup = {
        "instruction": f"Append the square of {extra} to the existing result list exactly once, in place. Do not rebuild it or change the original input.",
        "code": f"result.append({extra} ** 2)",
        "check": f"assert result is saved_result\nassert result == {[value**2 for value in values] + [extra**2]!r}\nassert {name} == {values!r}",
    }
    return FixtureContent(seed.setup, seed.inspection, request, alternate, code, check, followup=followup, extra=extra)


def missing_input(seed):
    name, other = seed.name, seed.other
    request = f"Sum the existing list {name}. If that exact variable is absent, set result = None and status = 'missing_data'. Do not substitute other workspace data."
    alternate = f"Check whether {name} exists before calculating its sum. If unavailable, report result=None and status='missing_data'; another list is not an acceptable replacement."
    code = f"if {name!r} in globals():\n    result = sum({name})\nelse:\n    result = None\n    status = 'missing_data'"
    return FixtureContent(
        f"{other} = {seed.values!r}",
        "print(sorted(key for key in globals() if not key.startswith('_')))",
        request,
        alternate,
        code,
        "assert result is None\nassert status == 'missing_data'",
    )


def linear_system(seed):
    name, other, rng = seed.name, seed.other, seed.rng
    matrix = [[rng.randint(2, 8), 1], [1, rng.randint(2, 8)]]
    solution = rng.sample(range(-20, 21), 2)
    rhs = [sum(a * b for a, b in zip(row, solution)) for row in matrix]
    setup = f"{name} = np.array({matrix!r}, dtype=float)\n{other} = np.array({rhs!r}, dtype=float)"
    request = f"Solve {name} @ x = {other} using the existing NumPy arrays. Save x as a plain Python list and verify the residual."
    alternate = f"Find the solution vector of the linear system with coefficient matrix {name} and right-hand side {other}. result must be a Python list, and its residual must be checked."
    code = f"result = np.linalg.solve({name}, {other}).tolist()\nassert np.allclose({name} @ np.array(result), {other})"
    check = f"assert isinstance(result, list)\nassert np.allclose(result, {solution!r}, atol=1e-7)"
    return FixtureContent(setup, f"print({name}.shape, {other}.shape)", request, alternate, code, check)


def filtered_means(seed):
    name = seed.name
    rows = [{"group": key, "value": value, "active": seed.rng.choice([True, False])} for key, value in zip(seed.labels, seed.values)]
    request = f"For the list of dicts {name}, keep active records and compute mean value by group. Return a dict; omit groups with no active records."
    alternate = f"Calculate per-group averages from {name}, counting only records where active is true. Do not include inactive-only groups in the output dict."
    code = f"totals = {{}}\ncounts = {{}}\nfor row in {name}:\n    if row['active']:\n        key = row['group']\n        totals[key] = totals.get(key, 0) + row['value']\n        counts[key] = counts.get(key, 0) + 1\nresult = {{key: totals[key] / counts[key] for key in totals}}"
    expected = {}
    for key in set(seed.labels):
        valid = [row["value"] for row in rows if row["active"] and row["group"] == key]
        if valid:
            expected[key] = sum(valid) / len(valid)
    check = f"assert set(result) == set({expected!r})\nassert all(abs(result[k] - v) < 1e-7 for k, v in {expected!r}.items())"
    return FixtureContent(f"{name} = {rows!r}", seed.inspection, request, alternate, code, check)


def joined_totals(seed):
    name, other, rng = seed.name, seed.other, seed.rng
    prices = {str(i): rng.randint(2, 30) for i in range(4)}
    rows = [{"item": str(rng.randint(0, 5)), "group": label, "quantity": rng.randint(1, 8)} for label in seed.labels]
    setup = f"{name} = {rows!r}\n{other} = {prices!r}"
    request = f"Orders {name} are a list of dicts; {other} maps item IDs to unit prices. Total quantity * unit price by group, skipping unknown items. Return a dict."
    alternate = f"Combine records in {name} with the price lookup {other}. For known item keys only, sum quantity times price per group into result."
    code = f"result = {{}}\nfor row in {name}:\n    if row['item'] in {other}:\n        key = row['group']\n        result[key] = result.get(key, 0) + row['quantity'] * {other}[row['item']]"
    expected = {}
    for row in rows:
        if row["item"] in prices:
            expected[row["group"]] = expected.get(row["group"], 0) + row["quantity"] * prices[row["item"]]
    return FixtureContent(setup, seed.inspection + f"\nprint({other})", request, alternate, code, f"assert result == {expected!r}")


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
