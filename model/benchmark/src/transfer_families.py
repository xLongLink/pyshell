import random
import statistics

from model.benchmark.src.task_types import ConstructedTask, TaskSeed


def check_number(expected):
    return f"assert isinstance(result, (int, float, np.number))\nassert abs(result - {expected!r}) < 1e-7"


def check_mapping(expected):
    return f"expected = {expected!r}\nassert isinstance(result, dict)\nassert set(result) == set(expected)\nassert all(result[k] is None if v is None else result[k] is not None and abs(result[k] - v) < 1e-7 for k, v in expected.items())"


def check_sequence(expected):
    return f"assert isinstance(result, list)\nassert np.allclose(result, {expected!r}, atol=1e-7)"


def build_statistics(seed, pair, variant, split, slot):
    wording = {}
    name, values = seed.name, seed.values
    if variant:
        wording["stat_request"] = [
            "the median as a single number, averaging the middle pair for even-length inputs",
            "one scalar median, rather than a dictionary",
            "the statistical median, with the usual even-count averaging rule",
            "the central sorted value, or the average of the two central values when needed, as one number",
            "only the median of the observations as a scalar",
            "the 50th-percentile statistic (median), returning a number",
            "the sample median as one numeric answer, handling both odd and even lengths",
            "one value representing the usual median, not the mean",
            "the central order statistic, averaging the central pair if there are two, as a single numeric result",
            "a scalar median of the live observations, without returning any other statistics",
        ][slot]
        expected = statistics.median(values)
        task = ConstructedTask(
            seed.setup, seed.inspection,
            f"result = statistics.median({name})", expected, check_number(expected)
        )
        return task, wording
    wording["stat_request"] = [
        "a dict with keys count, mean, median, population_std; standard deviation is population, not sample",
        "count/mean/median/population_std in one dictionary",
        "the length, average, median and population standard deviation under keys count, mean, median, population_std",
        "a four-entry summary dictionary named by count, mean, median and population_std",
        "the statistics count, mean, median and population_std, all in a plain dict",
        "a mapping containing count, mean, median, population_std; use the standard median for even as well as odd lengths",
        "a dictionary of observation count, arithmetic average, median, and population spread using count, mean, median, population_std as keys",
        "four descriptive statistics, keyed count, mean, median, population_std in a dict",
        "a count/mean/median/population_std mapping describing the actual observations",
        "the number of observations, their arithmetic mean, their median and their population standard deviation as a dictionary with keys count, mean, median, population_std",
    ][slot]
    code = f"result = dict(count=len({name}), mean=statistics.mean({name}), median=statistics.median({name}), population_std=statistics.pstdev({name}))"
    expected = dict(
        count=len(values), mean=statistics.mean(values), median=statistics.median(values), population_std=statistics.pstdev(values)
    )
    task = ConstructedTask(
        seed.setup, seed.inspection, code, expected, check_mapping(expected)
    )
    return task, wording


def build_grouped_revenue(seed, pair, variant, split, slot):
    name, rng, values = seed.name, seed.rng, seed.values
    states = [rng.choice(["paid", "pending", "cancelled"]) for _ in values]
    group, amount, status = rng.sample(["region", "sector", "category", "zone", "value", "total", "state", "status"], 3)
    rows = [{group: key, amount: value, status: state} for key, value, state in zip(seed.labels, values, states)]
    setup = f"{name} = pd.DataFrame({rows!r})"
    wording = {}
    wording["group_roles"] = (
        f"The grouping column is {group!r}, measurement is {amount!r}, lifecycle state is {status!r}."
        if pair % 2
        else "Discover the grouping, numeric measurement, and lifecycle-state columns by inspecting the table."
    )
    wording["group_aggregate"] = "the arithmetic mean measurement" if variant else "the total measurement"
    method = "mean" if variant else "sum"
    if pair % 2:
        prefix = f"group = {group!r}\namount = {amount!r}\nstatus = {status!r}"
    else:
        prefix = f"amount = {name}.select_dtypes(include='number').columns[0]\nstatus = next(col for col in {name} if set({name}[col]).issubset({{'paid', 'pending', 'cancelled'}}))\ngroup = next(col for col in {name} if col not in {{amount, status}})"
    code = prefix + f"\nfiltered = {name}.loc[{name}[status] != 'cancelled']\nresult = filtered.groupby(group)[amount].{method}().to_dict()"
    groups = {}
    for row in rows:
        if row[status] != "cancelled":
            groups.setdefault(row[group], []).append(row[amount])
    expected = {key: sum(values) / len(values) if variant else sum(values) for key, values in groups.items()}
    task = ConstructedTask(
        setup, seed.inspection, code, expected, check_mapping(expected)
    )
    return task, wording


def build_monthly_totals(seed, pair, variant, split, slot):
    name, rng = seed.name, seed.rng
    rows = [{"date": f"2031-{rng.randint(1, 4):02d}-{rng.randint(1, 28):02d}", "amount": value} for value in seed.values]
    setup = f"{name} = {rows!r}"
    wording = {}
    wording["months"] = (
        "Include only months 2031-01 and 2031-02 that occur in the input."
        if variant
        else "Include every observed month, but no unobserved months."
    )
    guard = "\n    if month not in {'2031-01', '2031-02'}:\n        continue" if variant else ""
    code = (
        f"result = {{}}\nfor row in {name}:\n    month = row['date'][:7]{guard}\n    result[month] = result.get(month, 0) + row['amount']"
    )
    expected = {}
    for row in rows:
        key = row["date"][:7]
        if not variant or key in {"2031-01", "2031-02"}:
            expected[key] = expected.get(key, 0) + row["amount"]
    task = ConstructedTask(
        setup, seed.inspection, code, expected, check_mapping(expected)
    )
    return task, wording


def build_missing_values(seed, pair, variant, split, slot):
    name = seed.name
    rows = [{"sensor": key, "reading": value if seed.rng.random() > 0.35 else None} for key, value in zip(seed.labels, seed.values)]
    rows += [{"sensor": "empty", "reading": None}]
    setup = f"{name} = pd.DataFrame({rows!r})"
    wording = {}
    wording["reading_aggregate"] = "the sum" if variant else "the arithmetic mean"
    reduction = "sum(min_count=1)" if variant else "mean()"
    code = f"reduced = {name}.groupby('sensor')['reading'].{reduction}\nresult = {{key: None if pd.isna(value) else float(value) for key, value in reduced.items()}}"
    expected = {}
    for key in dict.fromkeys(row["sensor"] for row in rows):
        values = [float(row["reading"]) for row in rows if row["sensor"] == key and row["reading"] is not None]
        expected[key] = (sum(values) if variant else sum(values) / len(values)) if values else None
    task = ConstructedTask(
        setup, seed.inspection, code, expected, check_mapping(expected)
    )
    return task, wording


def build_left_join(seed, pair, variant, split, slot):
    name, other, rng = seed.name, seed.other, seed.rng
    keys = rng.sample(range(10, 90), 4)
    orders = [{"order_id": i + 101, "customer_id": rng.choice(keys)} for i in range(len(seed.values))]
    customers = [{"customer_id": key, "name": label} for key, label in zip(keys[:3], ["Nia", "Ivo", "Mara"])]
    wording = {}
    orders[-1]["customer_id"] = 999999
    setup = f"{name} = pd.DataFrame({orders!r})\n{other} = pd.DataFrame({customers!r})"
    wording["join_rule"] = (
        [
            "Keep only orders with a matched customer; omit unmatched orders.",
            "Use an inner join: drop orders with no matching lookup entry.",
            "An order without a known customer must not appear in the answer.",
            "Only matched orders qualify for the returned list.",
            "Exclude any order whose customer identifier is unknown to the lookup.",
            "Do not retain orders for which no customer can be found.",
            "The answer must contain matching orders only, with unmatched ones discarded.",
            "Include an order if and only if its customer occurs in the lookup.",
            "Leave out unresolvable orders rather than reporting a missing name.",
            "Unknown customer identifiers disqualify their orders from the output.",
        ]
        if variant
        else [
            "Retain unmatched orders and put None in customer_name for them.",
            "Use left-join behavior, representing missing customer names with Python None.",
            "Every order must appear, even if its customer is unknown; use None for that name.",
            "A missing customer must produce a None name, not removal of the order.",
            "Keep all orders, filling in None when a name cannot be resolved.",
            "Orders without a lookup match stay in the answer with customer_name=None.",
            "No order may be dropped because a customer is missing; its name is None instead.",
            "The result includes unknown customers' orders as well, but with None for their names.",
            "Do not lose orders with unrecognized customer IDs: their customer_name should be None.",
            "Report even unresolvable orders and represent an unavailable name by None.",
        ]
    )[slot]
    how = "inner" if variant else "left"
    code = f"joined = {name}.merge({other}, on='customer_id', how={how!r}, sort=False)\nresult = [{{'order_id': int(row.order_id), 'customer_name': None if pd.isna(row.name) else row.name}} for row in joined.itertuples()]"
    lookup = {row["customer_id"]: row["name"] for row in customers}
    expected = [
        {"order_id": row["order_id"], "customer_name": lookup.get(row["customer_id"])}
        for row in orders
        if not variant or row["customer_id"] in lookup
    ]
    task = ConstructedTask(
        setup, seed.inspection + f"\nprint({other})", code, expected,
        f"assert result == {expected!r}"
    )
    return task, wording


def build_inspect_callable(seed, pair, variant, split, slot):
    name, other, rng = seed.name, seed.other, seed.rng
    gain, offset = rng.randint(2, 8), rng.randint(-9, 9)
    multiplier, shift = rng.choice([("scale", "offset"), ("multiplier", "shift"), ("factor", "bias")])
    setup = (
        seed.setup
        + f'\nclass Converter:\n    def convert(self, values, *, {multiplier}, {shift}):\n        """Return value * {multiplier} + {shift} for each item. Both options are keyword-only."""\n        return [value * {multiplier} + {shift} for value in values]\n{other} = Converter()'
    )
    wording = {}
    gain, offset = gain + variant * 2, offset - variant * 3
    wording.update(gain=gain, shift=offset)
    code = f"result = {other}.convert({name}, {multiplier}={gain}, {shift}={offset})"
    expected = [value * gain + offset for value in seed.values]
    task = ConstructedTask(
        setup, f"help({other}.convert)", code, expected, check_sequence(expected)
    )
    return task, wording


def build_recover_execution(seed, pair, variant, split, slot):
    name, values = seed.name, seed.values
    setup = f'class Reader:\n    def __init__(self):\n        self.ready = False\n        self.attempts = 0\n    def read(self):\n        """Read numbers; on disconnection call recover(), then retry read()."""\n        self.attempts += 1\n        if not self.ready:\n            raise RuntimeError(\'Disconnected; recover and retry\')\n        return {values!r}\n    def recover(self):\n        self.ready = True\n{name} = Reader()'
    wording = {}
    wording["reader_aggregate"] = "the arithmetic mean" if variant else "the numeric sum"
    expression = "statistics.mean(values)" if variant else "sum(values)"
    code = (
        f"try:\n    values = {name}.read()\nexcept RuntimeError:\n    {name}.recover()\n    values = {name}.read()\nresult = {expression}"
    )
    recovery = f"{name}.recover()\nvalues = {name}.read()\nresult = {expression}\ndone(str(result))"
    expected = statistics.mean(values) if variant else sum(values)
    check = check_number(expected) + f"\nassert {name}.ready\nassert {name}.attempts == 2"
    task = ConstructedTask(
        setup, f"help({name}.read)", code, expected, check, recovery
    )
    return task, wording


def build_stateful_followup(seed, pair, variant, split, slot):
    name, values = seed.name, seed.values
    extra = seed.rng.randint(2, 30)
    wording = {}
    power = 2 + variant
    wording["power"] = power
    code = f"result = [value ** {power} for value in {name}]"
    expected = [value**power for value in values]
    followups = {
        "train": "Append {extra} ** {power} to result in place, once. Preserve its identity and the original input.",
        "dev": "Modify the existing result list by appending the {power}-th power of {extra}; do not replace the list or alter the source.",
        "test": "Keep the same result object and extend it with one extra value, {extra} raised to {power}. Leave the input unchanged.",
    }
    followup = {
        "instruction": followups[split].format(extra=extra, power=power),
        "code": f"result.append({extra} ** {power})",
        "check": f"assert result is saved_result\nassert result == {expected + [extra**power]!r}\nassert {name} == {values!r}",
    }
    task = ConstructedTask(
        seed.setup, seed.inspection, code, expected, check_sequence(expected),
        followup=followup, extra=extra
    )
    return task, wording


def build_missing_input(seed, pair, variant, split, slot):
    name, values = seed.name, seed.values
    setup = f"{seed.other} = {values!r}"
    wording = {}
    code = f"if {name!r} in globals():\n    result = sum({name})\nelse:\n    result = None\n    status = 'missing_data'"
    if variant:
        setup += f"\n{name} = {values!r}"
        expected = sum(values)
        check = check_number(expected)
    else:
        expected = None
        check = "assert result is None\nassert status == 'missing_data'"
    task = ConstructedTask(
        setup, "print(sorted(key for key in globals() if not key.startswith('_')))",
        code, expected, check
    )
    return task, wording


def build_linear_system(seed, pair, variant, split, slot):
    name, other, rng = seed.name, seed.other, seed.rng
    matrix = [[rng.randint(2, 8), 1], [1, rng.randint(2, 8)]]
    solution = rng.sample(range(-20, 21), 2)
    rhs = [sum(a * b for a, b in zip(row, solution)) for row in matrix]
    setup = f"{name} = np.array({matrix!r}, dtype=float)\n{other} = np.array({rhs!r}, dtype=float)"
    wording = {}
    matrix = [[float(value) for value in row] for row in matrix]
    rhs = [float(value) for value in rhs]
    a, b = matrix[0]
    c, d = matrix[1]
    determinant = a * d - b * c
    solution = [(d * rhs[0] - b * rhs[1]) / determinant, (a * rhs[1] - c * rhs[0]) / determinant]
    wording["linear_output"] = (
        "Store the sum of the solution components as one number in result."
        if variant
        else "Store the solution as a plain Python list in result."
    )
    code = f"solution = np.linalg.solve({name}, {other})\nassert np.allclose({name} @ solution, {other})\nresult = " + (
        "float(solution.sum())" if variant else "solution.tolist()"
    )
    expected = sum(solution) if variant else solution
    task = ConstructedTask(
        setup, f"print({name}.shape, {other}.shape)", code, expected,
        check_number(expected) if variant else check_sequence(expected)
    )
    return task, wording


def build_filtered_means(seed, pair, variant, split, slot):
    name = seed.name
    rows = [{"group": key, "value": value, "active": seed.rng.choice([True, False])} for key, value in zip(seed.labels, seed.values)]
    setup = f"{name} = {rows!r}"
    wording = {}
    wording["active_aggregate"] = "the sum" if variant else "the arithmetic mean"
    code = (
        f"totals = {{}}\ncounts = {{}}\nfor row in {name}:\n    if row['active']:\n        key = row['group']\n        totals[key] = totals.get(key, 0) + row['value']\n        counts[key] = counts.get(key, 0) + 1\nresult = "
        + ("totals" if variant else "{key: totals[key] / counts[key] for key in totals}")
    )
    grouped = {}
    for row in rows:
        if row["active"]:
            grouped.setdefault(row["group"], []).append(row["value"])
    expected = {key: sum(values) if variant else sum(values) / len(values) for key, values in grouped.items()}
    task = ConstructedTask(
        setup, seed.inspection, code, expected, check_mapping(expected)
    )
    return task, wording


def build_joined_totals(seed, pair, variant, split, slot):
    name, other, rng = seed.name, seed.other, seed.rng
    prices = {str(i): rng.randint(2, 30) for i in range(4)}
    rows = [{"item": str(rng.randint(0, 5)), "group": label, "quantity": rng.randint(1, 8)} for label in seed.labels]
    setup = f"{name} = {rows!r}\n{other} = {prices!r}"
    wording = {}
    wording["spending_output"] = "total quantity, not spending" if variant else "total spending (quantity times unit price)"
    term = "row['quantity']" if variant else f"row['quantity'] * {other}[row['item']]"
    code = f"result = {{}}\nfor row in {name}:\n    if row['item'] in {other}:\n        key = row['group']\n        result[key] = result.get(key, 0) + {term}"
    expected = {}
    for row in rows:
        if row["item"] in prices:
            expected[row["group"]] = expected.get(row["group"], 0) + row["quantity"] * (
                1 if variant else prices[row["item"]]
            )
    task = ConstructedTask(
        setup, seed.inspection + f"\nprint({other})", code, expected, check_mapping(expected)
    )
    return task, wording


BUILDERS = {
    "statistics": build_statistics,
    "grouped_revenue": build_grouped_revenue,
    "monthly_totals": build_monthly_totals,
    "missing_values": build_missing_values,
    "left_join": build_left_join,
    "inspect_callable": build_inspect_callable,
    "recover_execution": build_recover_execution,
    "stateful_followup": build_stateful_followup,
    "missing_input": build_missing_input,
    "linear_system": build_linear_system,
    "filtered_means": build_filtered_means,
    "joined_totals": build_joined_totals,
}


FAMILIES = tuple(BUILDERS)


def build_family(family, pair, variant, seed, split, slot, *, name):
    rng = random.Random(seed + pair * 104729)
    # Preserve the discarded name draw so existing datasets retain their values.
    rng.randrange(100000, 999999)
    count = rng.randint(4, 14)
    values = rng.sample(range(-300, 501), count)
    labels = [rng.choice(["copper", "silver", "gold"]) for _ in values]
    task_seed = TaskSeed(rng, name, values, labels)
    return BUILDERS[family](task_seed, pair, variant, split, slot)
