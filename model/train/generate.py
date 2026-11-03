"""Generate executable next-script supervision without using benchmark cases."""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import random
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def worker_context():
    tree = ast.parse((REPO / "sample/src/agents/script.py").read_text())
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"INSTRUCTIONS", "STARTUP_LINES"}
    }
    ui_tree = ast.parse((REPO / "src/api/src/api/ui/session.py").read_text())
    ui_doc = next(
        ast.get_docstring(node)
        for node in ui_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UI"
    )
    preloaded = [
        "inspect",
        "json",
        "math",
        "statistics",
        "np",
        "pd",
        "scipy",
        "openpyxl",
        "wb",
        "Figure",
        "Scatter",
        "Area",
        "Bar",
        "Pie",
        "Choropleth",
        "Layout",
        "geography_info",
    ]
    feedback = {
        "stdout": json.dumps({"preloaded": preloaded, "workspace": ["ui", "done"]})
        + "\n"
        + ui_doc
        + "\n",
        "stderr": "",
        "exception": None,
        "done": None,
        "more_input": False,
    }
    return [
        {"role": "system", "content": constants["INSTRUCTIONS"]},
        {
            "role": "user",
            "content": "Initialize the Python terminal, report the preloaded names, and display ui's docstring.",
        },
        {"role": "assistant", "content": "\n".join(constants["STARTUP_LINES"])},
        {"role": "user", "content": json.dumps(feedback)},
    ]


def execute(code, namespace):
    output, error_output = io.StringIO(), io.StringIO()
    completion = []

    def done(message):
        assert isinstance(message, str) and message.strip()
        completion.append(message)

    namespace["done"] = done
    error = None
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
        try:
            exec(compile(code, "<agent>", "exec"), namespace)
        except Exception:
            error = traceback.format_exc()
    return {
        "stdout": output.getvalue(),
        "stderr": error_output.getvalue(),
        "exception": error,
        "done": completion[-1] if completion and error is None else None,
        "more_input": False,
    }


def scenario(index, seed):
    rng = random.Random(seed + index * 7919)
    family = index % 20
    values = rng.sample(range(3, 180), rng.randint(4, 9))
    name = (
        rng.choice(["samples", "observations", "numbers", "entries"])
        + f"_{seed % 10000}_{index}"
    )
    setup = f"{name} = {values!r}"
    previous = []
    previous_task = None
    inspect_only = False

    if family == 0:
        task = f"Find the total of the existing {name} list. Store it in result and call done(). Do not replace the input."
        code = f"result = sum({name})\ndone(f'Total: {{result}}')"
        check = f"assert result == {sum(values)!r}\nassert {name} == {values!r}"
        label = "sum_existing"
    elif family == 1:
        setup += f"\n{name}.extend({values[:2]!r})"
        task = f"Deduplicate {name} and sort ascending into a new list result. Preserve the original list. Call done()."
        code = f"result = sorted(set({name}))\ndone(str(result))"
        check = f"assert result == {sorted(set(values))!r}\nassert len({name}) == {len(values) + 2}"
        label = "sort_deduplicate"
    elif family == 2:
        threshold = rng.randrange(30, 120)
        task = f"From {name}, keep values strictly above {threshold} and square them. Store the ordered list in result and call done()."
        code = f"result = [value ** 2 for value in {name} if value > {threshold}]\ndone(str(result))"
        check = f"assert result == {[v * v for v in values if v > threshold]!r}"
        label = "filter_transform"
    elif family == 3:
        rows = [
            {"kind": rng.choice(["alpha", "beta", "gamma"]), "quantity": value}
            for value in values
        ]
        setup = f"{name} = {rows!r}"
        expected = {}
        for row in rows:
            expected[row["kind"]] = expected.get(row["kind"], 0) + row["quantity"]
        task = f"Aggregate quantity by kind from the existing records {name}. Put a plain dict in result, then done()."
        code = f"result = {{}}\nfor row in {name}:\n    key = row['kind']\n    result[key] = result.get(key, 0) + row['quantity']\ndone(str(result))"
        check = f"assert result == {expected!r}"
        label = "aggregate_records"
    elif family == 4:
        text = ", ".join(map(str, values))
        setup = f"{name} = {text!r}"
        task = f"Parse the comma-separated integers in existing string {name}. Leave a list of integers in result and call done()."
        code = f"result = [int(part.strip()) for part in {name}.split(',')]\ndone(str(result))"
        check = f"assert result == {values!r}"
        label = "parse_text"
    elif family == 5:
        groups = [values[:2], [], values[2:]]
        setup = f"{name} = {groups!r}"
        task = f"Flatten one level of {name} into result, keeping order and ignoring empty sublists. Call done()."
        code = f"result = [value for group in {name} for value in group]\ndone(str(result))"
        check = f"assert result == {values!r}"
        label = "flatten"
    elif family == 6:
        setup = f"{name} = {values[:2] + [None] + values[2:]!r}"
        task = f"Compute the arithmetic mean of non-None entries in {name}. Store result; use None if no valid entries. Call done()."
        code = f"valid = [value for value in {name} if value is not None]\nresult = sum(valid) / len(valid) if valid else None\ndone(str(result))"
        check = f"assert abs(result - {sum(values) / len(values)!r}) < 1e-9"
        label = "missing_entries"
    elif family == 7:
        factor = rng.randint(2, 8)
        task = f"Multiply each value of {name} by {factor}. Create result without modifying {name}. Call done()."
        code = f"result = [value * {factor} for value in {name}]\ndone(str(result))"
        check = f"assert result == {[v * factor for v in values]!r}\nassert {name} == {values!r}"
        label = "preserve_input"
    elif family == 8:
        task = f"Compute running cumulative totals of {name} as a list named result. Call done()."
        code = f"import itertools\nresult = list(itertools.accumulate({name}))\ndone(str(result))"
        check = (
            f"assert result == {[sum(values[: i + 1]) for i in range(len(values))]!r}"
        )
        label = "cumulative"
    elif family == 9:
        task = f"Normalize {name} so the first entry equals 1. Leave result as a new list and call done()."
        code = f"base = {name}[0]\nresult = [value / base for value in {name}]\ndone(str(result))"
        check = f"assert result == {[v / values[0] for v in values]!r}"
        label = "normalization"
    elif family == 10:
        setup += f"\nresult = list({name})\noriginal_list = result"
        extra = rng.randint(200, 300)
        task = f"Append {extra} to the existing result list created from {name}, in place. Do not replace the list. Call done()."
        code = f"result.append({extra})\ndone(str(result))"
        check = f"assert result is original_list\nassert result == {values + [extra]!r}"
        label = "inplace_followup"
    elif family == 11:
        setup = f"{name} = {values + [0]!r}"
        task = f"Compute reciprocals of {name}, representing zero by None. Save result and call done()."
        previous = [f"result = [1 / value for value in {name}]"]
        code = f"result = [1 / value if value != 0 else None for value in {name}]\ndone(str(result))"
        check = f"assert result == {[1 / v for v in values] + [None]!r}"
        label = "recover_zero_division"
    elif family in (12, 13, 14):
        scale, bias = rng.randint(2, 7), rng.randint(-10, 10)
        obj = f"transformer_{rng.randrange(1000)}"
        setup += '\nclass Transformer:\n    def apply(self, values, *, gain, bias):\n        """Return value * gain + bias for each item; gain and bias must be keywords."""\n        return [value * gain + bias for value in values]\n'
        setup += f"{obj} = Transformer()"
        task = f"Inspect {obj}.apply documentation before using it on {name} with gain {scale} and bias {bias}. Store returned values in result and call done()."
        if family == 12:
            code = f"help({obj}.apply)"
            check = "assert 'gain and bias must be keywords' in feedback['stdout']"
            inspect_only = True
            label = "inspect_before_calling"
        else:
            previous = (
                [f"help({obj}.apply)"]
                if family == 13
                else [f"result = {obj}.apply({name}, {scale}, {bias})"]
            )
            code = f"result = {obj}.apply({name}, gain={scale}, bias={bias})\ndone(str(result))"
            check = f"assert result == {[v * scale + bias for v in values]!r}"
            label = (
                "use_inspection_feedback"
                if family == 13
                else "recover_keyword_arguments"
            )
    elif family == 15:
        present = rng.randint(2000, 2020)
        wanted = present + 3
        setup = f"{name} = {{{present}: {values!r}}}"
        task = f"Sum year {wanted} from existing mapping {name}, without external data or substituting years. If absent, set result=None and status='unavailable'. Call done()."
        code = f"if {wanted} not in {name}:\n    result = None\n    status = 'unavailable'\n    done('Requested year {wanted} is unavailable.')\nelse:\n    result = sum({name}[{wanted}])\n    status = 'ok'\n    done(str(result))"
        check = "assert result is None\nassert status == 'unavailable'"
        label = "honest_missing_data"
    elif family == 16:
        dates = [f"2031-0{1 + i % 3}-{1 + i:02d}" for i in range(len(values))]
        rows = list(zip(dates, values))
        setup = f"{name} = {rows!r}"
        expected = {}
        for date, value in rows:
            expected[date[:7]] = expected.get(date[:7], 0) + value
        task = f"Group the existing (ISO date, value) pairs {name} by YYYY-MM, summing values. Store a dict result and call done()."
        code = f"result = {{}}\nfor date, value in {name}:\n    month = date[:7]\n    result[month] = result.get(month, 0) + value\ndone(str(result))"
        check = f"assert result == {expected!r}"
        label = "date_groups"
    elif family == 17:
        words = [rng.choice(["oak", "pine", "ash", "elm"]) for _ in values]
        setup = f"{name} = {words!r}"
        task = f"Count occurrences of each string in {name}. Save a plain dict result and call done()."
        code = f"from collections import Counter\nresult = dict(Counter({name}))\ndone(str(result))"
        check = f"assert result == {dict((w, words.count(w)) for w in set(words))!r}"
        label = "frequency_counts"
    elif family == 18:
        task = f"Find the largest number in existing {name}; leave it in result and call done(). Respond with executable Python only."
        previous = [f"```python\nresult = max({name})\ndone(str(result))\n```"]
        code = f"result = max({name})\ndone(str(result))"
        check = f"assert result == {max(values)!r}"
        label = "recover_markdown_syntax"
    else:
        previous_task = f"Compute the minimum of {name} into smallest and call done()."
        previous = [f"smallest = min({name})\ndone(str(smallest))"]
        task = f"Now compute the range of the same {name} using smallest. Store it in result, preserve smallest, and call done()."
        code = f"result = max({name}) - smallest\ndone(str(result))"
        check = f"assert result == {max(values) - min(values)!r}\nassert smallest == {min(values)!r}"
        label = "new_task_after_done"
    return dict(
        family=label,
        setup=setup,
        instruction=task,
        target=code,
        check=check,
        previous=previous,
        previous_task=previous_task,
        inspect_only=inspect_only,
    )


def build(index, seed, split):
    item = scenario(index, seed)
    namespace = {}
    exec("import inspect, json, math, statistics", namespace)
    exec(item["setup"], namespace)
    messages = worker_context()
    messages.append(
        {"role": "user", "content": item["previous_task"] or item["instruction"]}
    )
    for code in item["previous"]:
        messages.append({"role": "assistant", "content": code})
        feedback = execute(code, namespace)
        text = json.dumps(feedback)
        if item["previous_task"]:
            text += "\n" + item["instruction"]
        messages.append({"role": "user", "content": text})
    feedback = execute(item["target"], namespace)
    assert feedback["exception"] is None, feedback
    assert (feedback["done"] is None) == item["inspect_only"]
    namespace["feedback"] = feedback
    exec(item["check"], namespace)
    messages.append({"role": "assistant", "content": item["target"]})
    return {
        "id": f"{split}-{index:04}",
        "family": item["family"],
        "messages": messages,
        "setup": item["setup"],
        "check": item["check"],
        "inspect_only": item["inspect_only"],
        "previous": item["previous"],
        "target_feedback": feedback,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    for split, count, seed in [("train", 500, 20261102), ("validation", 60, 71921)]:
        items = [build(index, seed, split) for index in range(count)]
        records[split] = items
        path = args.output / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
        )
        print(f"{split}: {len(items)} executable-validated samples -> {path}")
    prompts = [
        {json.dumps(item["messages"][:-1], sort_keys=True) for item in records[split]}
        for split in records
    ]
    assert not prompts[0].intersection(prompts[1])
    assert all(
        len({json.dumps(item["messages"], sort_keys=True) for item in items})
        == len(items)
        for items in records.values()
    )
    manifest = {
        "train": 500,
        "validation": 60,
        "families": 20,
        "source": "deterministic executable templates",
        "benchmark_cases_used": False,
        "worker_sha256": hashlib.sha256(
            (REPO / "sample/src/agents/script.py").read_bytes()
        ).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
