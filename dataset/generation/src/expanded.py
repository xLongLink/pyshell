"""Build execution-verified raw worker supervision without reading benchmark cases."""

import argparse
import io
import json
import math
import random
import statistics
import traceback
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.benchmark.src.corpus import sha256, write_jsonl
from dataset.benchmark.src.serialization import serialize
from dataset.generation.src.legacy import worker_context

REPO = Path(__file__).resolve().parents[3]


def execute(code, namespace):
    stdout, stderr, completions = io.StringIO(), io.StringIO(), []

    def done(message):
        if not isinstance(message, str) or not message.strip():
            raise ValueError("done requires a nonempty message")
        completions.append(message)

    namespace["done"] = done
    error = None
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            exec(compile(code, "<agent>", "exec"), namespace)
        except Exception:
            error = traceback.format_exc()
            if len(error) > 1800:
                error = "Traceback (earlier frames omitted):\n" + error[-1800:]
    return {
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "exception": error,
        "done": completions[-1] if completions and error is None else None,
        "more_input": False,
    }


def scenario(index, family, seed):
    rng = random.Random(seed + index * 31337)
    values = rng.sample(range(2, 95), rng.randint(4, 8))
    suffix = f"{seed % 1000}_{index}"
    name = (
        rng.choice(["records", "readings", "observations", "inventory", "dataset"])
        + "_"
        + suffix
    )
    obj = "service_" + suffix
    column = rng.choice(["units", "quantity", "volume", "score"])
    group = rng.choice(["department", "category", "zone", "segment"])
    group_values = [rng.choice(["amber", "blue", "green"]) for _ in values]
    setup = f"{name} = {values!r}"
    history = []
    inspect_code = f"print(type({name}).__name__, {name}[:3])"
    body = f"result = sum({name})"
    check = f"assert result == {sum(values)}"
    task = ""
    label = ""

    if family == 0:
        label = "discover_workspace_list"
        task = f"Find the existing list whose name starts with {name!r}; inspect workspace names before summing it into result."
        inspect_code = "print(sorted(key for key, value in list(globals().items()) if not key.startswith('_') and isinstance(value, list)))"
    elif family == 1:
        label = "inspect_frame_columns"
        setup = f"{name} = pd.DataFrame({{{column!r}: {values!r}, 'description': {['item'] * len(values)!r}}})"
        task = f"Inspect {name}'s type, columns, and first rows. Sum the numeric {column!r} column into result."
        inspect_code = f"print(type({name}).__name__); print({name}.columns.tolist()); print({name}.head(3).to_dict('records'))"
        body = f"result = float({name}[{column!r}].sum())"
    elif family == 2:
        label = "inspect_filtered_frame_aggregation"
        flags = [i % 3 != 0 for i in range(len(values))]
        setup = f"{name} = pd.DataFrame({{{group!r}: {group_values!r}, {column!r}: {values!r}, 'active': {flags!r}}})"
        task = f"Inspect the existing DataFrame {name}. Sum {column!r} by {group!r}, using only active rows; result must be a plain dict."
        inspect_code = (
            f"print({name}.columns.tolist()); print({name}.head(3).to_dict('records'))"
        )
        body = f"result = {name}.loc[{name}['active']].groupby({group!r})[{column!r}].sum().to_dict()"
        expected = {}
        for key, value, active in zip(group_values, values, flags):
            if active:
                expected[key] = expected.get(key, 0) + value
        check = f"assert result == {expected!r}"
    elif family in (3, 4, 29):
        label = {
            3: "callable_documentation",
            4: "repair_keyword_only_call",
            29: "unfamiliar_parameter_names",
        }[family]
        gain, shift = rng.randint(2, 6), rng.randint(-6, 6)
        setup += '\nclass Rescaler:\n    def transform(self, values, *, multiplier, shift):\n        """Return value * multiplier + shift for each item; both options are keyword-only."""\n        return [value * multiplier + shift for value in values]\n'
        setup += f"{obj} = Rescaler()"
        task = f"Inspect {obj}.transform before applying multiplier {gain} and shift {shift} to existing {name}; save its output as result."
        inspect_code = f"help({obj}.transform)"
        if family == 4:
            history = [f"result = {obj}.transform({name}, {gain}, {shift})"]
        body = f"result = {obj}.transform({name}, multiplier={gain}, shift={shift})"
        check = f"assert result == {[v * gain + shift for v in values]!r}"
    elif family == 5:
        label = "recover_undefined_variable"
        task = f"Compute the sum of the existing {name} list into result. If a variable is missing, inspect the workspace rather than inventing values."
        history = ["result = sum(missing_values_for_this_task)"]
        inspect_code = (
            "print(sorted(key for key in list(globals()) if not key.startswith('_')))"
        )
    elif family in (6, 7):
        label = "recover_frame_attribute" if family == 6 else "recover_frame_key"
        setup = f"{name} = pd.DataFrame({{{column!r}: {values!r}, 'label': {group_values!r}}})"
        task = f"Use existing DataFrame {name} to sum {column!r} into result. Discover its schema after any access error."
        history = (
            [f"print({name}.not_a_real_attribute)"]
            if family == 6
            else [f"result = {name}['missing_metric'].sum()"]
        )
        inspect_code = (
            f"print({name}.columns.tolist()); print({name}.head(2).to_dict('records'))"
        )
        body = f"result = float({name}[{column!r}].sum())"
    elif family == 8:
        label = "distinguish_records_from_dataframe"
        rows = [{group: key, column: value} for key, value in zip(group_values, values)]
        setup = f"{name} = {rows!r}"
        task = f"Inspect {name}'s type. Sum {column!r} by {group!r} into a plain dict result without assuming pandas methods exist."
        history = [f"result = {name}.groupby({group!r})[{column!r}].sum()"]
        body = f"result = {{}}\nfor row in {name}:\n    key = row[{group!r}]\n    result[key] = result.get(key, 0) + row[{column!r}]"
        expected = {}
        for key, value in zip(group_values, values):
            expected[key] = expected.get(key, 0) + value
        check = f"assert result == {expected!r}"
    elif family == 9:
        label = "recover_date_api"
        day = rng.randint(1, 26)
        setup = f"import datetime\n{name} = datetime.date(2034, 7, {day})"
        task = f"Format the existing date object {name} as YYYY-MM-DD in result. Inspect its type rather than assuming another library's API."
        history = [f"result = {name}.to('string')"]
        inspect_code = (
            f"print(type({name}).__name__); print(hasattr({name}, 'isoformat'))"
        )
        body = f"result = {name}.isoformat()"
        check = f"assert result == '2034-07-{day:02d}'"
    elif family == 10:
        label = "empty_input_limitation"
        setup = f"{name} = []"
        task = f"Inspect existing {name} and compute its mean into result, using None when empty. Do not fabricate measurements."
        body = f"result = sum({name}) / len({name}) if {name} else None"
        check = "assert result is None"
    elif family == 11:
        label = "inspect_mapping_key_types"
        key = str(rng.randint(2040, 2060))
        setup = f"{name} = {{{key!r}: {values!r}}}"
        task = f"Find the requested key {key!r} in {name}, inspect its key types if needed, and sum its list into result. Do not substitute another key."
        history = [f"result = sum({name}[{int(key)}])"]
        inspect_code = f"print([(key, type(key).__name__) for key in {name}])"
        body = f"result = sum({name}[{key!r}])"
    elif family == 12:
        label = "recover_then_retry_reader"
        setup = (
            'class Source:\n    def __init__(self):\n        self.ready = False\n        self.attempts = 0\n    def fetch(self):\n        """Return numeric records. On failure, call reconnect(), then fetch again."""\n        self.attempts += 1\n        if not self.ready:\n            raise RuntimeError(\'Disconnected: call reconnect(), then retry fetch().\')\n        return '
            + repr(values)
            + '\n    def reconnect(self):\n        """Restore connection; this returns None, not the data."""\n        self.ready = True\n'
        )
        setup += f"{obj} = Source()"
        task = f"Sum records returned by {obj}.fetch() into result. Recover connection failures and actually fetch the data after reconnecting."
        history = [f"result = sum({obj}.fetch())"]
        inspect_code = f"help({obj}.reconnect)"
        body = f"{obj}.reconnect()\nresult = sum({obj}.fetch())"
        check = f"assert result == {sum(values)}\nassert {obj}.ready and {obj}.attempts >= 2"
    elif family == 13:
        label = "missing_dependency_fallback"
        task = f"Calculate the median of existing {name} into result. If an optional library is missing, use the standard library instead."
        history = ["import nonexistent_optional_statistics_package"]
        inspect_code = "import statistics\nprint(inspect.signature(statistics.median))"
        body = f"import statistics\nresult = statistics.median({name})"
        check = f"assert result == {statistics.median(values)!r}"
    elif family == 14:
        label = "inspect_matrix_shape"
        x, y = rng.randint(1, 9), rng.randint(1, 9)
        setup = f"{name} = np.array([[4.0, 1.0], [2.0, 3.0]])\nright_{suffix} = np.array([{4 * x + y}, {2 * x + 3 * y}])"
        task = f"Inspect shapes, then solve {name} @ x = right_{suffix}. Leave result as a plain Python list and verify the residual."
        inspect_code = f"print({name}.shape, right_{suffix}.shape)"
        body = f"result = np.linalg.solve({name}, right_{suffix}).tolist()\nassert np.allclose({name} @ np.array(result), right_{suffix})"
        check = (
            f"assert np.allclose(result, [{x}, {y}])\nassert isinstance(result, list)"
        )
    elif family == 15:
        label = "recover_numeric_strings"
        text_values = [str(value) for value in values]
        setup = f"{name} = {text_values!r}"
        task = f"Inspect the entries in {name}; sum them numerically into result rather than concatenating strings."
        history = [f"result = sum({name})"]
        body = f"result = sum(float(value) for value in {name})"
    elif family == 16:
        label = "record_field_discovery"
        rows = [{column: value, "tag": key} for key, value in zip(group_values, values)]
        setup = f"{name} = {rows!r}"
        task = f"Inspect the keys in records {name}, then extract {column!r} values into result in original order."
        inspect_code = f"print({name}[0].keys()); print({name}[:2])"
        body = f"result = [row[{column!r}] for row in {name}]"
        check = f"assert result == {values!r}"
    elif family == 17:
        label = "weighted_average"
        weights = [rng.randint(0, 4) for _ in values]
        setup += f"\nweights_{suffix} = {weights!r}"
        task = f"Compute the weighted mean of {name} using weights_{suffix}. Inspect lengths first; result is None if the weights sum to zero."
        inspect_code = f"print(len({name}), len(weights_{suffix}))"
        body = f"weight_total = sum(weights_{suffix})\nresult = sum(value * weight for value, weight in zip({name}, weights_{suffix})) / weight_total if weight_total else None"
        expected = (
            sum(v * w for v, w in zip(values, weights)) / sum(weights)
            if sum(weights)
            else None
        )
        check = f"assert result == {expected!r}"
    elif family == 18:
        label = "population_statistics"
        task = f"Inspect {name}, then store its median and population variance in result under 'middle' and 'variance'. Use population, not sample variance."
        body = f"import statistics\nresult = {{'middle': statistics.median({name}), 'variance': statistics.pvariance({name})}}"
        check = f"assert result == {{'middle': {statistics.median(values)!r}, 'variance': {statistics.pvariance(values)!r}}}"
    elif family == 19:
        label = "inspect_left_join"
        setup = f"{name} = pd.DataFrame({{'ticket': [17, 12, 18], 'owner': [2, 9, 1]}})\nlookup_{suffix} = pd.DataFrame({{'owner': [1, 2], 'display': ['Ivo', 'Nia']}})"
        task = f"Inspect {name} and lookup_{suffix}. Left join on owner, preserving input ticket order. Store ticket and display records in result, with None for unknown owners."
        inspect_code = (
            f"print({name}.columns.tolist()); print(lookup_{suffix}.columns.tolist())"
        )
        body = f"joined = {name}.merge(lookup_{suffix}, on='owner', how='left', sort=False)\nresult = [{{'ticket': int(row.ticket), 'display': None if pd.isna(row.display) else row.display}} for row in joined.itertuples()]"
        check = "assert result == [{'ticket': 17, 'display': 'Nia'}, {'ticket': 12, 'display': None}, {'ticket': 18, 'display': 'Ivo'}]"
    elif family == 20:
        label = "latest_record_per_key"
        rows = [
            {"key": key, "version": i, "value": value}
            for i, (key, value) in enumerate(zip(group_values, values))
        ]
        setup = f"{name} = {rows!r}"
        task = f"Inspect {name}, then keep each key's value from the highest version. Store a key-to-value dict in result."
        body = f"result = {{}}\nfor row in sorted({name}, key=lambda row: row['version']):\n    result[row['key']] = row['value']"
        check = f"assert result == {dict(zip(group_values, values))!r}"
    elif family == 21:
        label = "inspect_dates_before_bucket"
        dates = [f"2033-0{i % 3 + 1}-{i + 1:02d}" for i in range(len(values))]
        setup = (
            f"{name} = pd.DataFrame({{'timestamp': {dates!r}, {column!r}: {values!r}}})"
        )
        task = f"Inspect {name}, parse timestamp as dates, and aggregate {column!r} by YYYY-MM into result. Do not modify the source frame."
        inspect_code = f"print({name}.dtypes.astype(str).to_dict()); print({name}.head(2).to_dict('records'))"
        body = f"months = pd.to_datetime({name}['timestamp']).dt.strftime('%Y-%m')\nresult = {name}.groupby(months)[{column!r}].sum().to_dict()"
        expected = {}
        for date, value in zip(dates, values):
            expected[date[:7]] = expected.get(date[:7], 0) + value
        check = f"assert result == {expected!r}\nassert {name}['timestamp'].tolist() == {dates!r}"
    elif family in (22, 23, 24):
        label = {
            22: "repair_dictionary_parentheses",
            23: "repair_fenced_code",
            24: "repair_indentation",
        }[family]
        task = f"Compute the total of existing {name} into result. Correct any syntax error using raw executable Python, not Markdown."
        history = [
            {
                22: f"result = {{'total': sum({name}))}}",
                23: f"```python\nresult = sum({name})\n```",
                24: f"if {name}:\nresult = sum({name})",
            }[family]
        ]
        inspect_code = f"print(type({name}).__name__, len({name}))"
    elif family == 25:
        label = "preserve_successful_mutation"
        setup += f"\n{name}.append(101)\nresult = None"
        task = f"The append of 101 to {name} has already succeeded. Do not repeat it. Inspect the list and store its current length in result."
        body = f"result = len({name})"
        check = (
            f"assert result == {len(values) + 1}\nassert {name} == {values + [101]!r}"
        )
    elif family == 26:
        label = "reuse_existing_result_object"
        setup += f"\nresult = [value * 3 for value in {name}]\noriginal_result = result"
        task = f"Reuse the existing result list derived from {name}. Reverse it in place without replacing it."
        inspect_code = "print(type(result).__name__, result[:3])"
        body = "result.reverse()"
        check = f"assert result is original_result\nassert result == {[v * 3 for v in reversed(values)]!r}"
    elif family == 27:
        label = "finish_after_successful_code"
        task = f"Store the sum of {name} in result and notify the coordinator with done(). If result was already computed correctly, do not redo successful work."
        history = [f"result = sum({name})\nprint(result)"]
        inspect_code = "print(result)"
        body = ""
    elif family == 28:
        label = "report_retrieval_failure"
        setup = "class Unavailable:\n    def fetch(self):\n        raise ConnectionError('Fixture source unavailable; no observations returned')\n"
        setup += f"{obj} = Unavailable()"
        task = f"Use {obj}.fetch() to retrieve observations. If retrieval fails, leave result as None and status='unavailable'; do not invent or substitute data."
        history = [f"result = {obj}.fetch()"]
        inspect_code = f"print(type({obj}).__name__)"
        body = "result = None\nstatus = 'unavailable'"
        check = "assert result is None\nassert status == 'unavailable'"
    elif family == 30:
        label = "inspect_nested_json"
        setup = f"{name} = {json.dumps({'payload': {'samples': values}, 'unit': 'items'})!r}"
        task = f"Parse existing JSON string {name}, inspect its top-level fields, and sum payload.samples into result."
        inspect_code = f"parsed = json.loads({name})\nprint(parsed.keys())"
        body = (
            f"parsed = json.loads({name})\nresult = sum(parsed['payload']['samples'])"
        )
    elif family == 31:
        label = "group_with_all_missing_values"
        setup = f"{name} = pd.DataFrame({{{group!r}: ['amber', 'amber', 'blue', 'blue', 'green'], {column!r}: [{values[0]}, None, {values[1]}, {values[2]}, None]}})"
        task = f"Inspect {name}. Compute mean {column!r} by {group!r} into a dict result, keeping all-missing groups as None rather than dropping them."
        inspect_code = (
            f"print({name}.columns.tolist()); print({name}.isna().sum().to_dict())"
        )
        body = f"means = {name}.groupby({group!r})[{column!r}].mean()\nresult = {{key: None if pd.isna(value) else float(value) for key, value in means.items()}}"
        check = f"assert result == {{'amber': {float(values[0])}, 'blue': {(values[1] + values[2]) / 2}, 'green': None}}"
    elif family == 32:
        label = "inspect_mode_argument"
        setup += "\nclass Reducer:\n    def aggregate(self, values, *, operation):\n        \"\"\"operation must be 'total' for sum or 'size' for count. Returns one number.\"\"\"\n        return sum(values) if operation == 'total' else len(values) if operation == 'size' else None\n"
        setup += f"{obj} = Reducer()"
        task = f"Inspect {obj}.aggregate and use its documented operation for summing {name}. Store its return value as result."
        inspect_code = f"help({obj}.aggregate)"
        body = f"result = {obj}.aggregate({name}, operation='total')"
    elif family == 33:
        label = "percentage_points_vs_percent"
        initial, final = sorted(rng.sample(range(5, 95), 2))
        setup = f"{name} = {{'before': {initial}, 'after': {final}}}"
        task = f"Inspect percentage observations {name}. Store percentage-point change as result['points'] and relative percent growth as result['percent']."
        inspect_code = f"print({name})"
        body = f"before = {name}['before']\nafter = {name}['after']\nresult = {{'points': after - before, 'percent': (after / before - 1) * 100}}"
        check = f"assert result['points'] == {final - initial}\nassert abs(result['percent'] - {(final / initial - 1) * 100!r}) < 1e-9"
    elif family == 34:
        label = "stable_tie_sorting"
        rows = [
            {"label": f"item{i}", "priority": value % 3}
            for i, value in enumerate(values)
        ]
        setup = f"{name} = {rows!r}"
        task = f"Inspect {name} and sort rows by descending priority, preserving source order for ties. Save the sorted list in result."
        body = f"result = sorted({name}, key=lambda row: row['priority'], reverse=True)"
        check = f"assert result == {sorted(rows, key=lambda row: row['priority'], reverse=True)!r}"
    elif family == 35:
        label = "extract_numeric_text"
        text = " ".join(f"count={v}" for v in values)
        setup = f"{name} = {text!r}"
        task = f"Extract the integers following count= from existing text {name}, in order, into result. Inspect the text; do not use eval()."
        inspect_code = f"print({name})"
        body = f"import re\nresult = [int(value) for value in re.findall(r'count=(\\d+)', {name})]"
        check = f"assert result == {values!r}"
    elif family == 36:
        label = "finite_array_statistics"
        setup = f"{name} = np.array({values!r} + [float('nan'), float('inf')])"
        task = f"Inspect {name}'s shape and finite mask. Compute the mean of finite entries into result, preserving the array."
        inspect_code = f"print({name}.shape); print(np.isfinite({name}).tolist())"
        body = f"result = float({name}[np.isfinite({name})].mean())"
        check = f"assert abs(result - {statistics.mean(values)!r}) < 1e-9\nassert len({name}) == {len(values) + 2}"
    elif family == 37:
        label = "rolling_window"
        window = rng.randint(2, 3)
        task = f"Inspect {name}. Compute complete trailing window sums of width {window} into result. Exclude incomplete windows."
        body = f"result = [sum({name}[end-{window}:end]) for end in range({window}, len({name})+1)]"
        check = f"assert result == {[sum(values[i - window : i]) for i in range(window, len(values) + 1)]!r}"
    elif family == 38:
        label = "normalize_strings_without_mutation"
        words = [" Amber ", "BLUE", " amber", "Green "]
        setup = f"{name} = {words!r}"
        task = f"Inspect {name}. Strip whitespace and lowercase strings into result, preserving order and the original list."
        body = f"result = [value.strip().lower() for value in {name}]"
        check = f"assert result == ['amber', 'blue', 'amber', 'green']\nassert {name} == {words!r}"
    elif family == 39:
        label = "retain_full_precision"
        floats = [v / 7 for v in values]
        setup = f"{name} = {floats!r}"
        task = f"Inspect {name}. Store its full-precision sum in result; do not round the stored value."
        body = f"result = sum({name})"
        check = f"assert result == {sum(floats)!r}"
    elif family == 40:
        label = "heldout_pivot_counts"
        setup = f"{name} = pd.DataFrame({{'kind': {group_values!r}, 'flag': {[v % 2 for v in values]!r}}})"
        task = (
            f"Inspect {name}. Count each kind into a plain dict result, ignoring flag."
        )
        inspect_code = (
            f"print({name}.columns.tolist()); print({name}.head(2).to_dict('records'))"
        )
        body = f"result = {name}['kind'].value_counts().to_dict()"
        check = f"assert result == {dict(Counter(group_values))!r}"
    elif family == 41:
        label = "heldout_partition_maximum"
        setup = f"{name} = pd.DataFrame({{{group!r}: {group_values!r}, {column!r}: {values!r}}})"
        task = f"Inspect {name}; find maximum {column!r} for each {group!r} and save a dict result."
        inspect_code = f"print({name}.head(3).to_dict('records'))"
        body = f"result = {name}.groupby({group!r})[{column!r}].max().to_dict()"
        check = f"assert result == {dict((k, max(v for key, v in zip(group_values, values) if key == k)) for k in set(group_values))!r}"
    elif family == 42:
        label = "heldout_cumulative_per_group"
        setup = f"{name} = {list(zip(group_values, values))!r}"
        task = f"Inspect {name}. For each (key, value), append that key's running sum to result, preserving row order."
        body = f"totals = {{}}\nresult = []\nfor key, value in {name}:\n    totals[key] = totals.get(key, 0) + value\n    result.append(totals[key])"
        totals, expected = {}, []
        for key, value in zip(group_values, values):
            totals[key] = totals.get(key, 0) + value
            expected.append(totals[key])
        check = f"assert result == {expected!r}"
    elif family == 43:
        label = "heldout_lookup_then_group"
        setup = f"{name} = {list(zip(range(len(values)), values))!r}\nlabels_{suffix} = {dict(enumerate(group_values))!r}"
        task = f"Inspect {name} and labels_{suffix}. Map record IDs to labels and sum values by label into a dict result."
        body = f"result = {{}}\nfor identifier, value in {name}:\n    label = labels_{suffix}[identifier]\n    result[label] = result.get(label, 0) + value"
        expected = {}
        for key, value in zip(group_values, values):
            expected[key] = expected.get(key, 0) + value
        check = f"assert result == {expected!r}"
    elif family == 44:
        label = "heldout_nested_container"
        setup = f"{name} = {{'payload': {{'entries': {values!r}}}, 'unit': 'items'}}"
        task = f"Inspect existing mapping {name}. Find the largest payload.entries value and save it in result."
        inspect_code = f"print({name}.keys()); print({name}['payload'].keys())"
        body = f"result = max({name}['payload']['entries'])"
        check = f"assert result == {max(values)}"
    elif family == 45:
        label = "heldout_align_missing_keys"
        setup = f"{name} = {{'a': {values[0]}, 'c': {values[1]}}}\nrequested_{suffix} = ['a', 'b', 'c', 'd']"
        task = f"Inspect {name}. Align its values to requested_{suffix}, using None for absent keys; save the list as result."
        inspect_code = f"print({name}.keys()); print(requested_{suffix})"
        body = f"result = [{name}.get(key) for key in requested_{suffix}]"
        check = f"assert result == [{values[0]}, None, {values[1]}, None]"
    elif family == 46:
        label = "heldout_paginated_callable"
        setup = (
            'class Pages:\n    def fetch(self, *, cursor):\n        """cursor=0 gives the first page, cursor=1 the last. Each page is (values, next_cursor); None means end."""\n        return ('
            + repr(values[:2])
            + ", 1) if cursor == 0 else ("
            + repr(values[2:])
            + ", None)\n"
        )
        setup += f"{obj} = Pages()"
        task = f"Inspect {obj}.fetch, retrieve all pages starting at cursor 0, and sum all retrieved values into result."
        inspect_code = f"help({obj}.fetch)"
        body = f"cursor = 0\nall_values = []\nwhile cursor is not None:\n    page, cursor = {obj}.fetch(cursor=cursor)\n    all_values.extend(page)\nresult = sum(all_values)"
    elif family == 47:
        label = "heldout_rounded_copy"
        floats = [v / 13 for v in values]
        setup = f"{name} = {floats!r}"
        task = f"Inspect {name}. Make a two-decimal rounded copy in result, preserving the raw input list."
        body = f"result = [round(value, 2) for value in {name}]"
        check = f"assert result == {[round(v, 2) for v in floats]!r}\nassert {name} == {floats!r}"
    elif family == 48:
        label = "heldout_partial_update_recovery"
        task = f"Append 107 to {name} once, then store its sum in result. If the append succeeded before a later exception, do not repeat it. Inspect the actual state first."
        history = [
            f"{name}.append(107)\nraise ValueError('Append succeeded before this failure; inspect before retrying')"
        ]
        body = f"result = sum({name})"
        check = (
            f"assert {name} == {values + [107]!r}\nassert result == {sum(values) + 107}"
        )
    elif family == 49:
        label = "heldout_json_select_filter"
        setup = f"{name} = {json.dumps({'items': [{'value': v, 'keep': v % 2 == 0} for v in values]})!r}"
        task = f"Inspect JSON text {name}. Parse items, select records where keep is true, and store their numeric values in result."
        inspect_code = f"parsed = json.loads({name})\nprint(parsed.keys()); print(parsed['items'][:2])"
        body = f"parsed = json.loads({name})\nresult = [row['value'] for row in parsed['items'] if row['keep']]"
        check = f"assert result == {[v for v in values if v % 2 == 0]!r}"
    else:
        raise ValueError(family)

    task += " Use only the existing workspace data. Call done() with a concise verified result when finished."
    final = (body + "\n" if body else "") + "done(str(result))"
    actions = [(code, False) for code in history] + [
        (inspect_code, True),
        (final, True),
    ]
    return {
        "setup": setup,
        "instruction": task,
        "actions": actions,
        "check": check,
        "family": label,
    }


def make_record(index, family, seed, split, startup):
    spec = scenario(index, family, seed)
    namespace = {
        "np": np,
        "pd": pd,
        "json": json,
        "math": math,
        "statistics": statistics,
    }
    exec("import inspect", namespace)
    exec(spec["setup"], namespace)
    turns = list(startup) + [{"role": "user", "content": spec["instruction"]}]
    supervised_positions = [
        position for position, (_, supervised) in enumerate(spec["actions"]) if supervised
    ]
    variant = index // (40 if split == "train" else 10)
    chosen_position = supervised_positions[-1 if variant % 5 < 3 else 0]
    for position, (code, supervised) in enumerate(spec["actions"]):
        prompt = serialize(turns)
        feedback = execute(code, namespace)
        if supervised:
            if feedback["exception"]:
                raise AssertionError((spec["family"], code, feedback["exception"]))
            if position == chosen_position:
                chosen = {
                    "prompt": prompt,
                    "completion": code,
                    "previous": [item[0] for item in spec["actions"][:position]],
                    "target_feedback": feedback,
                    "inspect_only": feedback["done"] is None,
                }
        elif (
            feedback["exception"] is None
            and spec["family"] != "finish_after_successful_code"
        ):
            raise AssertionError(f"Intended error did not occur: {spec['family']}")
        turns += [
            {"role": "assistant", "content": code},
            {"role": "user", "content": json.dumps(feedback)},
        ]
    exec(spec["check"], namespace)
    assert feedback["done"] is not None
    return {
        "id": f"{split}-expanded-{index:05d}",
        "family": spec["family"],
        "setup": spec["setup"],
        **chosen,
        "check": spec["check"]
        if not chosen["inspect_only"]
        else "assert feedback['exception'] is None",
        "verification": "target executed without error; complete reference trajectory passed independent state assertions",
    }


def verify_legacy(record):
    namespace = {
        "np": np,
        "pd": pd,
        "json": json,
        "math": math,
        "statistics": statistics,
    }
    exec("import inspect", namespace)
    exec(record["setup"], namespace)
    for code in record["previous"]:
        execute(code, namespace)
    feedback = execute(record["completion"], namespace)
    assert feedback["exception"] is None
    namespace["feedback"] = feedback
    exec(record["check"], namespace)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "validation.jsonl", "manifest.json"):
        if (args.output / name).exists():
            raise FileExistsError(args.output / name)
    legacy_path = REPO / "model/train/data/raw/train.jsonl"
    legacy = [json.loads(line) for line in legacy_path.read_text().splitlines()]
    assert len(legacy) == 500
    for record in legacy:
        verify_legacy(record)
    startup = worker_context()
    train = legacy + [
        make_record(index, index % 40, 862019, "train", startup)
        for index in range(4500)
    ]
    validation = [
        make_record(index, 40 + index % 10, 429371, "validation", startup)
        for index in range(500)
    ]
    assert len({item["prompt"] for item in train}) == len(train)
    assert len({item["prompt"] for item in validation}) == len(validation)
    assert not {item["prompt"] for item in train}.intersection(
        item["prompt"] for item in validation
    )
    new_families = {item["family"] for item in train}
    assert not new_families.intersection(item["family"] for item in validation)
    manifest = {
        "serialization": "task-code-console-v1",
        "system_role": False,
        "benchmark_cases_used": False,
        "legacy_train_examples": 500,
        "new_train_examples": 4500,
        "validation_split": "held-out task compositions",
        "generator_sha256": sha256(Path(__file__)),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "splits": {},
    }
    for split, records in [("train", train), ("validation", validation)]:
        path = args.output / f"{split}.jsonl"
        write_jsonl(path, records, ensure_ascii=False)
        manifest["splits"][split] = {
            "examples": len(records),
            "families": dict(Counter(record["family"] for record in records)),
            "sha256": sha256(path),
        }
        print(
            f"{split}: {len(records)} verified samples across {len(manifest['splits'][split]['families'])} families -> {path}",
            flush=True,
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
