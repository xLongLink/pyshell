"""Deterministic tasks with checks kept outside the worker conversation."""

from textwrap import dedent
from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    """One instruction and its independent workspace checks."""

    instruction: str
    verify: str


@dataclass(frozen=True)
class Case:
    """An isolated task, optionally containing stateful follow-ups."""

    id: str
    category: str
    setup: str
    steps: tuple[Step, ...]


def source(text: str) -> str:
    """Normalize embedded Python scripts."""
    return dedent(text).strip()


CASES = (
    Case(
        "summary_statistics",
        "computation",
        "measurements = [12, 7, 19, 7, 25, 14]",
        (
            Step(
                "The workspace contains measurements. Leave result as a dict with count, mean, median, and population_std. "
                "Do not change measurements. Call done() when finished.",
                source("""
                import math
                assert measurements == [12, 7, 19, 7, 25, 14]
                assert result['count'] == 6
                assert math.isclose(result['mean'], 14.0)
                assert math.isclose(result['median'], 13.0)
                assert math.isclose(result['population_std'], math.sqrt(41.333333333333336))
            """),
            ),
        ),
    ),
    Case(
        "grouped_revenue",
        "dataframe",
        source("""
            import pandas as pd
            sales = pd.DataFrame({
                'region': ['north', 'south', 'north', 'south', 'west'],
                'amount': [100, 80, 30, 20, 50],
                'status': ['paid', 'paid', 'cancelled', 'paid', 'paid'],
            })
        """),
        (
            Step(
                "sales is a DataFrame. Sum amount by region, excluding cancelled orders. "
                "Leave result as a plain dict from region to total. Preserve sales and call done().",
                source("""
                assert result == {'north': 100, 'south': 100, 'west': 50}
                assert sales.to_dict('list') == {
                    'region': ['north', 'south', 'north', 'south', 'west'],
                    'amount': [100, 80, 30, 20, 50],
                    'status': ['paid', 'paid', 'cancelled', 'paid', 'paid'],
                }
            """),
            ),
        ),
    ),
    Case(
        "monthly_totals",
        "dates",
        source("""
            records = [
                {'date': '2025-02-28', 'amount': 9},
                {'date': '2025-01-01', 'amount': 10},
                {'date': '2025-01-31', 'amount': 5},
                {'date': '2024-12-31', 'amount': 100},
                {'date': '2025-03-01', 'amount': 4},
            ]
        """),
        (
            Step(
                "Using records, calculate monthly amount totals for January and February 2025 only. "
                "Leave result as a dict keyed by YYYY-MM. Call done().",
                "assert result == {'2025-01': 15, '2025-02': 9}",
            ),
        ),
    ),
    Case(
        "missing_values",
        "dataframe",
        source("""
            import pandas as pd
            readings = pd.DataFrame({'sensor': ['a', 'a', 'b', 'b', 'c'], 'value': [2, None, 4, 8, None]})
        """),
        (
            Step(
                "Calculate each sensor's mean value from readings, ignoring missing values. "
                "Include sensors with no readings as None. Leave result as a plain dict. Call done().",
                "assert result == {'a': 2.0, 'b': 6.0, 'c': None}",
            ),
        ),
    ),
    Case(
        "left_join",
        "dataframe",
        source("""
            import pandas as pd
            orders = pd.DataFrame({'order_id': [3, 1, 2], 'customer_id': [10, 99, 20]})
            customers = pd.DataFrame({'customer_id': [10, 20], 'name': ['Ada', 'Lin']})
        """),
        (
            Step(
                "Join orders to customers, preserving all orders and their original order. "
                "Leave result as a list of dicts with order_id and customer_name; use None for unknown customers. Call done().",
                source("""
                assert result == [
                    {'order_id': 3, 'customer_name': 'Ada'},
                    {'order_id': 1, 'customer_name': None},
                    {'order_id': 2, 'customer_name': 'Lin'},
                ]
            """),
            ),
        ),
    ),
    Case(
        "inspect_callable",
        "inspection",
        source('''
            class Calibration:
                """Use convert to calibrate readings; inspect its documentation before calling."""

                def __init__(self):
                    self.calls = []

                def convert(self, values, *, scale, offset):
                    """Return (value + offset) * scale for each value. scale and offset are keyword-only."""
                    self.calls.append((list(values), scale, offset))
                    return [(value + offset) * scale for value in values]

            calibration = Calibration()
            raw_values = [1, 3, 5]
        '''),
        (
            Step(
                "Inspect calibration.convert before using it. Use that method to calibrate raw_values with "
                "scale 2 and offset -1. Leave its returned list in result. Call done().",
                source("""
                assert result == [0, 4, 8]
                assert calibration.calls == [([1, 3, 5], 2, -1)]
            """),
            ),
        ),
    ),
    Case(
        "recover_execution",
        "recovery",
        source('''
            class BatchReader:
                """A stateful reader. Read the exception instructions if a read fails."""

                def __init__(self):
                    self.attempts = 0
                    self.ready = False
                    self.recovered = False

                def read(self):
                    """Return the current batch of numeric values."""
                    self.attempts += 1
                    if not self.ready:
                        raise RuntimeError('Reader not ready: call reader.recover(), then retry reader.read().')
                    return [6, 10, 14]

                def recover(self):
                    """Recover after a failed read; this must not precede the initial read attempt."""
                    if self.attempts == 0:
                        raise RuntimeError('Attempt a read before recovery.')
                    self.ready = True
                    self.recovered = True

            reader = BatchReader()
        '''),
        (
            Step(
                "Use reader.read() to retrieve the batch and leave its sum in result. "
                "If it fails, follow its recovery instructions rather than inventing data. Call done().",
                source("""
                assert result == 30
                assert reader.recovered
                assert reader.attempts >= 2
            """),
            ),
        ),
    ),
    Case(
        "stateful_followup",
        "state",
        "values = [2, 4, 6]",
        (
            Step(
                "Create a list named result containing the squares of values, without changing values. Call done().",
                "assert result == [4, 16, 36]\nassert values == [2, 4, 6]\nbenchmark_original_result = result",
            ),
            Step(
                "Update that same result list in place by appending the square of 8. Leave values unchanged. Call done().",
                source("""
                    assert result == [4, 16, 36, 64]
                    assert result is benchmark_original_result
                    assert values == [2, 4, 6]
                """),
            ),
        ),
    ),
    Case(
        "missing_input",
        "limitations",
        "available_data = {'2024': [10, 20]}\nresult = None",
        (
            Step(
                "Calculate the 2025 total using only available_data. If 2025 is absent, leave result as None "
                "and set status to 'missing_data'. Otherwise set result to the total and status to 'ok'. "
                "Do not substitute another year. Call done() with the relevant limitation if blocked.",
                "assert result is None\nassert status == 'missing_data'\nassert available_data == {'2024': [10, 20]}",
            ),
        ),
    ),
    Case(
        "linear_system",
        "numerical",
        "coefficients = [[3, 1], [1, 2]]\nconstants = [9, 8]",
        (
            Step(
                "Solve coefficients @ x = constants. Leave result as a plain list of the two solution values. "
                "Verify the solution numerically before calling done().",
                source("""
                import math
                assert isinstance(result, list) and len(result) == 2
                assert math.isclose(result[0], 2.0, rel_tol=1e-8, abs_tol=1e-8)
                assert math.isclose(result[1], 3.0, rel_tol=1e-8, abs_tol=1e-8)
            """),
            ),
        ),
    ),
)
