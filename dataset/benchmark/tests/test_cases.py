"""Check fixture setup and grading independently of any model provider."""

import pytest
from importlib.util import find_spec
from dataset.benchmark.src.cases import CASES

SOLUTIONS = {
    "summary_statistics": [
        "import statistics\nresult = dict(count=len(measurements), mean=statistics.mean(measurements), median=statistics.median(measurements), population_std=statistics.pstdev(measurements))"
    ],
    "grouped_revenue": ["result = sales.loc[sales.status != 'cancelled'].groupby('region').amount.sum().to_dict()"],
    "monthly_totals": [
        "result = {}\nfor record in records:\n    month = record['date'][:7]\n    if month in ('2025-01', '2025-02'):\n        result[month] = result.get(month, 0) + record['amount']"
    ],
    "missing_values": [
        "result = {sensor: None if pd.isna(value) else value for sensor, value in readings.groupby('sensor').value.mean().items()}"
    ],
    "left_join": [
        "joined = orders.merge(customers, on='customer_id', how='left')\nresult = [{'order_id': row.order_id, 'customer_name': None if pd.isna(row.name) else row.name} for row in joined.itertuples()]"
    ],
    "inspect_callable": ["help(calibration.convert)\nresult = calibration.convert(raw_values, scale=2, offset=-1)"],
    "recover_execution": [
        "try:\n    result = sum(reader.read())\nexcept RuntimeError:\n    reader.recover()\n    result = sum(reader.read())"
    ],
    "stateful_followup": ["result = [value ** 2 for value in values]", "result.append(8 ** 2)"],
    "missing_input": ["result = None\nstatus = 'missing_data'"],
    "linear_system": ["import numpy as np\nresult = np.linalg.solve(coefficients, constants).tolist()"],
}


class TestCase:
    def test_unique_case_ids_and_syntax(self, subtests):
        assert len({case.id for case in CASES}) == len(CASES)
        for case in CASES:
            with subtests.test(case=case.id):
                compile(case.setup, case.id, "exec")
                assert case.steps
                assert len(SOLUTIONS[case.id]) == len(case.steps)
                for step in case.steps:
                    compile(step.verify, case.id, "exec")

    @pytest.mark.skipif(
        not (find_spec("pandas") and find_spec("numpy")), reason="Run with the API environment to validate numerical fixtures"
    )
    def test_reference_solutions_pass_every_oracle(self, subtests):
        for case in CASES:
            with subtests.test(case=case.id):
                namespace = {}
                exec(case.setup, namespace)
                for solution, step in zip(SOLUTIONS[case.id], case.steps, strict=True):
                    exec(solution, namespace)
                    exec(step.verify, namespace)

    @pytest.mark.skipif(
        not (find_spec("pandas") and find_spec("numpy")), reason="Run with the API environment to validate numerical fixtures"
    )
    def test_wrong_results_are_rejected(self, subtests):
        for case in CASES:
            with subtests.test(case=case.id):
                namespace = {}
                exec(case.setup, namespace)
                exec(SOLUTIONS[case.id][0], namespace)
                namespace["result"] = "fabricated"
                with pytest.raises((AssertionError, TypeError)):
                    exec(case.steps[0].verify, namespace)
