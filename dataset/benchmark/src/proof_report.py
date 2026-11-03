"""Execution metrics, without inference or grading side effects."""

from collections import Counter


def execution_metrics(rows):
    groups = {}
    statuses = Counter()
    for row in rows:
        group = groups.setdefault(row["family"], {"tasks": 0, "passed": 0})
        group["tasks"] += 1
        group["passed"] += row["status"] == "passed"
        statuses[row["status"]] += 1
    passed = statuses["passed"]
    return {
        "tasks": len(rows),
        "passed": passed,
        "success_rate": passed / len(rows),
        "by_family": dict(sorted(groups.items())),
        "statuses": dict(statuses),
    }
