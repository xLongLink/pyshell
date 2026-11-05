"""Fixed acceptance rules for CPU execution-proof experiments."""

import math

MAX_CALLS = 4
FINAL_TASKS = 120
FINAL_SUCCESS_RATE = 0.8
REQUIRED_PASSES = math.floor(FINAL_TASKS * FINAL_SUCCESS_RATE) + 1


def acceptance_metadata():
    return {"test_tasks": FINAL_TASKS, "required_passes": REQUIRED_PASSES, "strictly_greater_than": FINAL_SUCCESS_RATE, "max_calls_per_stage": MAX_CALLS}


def development_ready(summary):
    return (
        summary["complete"]
        and summary["tasks"] == 60
        and summary["success_rate"] >= 0.9
        and all(group["passed"] / group["tasks"] >= 0.6 for group in summary["by_family"].values())
    )


def final_passed(tasks, passed):
    return tasks == FINAL_TASKS and passed >= REQUIRED_PASSES
