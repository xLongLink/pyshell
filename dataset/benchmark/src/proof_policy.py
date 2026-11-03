"""Fixed acceptance rules for CPU execution-proof experiments."""

MAX_CALLS = 4
FINAL_TASKS = 120
REQUIRED_PASSES = 97


def development_ready(summary):
    return (
        summary["complete"]
        and summary["tasks"] == 60
        and summary["success_rate"] >= 0.9
        and all(group["passed"] / group["tasks"] >= 0.6 for group in summary["by_family"].values())
    )


def final_passed(tasks, passed):
    return tasks == FINAL_TASKS and passed >= REQUIRED_PASSES
