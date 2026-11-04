"""Summaries that keep reliability separate from successful-task latency."""

import math
from typing import Any
from statistics import mean, median
from collections import Counter


def distribution(values: list[float]) -> dict[str, float | None]:
    """Use the nearest-rank p95, including for small samples."""
    if not values:
        return {"median": None, "p95": None}
    ordered = sorted(values)
    return {"median": median(ordered), "p95": ordered[math.ceil(0.95 * len(ordered)) - 1]}


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate all attempted trials, retaining per-case results."""
    passed = [trial for trial in trials if trial["status"] == "passed"]
    calls = [call for trial in trials for call in trial["call_timings"]]
    case_ids = sorted({trial["case"] for trial in trials})
    return {
        "trials": len(trials),
        "passed": len(passed),
        "success_rate": len(passed) / len(trials) if trials else None,
        "statuses": dict(Counter(trial["status"] for trial in trials)),
        "successful_task_seconds": distribution([trial["task_seconds"] for trial in passed]),
        "all_task_seconds": distribution([trial["task_seconds"] for trial in trials]),
        "initialization_seconds": distribution(
            [trial["initialization_seconds"] for trial in trials if trial.get("initialization_seconds") is not None]
        ),
        "first_text_seconds": distribution([call["first_text_seconds"] for call in calls if call["first_text_seconds"] is not None]),
        "complete_code_seconds": distribution(
            [call["total_seconds"] for call in calls if call["total_seconds"] is not None and call["error"] is None]
        ),
        "mean_calls": mean(trial["calls"] for trial in trials) if trials else None,
        "input_tokens": sum(trial["input_tokens"] for trial in trials),
        "output_tokens": sum(trial["output_tokens"] for trial in trials),
        "unmeasured_calls": sum(call["usage"] is None or bool(call["usage"].get("unmeasured_requests")) for call in calls),
        "per_case": {
            case_id: {
                "trials": sum(trial["case"] == case_id for trial in trials),
                "passed": sum(trial["case"] == case_id for trial in passed),
                "statuses": dict(Counter(trial["status"] for trial in trials if trial["case"] == case_id)),
                "successful_task_seconds": distribution([trial["task_seconds"] for trial in passed if trial["case"] == case_id]),
            }
            for case_id in case_ids
        },
    }
