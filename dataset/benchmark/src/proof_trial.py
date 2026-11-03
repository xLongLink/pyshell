"""Run a bounded task while keeping setup and grading out of model context."""

import json
import time
from dataset.benchmark.src.proof_policy import MAX_CALLS
from dataset.benchmark.src.serialization import serialize


def trial(worker, record, generate):
    setup = worker.execute(record["setup"], "reset")
    if setup["exception"]:
        raise RuntimeError(f"Invalid fixture {record['id']}: {setup['exception']}")
    turns = [{"role": "user", "content": record["instruction"]}]
    stages = [{"instruction": record["instruction"], "check": record["check"]}]
    if record["followup"]:
        stages.append(record["followup"])
    row = {"id": record["id"], "family": record["family"], "steps": [], "status": "call_limit"}
    for stage_index, stage in enumerate(stages):
        if stage_index:
            worker.execute("saved_result = result")
            turns.append({"role": "user", "content": stage["instruction"]})
        for _ in range(MAX_CALLS):
            started = time.monotonic()
            source, ended = generate(serialize(turns))
            step = {"source": source, "stage": stage_index, "end_marker_seen": ended, "seconds": time.monotonic() - started}
            row["steps"].append(step)
            if not ended or not source.strip():
                row["status"] = "boundary_error"
                return row
            result = worker.execute(source)
            step["feedback"] = result
            turns.extend([{"role": "assistant", "content": source}, {"role": "user", "content": json.dumps(result)}])
            if result["done"] is not None and not result["exception"]:
                grade = worker.execute(stage["check"])
                if grade["exception"]:
                    row.update(status="incorrect", error=grade["exception"])
                    return row
                break
        else:
            return row
    row["status"] = "passed"
    return row
