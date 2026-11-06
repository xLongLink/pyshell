from pyshell.serialization import Prompt


def trial(worker, record, generate, calls):
    setup = worker.execute(record["setup"], "reset")
    if setup["exception"]:
        raise RuntimeError(f"Invalid fixture {record['id']}: {setup['exception']}")
    prompt = Prompt()
    prompt.append({"role": "user", "content": record["instruction"]})
    stages = [{"instruction": record["instruction"], "check": record["check"]}]
    if record["followup"]:
        stages.append(record["followup"])
    row = {"id": record["id"], "family": record["family"], "steps": [], "status": "call_limit"}
    for stage_index, stage in enumerate(stages):
        if stage_index:
            worker.execute("saved_result = result")
            prompt.append({"role": "user", "content": stage["instruction"]})
        for _ in range(calls):
            source, ended = generate(prompt.render())
            step = {"source": source}
            row["steps"].append(step)
            if not ended or not source.strip():
                row["status"] = "boundary_error"
                return row
            result = worker.execute(source)
            step["feedback"] = result
            prompt.append({"role": "assistant", "content": source})
            prompt.append({"role": "console", "content": result})
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
