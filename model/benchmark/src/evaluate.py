import json
from contextlib import closing
from collections import Counter
from model.benchmark.src.trial import trial
from model.benchmark.src.corpus import load_tasks
from model.benchmark.src.execution import Worker


def run(config):
    if config.output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark results: {config.output}")
    records = load_tasks(config)
    if not records:
        raise ValueError("Evaluation tasks must not be empty")
    native = json.loads((config.adapter / "adapter_config.json").read_text())
    from model.benchmark.src.inference import HFCompletion

    generate = HFCompletion(
        config.adapter,
        config.threads,
        config.device,
        base_model=native["base_model_name_or_path"],
        revision=native.get("revision"),
        max_new_tokens=config.tokens,
    )
    config.output.parent.mkdir(parents=True, exist_ok=True)
    statuses = Counter()
    with config.output.open("x") as file, closing(Worker()) as worker:
        for index, record in enumerate(records, 1):
            row = trial(worker, record, generate, config.calls)
            file.write(json.dumps(row) + "\n")
            file.flush()
            statuses[row["status"]] += 1
            print(f"{index}/{len(records)}: {record['id']} {row['status']}", flush=True)
    summary = {
        "tasks": len(records),
        "passed": statuses["passed"],
        "success_rate": statuses["passed"] / len(records),
        "statuses": dict(statuses),
    }
    print(f"Passed {summary['passed']}/{summary['tasks']} ({summary['success_rate']:.1%})", flush=True)
    return summary
