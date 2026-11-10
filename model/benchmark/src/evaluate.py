import json
from pathlib import Path
from contextlib import closing
from model.benchmark.src.trial import trial
from model.benchmark.src.settings import Settings
from model.benchmark.src.execution import Worker


def run(config: Settings) -> None:
    """Write execution results for the configured evaluation tasks."""
    if config.output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark results: {config.output}")
    path = Path(config.data) / "dev-tasks.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if not records:
        raise ValueError("Evaluation tasks must not be empty")
    from model.benchmark.src.inference import HFCompletion

    generate = HFCompletion(config)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    with config.output.open("x") as file, closing(Worker()) as worker:
        for index, record in enumerate(records, 1):
            row = trial(worker, record, generate, config.calls)
            file.write(json.dumps(row) + "\n")
            file.flush()
            passed += row["status"] == "passed"
            print(f"{index}/{len(records)}: {record['id']} {row['status']}", flush=True)
    print(f"Passed {passed}/{len(records)} ({passed / len(records):.1%})", flush=True)
