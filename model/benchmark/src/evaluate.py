import json
from pathlib import Path
from contextlib import closing
from huggingface_hub import hf_hub_download
from model.benchmark.src.trial import trial
from model.benchmark.src.execution import Worker


def run(config):
    if config.output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark results: {config.output}")
    path = Path(config.data) / "dev-tasks.jsonl"
    if not path.parent.is_dir():
        path = Path(
            hf_hub_download(
                config.data,
                "experiments/executed-transfer-v1/evaluation/dev-tasks.jsonl",
                repo_type="dataset",
                revision=config.data_revision,
            )
        )
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
