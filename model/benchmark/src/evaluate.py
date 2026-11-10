import json
from pathlib import Path
from contextlib import ExitStack, closing
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
    with config.output.open("x") as file, ExitStack() as stack:
        workers = []
        for _ in range(min(config.batch_size, len(records))):
            worker = Worker()
            workers.append(stack.enter_context(closing(worker)))
        remaining = iter(records)
        active = []
        for worker in workers:
            task = trial(worker, next(remaining), config.calls)
            active.append((worker, task, next(task)))
        completed = 0
        while active:
            completions = generate([prompt for _, _, prompt in active])
            pending = []
            for (worker, task, _), completion in zip(active, completions, strict=True):
                try:
                    prompt = task.send(completion)
                except StopIteration as result:
                    row = result.value
                    file.write(json.dumps(row) + "\n")
                    file.flush()
                    passed += row["status"] == "passed"
                    completed += 1
                    print(f"{completed}/{len(records)}: {row['id']} {row['status']}", flush=True)
                    record = next(remaining, None)
                    if record is None:
                        continue
                    task = trial(worker, record, config.calls)
                    prompt = next(task)
                pending.append((worker, task, prompt))
            active = pending
    print(f"Passed {passed}/{len(records)} ({passed / len(records):.1%})", flush=True)
