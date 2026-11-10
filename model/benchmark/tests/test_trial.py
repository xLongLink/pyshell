import json
import pytest
from model.benchmark.src import trial, evaluate, inference
from model.benchmark.src.settings import Settings


def run(worker, record, generate, calls):
    task = trial.trial(worker, record, calls)
    prompt = next(task)
    while True:
        completion = generate(prompt)
        try:
            prompt = task.send(completion)
        except StopIteration as result:
            return result.value


class Worker:
    def __init__(self):
        self.namespace = {}
        self.sources = []

    def execute(self, source, action="execute"):
        self.sources.append((source, action))
        feedback = {"stdout": "", "stderr": "", "exception": None, "done": None}
        if action == "reset":
            self.namespace = {}

        def done(value):
            feedback["done"] = value

        self.namespace["done"] = done
        try:
            exec(source, self.namespace)
        except Exception as error:
            feedback.update(exception=f"{type(error).__name__}: {error}", done=None)
        return feedback

    def close(self):
        pass


@pytest.fixture
def record():
    return {
        "id": "sum",
        "family": "sum",
        "setup": "values = [4, 7]",
        "instruction": "Sum existing values into result.",
        "code": "REFERENCE_CODE_MUST_NOT_LEAK",
        "check": "assert result == 11",
        "followup": None,
    }


def test_feedback_is_returned_without_leaking_setup_or_checks(record):
    worker = Worker()
    prompts = []
    scripts = iter(["raise ValueError('inspect values first')", "result = sum(values)\ndone('ok')"])

    def generate(prompt):
        prompts.append(prompt)
        return next(scripts), True

    row = run(worker, record, generate, 4)
    assert row["status"] == "passed"
    assert "ValueError: inspect values first" in prompts[1]
    for prompt in prompts:
        assert all(record[field] not in prompt for field in ("setup", "check", "code"))
    assert worker.sources[0] == (record["setup"], "reset")
    assert worker.sources[-1] == (record["check"], "execute")


def test_done_does_not_bypass_checks(record):
    row = run(Worker(), record, lambda prompt: ("result = 999\ndone('ok')", True), 4)
    assert row["status"] == "incorrect"
    assert "AssertionError" in row["error"]


@pytest.mark.parametrize(("source", "ended"), [("result = 11", False), ("", True), ("  \n", True)])
def test_invalid_completion_is_not_executed(record, source, ended):
    worker = Worker()
    row = run(worker, record, lambda prompt: (source, ended), 4)
    assert row["status"] == "boundary_error"
    assert len(worker.sources) == 1


def test_call_limit_is_enforced(record):
    row = run(Worker(), record, lambda prompt: ("result = 11", True), 2)
    assert row["status"] == "call_limit"
    assert len(row["steps"]) == 2


def test_invalid_fixture_is_an_error(record):
    record["setup"] = "raise ValueError('bad fixture')"
    with pytest.raises(RuntimeError, match="Invalid fixture sum"):
        run(Worker(), record, lambda prompt: ("done('ok')", True), 4)


def test_followup_preserves_workspace_and_has_its_own_budget(record):
    record["followup"] = {"instruction": "Add one to the previous result.", "check": "assert saved_result == 11\nassert result == 12"}
    scripts = iter(["result = sum(values)", "done('ok')", "result = saved_result + 1", "done('ok')"])
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return next(scripts), True

    worker = Worker()
    row = run(worker, record, generate, 2)
    assert row["status"] == "passed"
    assert len(row["steps"]) == 4
    assert sum(action == "reset" for _, action in worker.sources) == 1
    assert prompts[2].endswith("<|task|>\nAdd one to the previous result.\n<|code|>\n")
    assert record["followup"]["check"] not in prompts[2]


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batch_preserves_independent_workspaces_retries_and_followups(record, tmp_path, monkeypatch, batch_size):
    records = [
        {
            **record,
            "id": str(index),
            "setup": f"values = [{index}, 10]",
            "check": f"assert result == {index + 10}",
            "followup": {"instruction": "Add one.", "check": f"assert saved_result == {index + 10}\nassert result == {index + 11}"},
        }
        for index in range(5)
    ]
    (tmp_path / "dev-tasks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))

    def generate(prompts):
        completions = []
        for prompt in prompts:
            if "Add one." in prompt:
                source = "result = saved_result + 1\ndone('ok')"
            elif "print(values)" in prompt:
                source = "result = sum(values)\ndone('ok')"
            else:
                source = "print(values)"
            completions.append((source, True))
        return completions

    monkeypatch.setattr(evaluate, "Worker", Worker)
    monkeypatch.setattr(inference, "HFCompletion", lambda config: generate)
    config = Settings(data=str(tmp_path), output=tmp_path / "results.jsonl", batch_size=batch_size)
    evaluate.run(config)
    rows = [json.loads(line) for line in config.output.read_text().splitlines()]
    assert {row["id"] for row in rows} == {row["id"] for row in records}
    assert all(row["status"] == "passed" and len(row["steps"]) == 3 for row in rows)
