import pytest
from model.benchmark.src.trial import trial


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

    row = trial(worker, record, generate, 4)
    assert row["status"] == "passed"
    assert "ValueError: inspect values first" in prompts[1]
    for prompt in prompts:
        assert all(record[field] not in prompt for field in ("setup", "check", "code"))
    assert worker.sources[0] == (record["setup"], "reset")
    assert worker.sources[-1] == (record["check"], "execute")


def test_done_does_not_bypass_checks(record):
    row = trial(Worker(), record, lambda prompt: ("result = 999\ndone('ok')", True), 4)
    assert row["status"] == "incorrect"
    assert "AssertionError" in row["error"]


@pytest.mark.parametrize(("source", "ended"), [("result = 11", False), ("", True), ("  \n", True)])
def test_invalid_completion_is_not_executed(record, source, ended):
    worker = Worker()
    row = trial(worker, record, lambda prompt: (source, ended), 4)
    assert row["status"] == "boundary_error"
    assert len(worker.sources) == 1


def test_call_limit_is_enforced(record):
    row = trial(Worker(), record, lambda prompt: ("result = 11", True), 2)
    assert row["status"] == "call_limit"
    assert len(row["steps"]) == 2


def test_invalid_fixture_is_an_error(record):
    record["setup"] = "raise ValueError('bad fixture')"
    with pytest.raises(RuntimeError, match="Invalid fixture sum"):
        trial(Worker(), record, lambda prompt: ("done('ok')", True), 4)


def test_followup_preserves_workspace_and_has_its_own_budget(record):
    record["followup"] = {"instruction": "Add one to the previous result.", "check": "assert saved_result == 11\nassert result == 12"}
    scripts = iter(["result = sum(values)", "done('ok')", "result = saved_result + 1", "done('ok')"])
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return next(scripts), True

    worker = Worker()
    row = trial(worker, record, generate, 2)
    assert row["status"] == "passed"
    assert len(row["steps"]) == 4
    assert sum(action == "reset" for _, action in worker.sources) == 1
    assert prompts[2].endswith("<|task|>\nAdd one to the previous result.\n<|code|>\n")
    assert record["followup"]["check"] not in prompts[2]
