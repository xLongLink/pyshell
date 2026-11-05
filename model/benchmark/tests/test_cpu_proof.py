"""Keep the CPU experiment's grading independent of model-visible context."""

import io
import sys
import json
import pytest
import hashlib
import tempfile
import subprocess
from types import SimpleNamespace
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import patch
from model.benchmark.src import cpu_evaluate
from pyshell.serialization import CODE, TASK, CONSOLE, END_CODE
from model.benchmark.src.resources import exclusive_cpu
from model.benchmark.src.cpu_evaluate import trial, evaluate, generator, completion


class Worker:
    def __init__(self, fail_check=False):
        self.sources = []
        self.fail_check = fail_check

    def execute(self, source, action="execute"):
        self.sources.append((source, action))
        return {
            "stdout": "[4, 7]" if source == "print(values)" else "",
            "stderr": "",
            "exception": "AssertionError" if self.fail_check and source.startswith("assert") else None,
            "done": "ok" if "done(" in source else None,
        }


class TestCPUProof:
    def test_evaluation_persists_full_trajectories_and_incremental_metrics(self, tmp_path, capsys):
        records = [{"id": str(index), "family": family} for index, family in enumerate(("join", "recovery", "join"))]
        tasks = tmp_path / "dev-tasks.jsonl"
        tasks.write_text("".join(json.dumps(record) + "\n" for record in records))
        digest = hashlib.sha256(tasks.read_bytes()).hexdigest()
        (tmp_path / "manifest.json").write_text(json.dumps({"scope": "fixture", "splits": {"dev": {"sha256": digest}}}))
        args = SimpleNamespace(
            data=tmp_path,
            split="dev",
            final=False,
            limit=None,
            backend="gguf",
            device="cpu",
            adapter=None,
            output=tmp_path / "output",
            label="fixture",
            gguf_base=tmp_path / "base.gguf",
        )
        rows = [
            {**record, "status": status, "steps": [{"source": "print('payload')", "feedback": {"stdout": "payload"}}]}
            for record, status in zip(records, ("passed", "call_limit", "incorrect"))
        ]
        summaries = []

        def run_trial(worker, record, generate):
            if record["id"] != "0":
                summaries.append(json.loads((args.output / "summary.json").read_text()))
            return rows[int(record["id"])]

        with (
            patch.object(cpu_evaluate, "exclusive_cpu", return_value=nullcontext()),
            patch.object(cpu_evaluate, "generator", return_value=nullcontext(None)),
            patch.object(cpu_evaluate, "Worker"),
            patch.object(cpu_evaluate, "trial", side_effect=run_trial),
        ):
            evaluate(args)
        summaries.append(json.loads((args.output / "summary.json").read_text()))
        assert [json.loads(line) for line in (args.output / "responses.jsonl").read_text().splitlines()] == rows
        assert [summary["tasks"] for summary in summaries] == [1, 2, 3]
        assert [summary["passed"] for summary in summaries] == [1, 1, 1]
        assert [summary["complete"] for summary in summaries] == [False, False, True]
        assert all(not summary["acceptance_met"] for summary in summaries)
        summary = summaries[-1]
        assert summary["task_file_sha256"] == digest
        assert summary["success_rate"] == 1 / 3
        assert summary["by_family"] == {"join": {"tasks": 2, "passed": 1}, "recovery": {"tasks": 1, "passed": 0}}
        assert summary["statuses"] == {"passed": 1, "call_limit": 1, "incorrect": 1}
        assert "fixture dev 3/3: 1 passed; join=incorrect" in capsys.readouterr().out

    def test_hf_generator_passes_the_explicit_base_model(self, tmp_path):
        args = SimpleNamespace(backend="hf", adapter=tmp_path / "adapter", threads=3, device="cpu", base_model=tmp_path / "base")
        with patch("model.benchmark.src.inference.HFCompletion") as model:
            with generator(args) as generate:
                assert generate is model.return_value
        model.assert_called_once_with(args.adapter, 3, "cpu", base_model=args.base_model)

    def test_completion_preserves_stop_order_and_end_marker_requirement(self, subtests):
        for stopping_word in (END_CODE, TASK, CONSOLE, CODE, ""):
            with subtests.test(stopping_word=stopping_word):
                response = io.StringIO(json.dumps({"content": "result = 11", "stopping_word": stopping_word}))
                with patch("model.benchmark.src.cpu_evaluate.urllib.request.urlopen", return_value=response) as urlopen:
                    assert completion("Sum values") == ("result = 11", stopping_word == END_CODE)
                payload = json.loads(urlopen.call_args.args[0].data)
                assert payload["stop"] == [END_CODE, TASK, CONSOLE, CODE]
                assert payload["prompt"] == "Sum values"

    def record(self):
        return {
            "id": "test",
            "family": "sum",
            "setup": "values = [4, 7]",
            "instruction": "Sum existing values into result.",
            "code": "REFERENCE_CODE_MUST_NOT_LEAK",
            "inspect": "REFERENCE_INSPECTION_MUST_NOT_LEAK",
            "check": "assert result == 11",
            "followup": None,
        }

    def test_setup_and_oracle_are_not_model_context(self):
        worker = Worker()
        prompts = []
        scripts = iter(["print(values)", "result = sum(values)\ndone(str(result))"])

        def complete(prompt):
            prompts.append(prompt)
            return next(scripts), True

        record = self.record()
        result = trial(worker, record, complete)
        assert result["status"] == "passed"
        for prompt in prompts:
            for hidden in (record["setup"], record["check"], record["code"], record["inspect"]):
                assert hidden not in prompt
        assert "[4, 7]" in prompts[1]
        assert worker.sources[0] == (record["setup"], "reset")
        assert worker.sources[-1][0] == record["check"]

    def test_done_claim_does_not_bypass_oracle(self):
        result = trial(Worker(fail_check=True), self.record(), lambda prompt: ("result = 999\ndone('success')", True))
        assert result["status"] == "incorrect"

    def test_missing_end_marker_is_not_executed(self):
        worker = Worker()
        result = trial(worker, self.record(), lambda prompt: ("result = 11", False))
        assert result["status"] == "boundary_error"
        assert len(worker.sources) == 1

    def test_followup_is_a_new_task_in_the_same_workspace(self):
        record = self.record()
        record["followup"] = {"instruction": "Append 9 in place.", "check": "assert result[-1] == 9"}
        prompts = []

        def complete(prompt):
            prompts.append(prompt)
            return "done('ok')", True

        worker = Worker()
        result = trial(worker, record, complete)
        assert result["status"] == "passed"
        assert sum((action == "reset" for _, action in worker.sources)) == 1
        assert "Append 9 in place." in prompts[1]
        assert "assert result[-1]" not in prompts[1]

    def final_request(self, directory, *, final=False, limit=None):
        root = Path(directory)
        tasks = root / "test-tasks.jsonl"
        tasks.write_text("{}\n")
        manifest = {"splits": {"test": {"sha256": hashlib.sha256(tasks.read_bytes()).hexdigest()}}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return SimpleNamespace(data=root, split="test", final=final, limit=limit)

    def test_final_requires_explicit_authorization_before_loading_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.final_request(directory)
            with pytest.raises(ValueError, match="requires --final"):
                evaluate(args)

    def test_final_cannot_be_reduced_to_a_selected_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.final_request(directory, final=True, limit=10)
            with pytest.raises(ValueError, match="cannot be reduced"):
                evaluate(args)

    def test_changed_frozen_tasks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.final_request(directory, final=True)
            (args.data / "test-tasks.jsonl").write_text('{"modified":true}\n')
            with pytest.raises(ValueError, match="tasks have changed"):
                evaluate(args)

    def test_cpu_experiments_wait_for_each_other_without_polling(self):
        source = "from model.benchmark.src.resources import exclusive_cpu\nwith exclusive_cpu(): print('acquired', flush=True)"
        with exclusive_cpu():
            child = subprocess.Popen([sys.executable, "-c", source], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                assert "Waiting for" in child.stdout.readline()
                assert child.poll() is None
            except BaseException:
                child.kill()
                child.communicate()
                raise
        output, error = child.communicate(timeout=10)
        assert child.returncode == 0, error
        assert "acquired" in output
