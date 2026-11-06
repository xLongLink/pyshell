import json
import pytest
import hashlib
from types import SimpleNamespace
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import patch
from model.benchmark.src import evaluate
from model.benchmark.src.proof_trial import trial


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


class TestEvaluation:
    def test_cli_owns_the_inference_defaults(self, tmp_path):
        with patch.object(evaluate, "evaluate") as run_evaluation:
            evaluate.main(["--label", "fixture", "--output", str(tmp_path / "output")])
        args = run_evaluation.call_args.args[0]
        assert args.adapter is None
        assert args.device == "cpu"
        assert args.threads == 16
        assert args.base_model == Path(__file__).resolve().parents[1] / "local/base"

    @pytest.mark.parametrize(
        "adapter_form,mutate_adapter", [(None, False), ("file", False), ("directory", False), ("file", True), ("directory", True)]
    )
    def test_evaluation_persists_full_trajectories_and_incremental_metrics(self, tmp_path, capsys, adapter_form, mutate_adapter):
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
            device="cpu",
            threads=3,
            base_model=tmp_path / "base",
            adapter=None,
            output=tmp_path / "output",
            label="fixture",
        )
        weights = None
        if adapter_form:
            args.adapter = tmp_path / "adapter"
            if adapter_form == "directory":
                args.adapter.mkdir()
                weights = args.adapter / "adapter_model.safetensors"
            else:
                weights = args.adapter
            weights.write_bytes(b"fixed weights")
        adapter_digest = hashlib.sha256(b"fixed weights").hexdigest() if weights else None
        rows = [
            {**record, "status": status, "steps": [{"source": "print('payload')", "feedback": {"stdout": "payload"}}]}
            for record, status in zip(records, ("passed", "call_limit", "incorrect"))
        ]
        summaries = []

        def run_trial(worker, record, generate):
            if record["id"] != "0":
                summaries.append(json.loads((args.output / "summary.json").read_text()))
                assert [json.loads(line) for line in (args.output / "responses.jsonl").read_text().splitlines()] == rows[
                    : int(record["id"])
                ]
            if mutate_adapter and record["id"] == "2":
                weights.write_bytes(b"changed weights")
            return rows[int(record["id"])]

        with (
            patch("model.benchmark.src.inference.HFCompletion") as model,
            patch.object(evaluate, "Worker"),
            patch.object(evaluate, "trial", side_effect=run_trial),
        ):
            model.return_value.runtime = {"fixture": True}
            with pytest.raises(RuntimeError, match="Adapter changed during evaluation") if mutate_adapter else nullcontext():
                evaluate.evaluate(args)
        model.assert_called_once_with(args.adapter, 3, "cpu", base_model=args.base_model)
        summaries.append(json.loads((args.output / "summary.json").read_text()))
        assert [json.loads(line) for line in (args.output / "responses.jsonl").read_text().splitlines()] == rows
        assert [summary["tasks"] for summary in summaries] == [1, 2, 3]
        assert [summary["passed"] for summary in summaries] == [1, 1, 1]
        assert [summary["complete"] for summary in summaries] == [False, False, True]
        assert [summary["statuses"] for summary in summaries] == [
            {"passed": 1},
            {"passed": 1, "call_limit": 1},
            {"passed": 1, "call_limit": 1, "incorrect": 1},
        ]
        assert all(not summary["acceptance_met"] for summary in summaries)
        summary = summaries[-1]
        assert summary["task_file_sha256"] == digest
        assert summary["adapter"] == (str(args.adapter) if args.adapter else None)
        assert summary["adapter_sha256"] == adapter_digest
        assert summary["backend"] == "hf"
        assert summary["dtype"] == "float32"
        assert summary["runtime"] == {"fixture": True}
        assert summary["success_rate"] == 1 / 3
        assert summary["by_family"] == {"join": {"tasks": 2, "passed": 1}, "recovery": {"tasks": 1, "passed": 0}}
        assert summary["statuses"] == {"passed": 1, "call_limit": 1, "incorrect": 1}
        assert "fixture dev 3/3: 1 passed; join=incorrect" in capsys.readouterr().out

    @pytest.mark.parametrize("fail_at", [0, 1])
    def test_trial_failure_closes_the_response_log_and_preserves_completed_rows(self, tmp_path, fail_at):
        records = [{"id": str(index), "family": "join"} for index in range(2)]
        args = SimpleNamespace(
            data=tmp_path,
            split="dev",
            final=False,
            limit=None,
            adapter=None,
            output=tmp_path / "output",
            threads=1,
            device="cpu",
            base_model=tmp_path / "base",
            label="fixture",
        )
        failure = RuntimeError("trial failed")
        handles = []
        path_open = Path.open

        def open_log(path, *arguments, **kwargs):
            file = path_open(path, *arguments, **kwargs)
            if path == args.output / "responses.jsonl" and arguments == ("a",):
                handles.append(file)
            return file

        def run_trial(worker, record, generate):
            if int(record["id"]) == fail_at:
                raise failure
            return {**record, "status": "passed", "steps": []}

        with (
            patch.object(evaluate, "load_tasks", return_value=({"scope": "fixture"}, "digest", records)),
            patch("model.benchmark.src.inference.HFCompletion") as model,
            patch.object(evaluate, "Worker") as worker,
            patch.object(evaluate, "trial", side_effect=run_trial),
            patch.object(Path, "open", open_log),
            pytest.raises(RuntimeError) as error,
        ):
            model.return_value.runtime = {}
            evaluate.evaluate(args)
        assert error.value is failure
        assert len(handles) == 1
        assert handles[0].closed
        worker.return_value.close.assert_called_once_with()
        rows = [json.loads(line) for line in (args.output / "responses.jsonl").read_text().splitlines()]
        assert rows == [{**record, "status": "passed", "steps": []} for record in records[:fail_at]]
        if fail_at:
            summary = json.loads((args.output / "summary.json").read_text())
            assert summary["tasks"] == summary["passed"] == fail_at
            assert not summary["complete"]
        else:
            assert not (args.output / "summary.json").exists()

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

    def test_structured_feedback_and_followup_keep_distinct_boundaries(self):
        record = self.record()
        record["followup"] = {"instruction": "Append 9 in place.", "check": "assert result[-1] == 9"}
        scripts = iter(["print(values)", "result = sum(values)\ndone(str(result))", "done('ok')"])
        prompts = []

        def complete(prompt):
            prompts.append(prompt)
            return next(scripts), True

        result = trial(Worker(), record, complete)
        assert result["status"] == "passed"
        assert len(prompts) == 3
        assert "[4, 7]" in prompts[1]
        assert prompts[-1].endswith("<|task|>\nAppend 9 in place.\n<|code|>\n")
        assert all(record["check"] not in prompt and record["followup"]["check"] not in prompt for prompt in prompts)

    def final_request(self, directory, *, final=False, limit=None):
        root = Path(directory)
        tasks = root / "test-tasks.jsonl"
        tasks.write_text("{}\n")
        manifest = {"splits": {"test": {"sha256": hashlib.sha256(tasks.read_bytes()).hexdigest()}}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return SimpleNamespace(data=root, split="test", final=final, limit=limit)

    def test_final_requires_explicit_authorization_before_loading_a_model(self, tmp_path):
        args = self.final_request(tmp_path)
        with pytest.raises(ValueError, match="requires --final"):
            evaluate.evaluate(args)

    def test_final_cannot_be_reduced_to_a_selected_subset(self, tmp_path):
        args = self.final_request(tmp_path, final=True, limit=10)
        with pytest.raises(ValueError, match="cannot be reduced"):
            evaluate.evaluate(args)

    def test_changed_frozen_tasks_are_rejected(self, tmp_path):
        args = self.final_request(tmp_path, final=True)
        (args.data / "test-tasks.jsonl").write_text('{"modified":true}\n')
        with pytest.raises(ValueError, match="tasks have changed"):
            evaluate.evaluate(args)
