"""Protect final replay provenance and report failures without model weights."""

import hashlib
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest
from model.train.src.verification import verify


class ReplayWorkspace:
    def __init__(self):
        self.closed = False

    def execute(self, source, action="execute"):
        if action == "reset":
            self.namespace = {}
        result = {"exception": None, "done": None}
        self.namespace["done"] = lambda text: result.update(done=text)
        try:
            exec(source, self.namespace)
        except Exception as error:
            result.update(exception=type(error).__name__, done=None)
        return result

    def close(self):
        self.closed = True


class TestVerification:
    def artifacts(self, root, directory_adapter=True):
        experiment, data = root / "experiment", root / "data"
        (experiment / "test").mkdir(parents=True)
        data.mkdir()
        adapter = root / "adapter"
        if directory_adapter:
            adapter.mkdir()
            weights = adapter / "adapter_model.safetensors"
        else:
            weights = adapter
        weights.write_bytes(b"fixed adapter")
        tasks = [
            {
                "id": f"task-{index}",
                "setup": f"value = {index}",
                "check": "assert result == value + 1",
                "followup": None,
            }
            for index in range(120)
        ]
        predictions = [
            {
                "id": task["id"],
                "status": "passed",
                "steps": [
                    {
                        "stage": 0,
                        "end_marker_seen": True,
                        "source": "result = value + 1\ndone('ok')",
                    }
                ],
            }
            for task in tasks
        ]
        task_path = data / "test-tasks.jsonl"
        task_path.write_text("".join(json.dumps(task) + "\n" for task in tasks))
        digest = hashlib.sha256(task_path.read_bytes()).hexdigest()
        (data / "manifest.json").write_text(
            json.dumps({"splits": {"test": {"sha256": digest}}})
        )
        (experiment / "test/responses.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in predictions)
        )
        (experiment / "experiment.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "development": {
                        "complete": True,
                        "tasks": 60,
                        "success_rate": 1.0,
                        "by_family": {"synthetic": {"tasks": 60, "passed": 60}},
                    },
                }
            )
        )
        (experiment / "test/summary.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "passed": 120,
                    "adapter": str(adapter),
                    "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                    "task_file_sha256": digest,
                    "backend": "hf",
                }
            )
        )
        return experiment, data, weights

    def test_valid_artifacts_replay_with_file_and_directory_adapters(self, subtests):
        for directory_adapter in (False, True):
            with (
                subtests.test(directory_adapter=directory_adapter),
                tempfile.TemporaryDirectory() as temporary,
            ):
                experiment, data, _ = self.artifacts(Path(temporary), directory_adapter)
                worker = ReplayWorkspace()
                with (
                    patch("model.train.src.verification.Worker", return_value=worker),
                    redirect_stdout(io.StringIO()),
                ):
                    verify(experiment, data)
                report = json.loads((experiment / "test/verification.json").read_text())
                assert report["objective_verified"]
                assert report["replayed_passed"] == 120
                assert report["status_mismatches"] == []
                assert worker.closed

    def test_worker_closes_and_preserves_replay_exceptions(self, subtests):
        for error in (TimeoutError("worker unavailable"), KeyboardInterrupt()):
            with (
                subtests.test(error=type(error).__name__),
                tempfile.TemporaryDirectory(dir="/tmp/opencode") as temporary,
            ):
                experiment, data, _ = self.artifacts(Path(temporary))
                worker = ReplayWorkspace()
                with (
                    patch("model.train.src.verification.Worker", return_value=worker),
                    patch.object(worker, "execute", side_effect=error),
                    pytest.raises(type(error)) as raised,
                ):
                    verify(experiment, data)
                assert raised.value is error
                assert worker.closed
                assert not (experiment / "test/verification.json").exists()

    def test_changed_tasks_or_weights_are_rejected_before_execution(self, subtests):
        for changed in ("tasks", "weights"):
            with (
                subtests.test(changed=changed),
                tempfile.TemporaryDirectory() as temporary,
            ):
                experiment, data, weights = self.artifacts(Path(temporary))
                path = weights if changed == "weights" else data / "test-tasks.jsonl"
                path.write_bytes(path.read_bytes() + b"\n")
                with patch(
                    "model.train.src.verification.Worker",
                    side_effect=AssertionError("must not execute"),
                ):
                    with pytest.raises(
                        ValueError, match="Adapter differs|Final tasks differ"
                    ):
                        verify(experiment, data)

    def test_predictions_must_cover_tasks_once_and_in_order(self, subtests):
        for invalid in ("missing", "duplicate", "reordered"):
            with (
                subtests.test(invalid=invalid),
                tempfile.TemporaryDirectory() as temporary,
            ):
                experiment, data, _ = self.artifacts(Path(temporary))
                path = experiment / "test/responses.jsonl"
                rows = path.read_text().splitlines()
                if invalid == "missing":
                    rows.pop()
                elif invalid == "duplicate":
                    rows[1] = rows[0]
                else:
                    rows.reverse()
                path.write_text("\n".join(rows) + "\n")
                with patch(
                    "model.train.src.verification.Worker",
                    side_effect=AssertionError("must not execute"),
                ):
                    with pytest.raises(ValueError, match="exactly once"):
                        verify(experiment, data)

    def test_blank_jsonl_records_are_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment, data, _ = self.artifacts(Path(temporary))
            path = experiment / "test/responses.jsonl"
            path.write_text("\n" + path.read_text())
            with patch(
                "model.train.src.verification.Worker",
                side_effect=AssertionError("must not execute"),
            ):
                with pytest.raises(json.JSONDecodeError):
                    verify(experiment, data)

    def test_failed_replay_publishes_mismatch_report_before_raising(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment, data, _ = self.artifacts(Path(temporary))
            path = experiment / "test/responses.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["steps"][0]["source"] = (
                "result = value + 2\ndone('claimed success')"
            )
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            worker = ReplayWorkspace()
            with (
                patch("model.train.src.verification.Worker", return_value=worker),
                redirect_stdout(io.StringIO()),
            ):
                with pytest.raises(RuntimeError, match="independent replay"):
                    verify(experiment, data)
            report = json.loads((experiment / "test/verification.json").read_text())
            assert not report["objective_verified"]
            assert report["replayed_passed"] == 119
            assert report["status_mismatches"] == [
                {"id": "task-0", "original": "passed", "replay": "incorrect"}
            ]
            assert worker.closed
