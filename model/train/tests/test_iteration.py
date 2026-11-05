"""Protect checkpoint selection and the one-shot final-test boundary."""

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from model.benchmark.src.cpu_evaluate import execution_metrics
from model.benchmark.src.task_families import FAMILIES
from model.train.src.iteration import main, run


class TestIteration:
    def results(self, directory, passed):
        directory.mkdir(parents=True)
        rows = [
            {
                "id": str(index),
                "family": FAMILIES[index % len(FAMILIES)],
                "status": "passed" if index < passed else "incorrect",
            }
            for index in range(60)
        ]
        summary = {"complete": True, **execution_metrics(rows)}
        (directory / "summary.json").write_text(json.dumps(summary))
        (directory / "responses.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        return summary

    def args(self, root):
        self.results(root / "baseline", 38)
        diagnostic = root / "diagnostic.json"
        diagnostic.write_text(json.dumps({"complete": True, "tasks": 24, "passed": 14}))
        return SimpleNamespace(
            diagnostic=diagnostic,
            output=root / "loop",
            initialize_adapter=root / "initial",
            baseline_dev=root / "baseline",
            budgets=[4, 4, 4],
            data=root / "data",
            training_data=root / "training",
            threads=1,
            device="cpu",
            learning_rate=5e-5,
        )

    def test_regression_does_not_replace_the_best_checkpoint_and_final_stays_unopened(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            initialized = []
            scores = iter([40, 39, 45])

            def experiment(options):
                initialized.append(options.initialize_adapter)
                summary = self.results(options.output / "dev", next(scores))
                return {"status": "needs_development", "development": summary}

            with (
                patch("model.train.src.iteration.experiment", side_effect=experiment),
                patch("model.train.src.iteration.verify") as verify,
                redirect_stdout(io.StringIO()),
            ):
                run(args)
            assert initialized == [
                args.initialize_adapter,
                args.output / "iteration-1-4/adapter",
                args.output / "iteration-1-4/adapter",
            ]
            verify.assert_not_called()
            state = json.loads((args.output / "loop.json").read_text())
            assert state["status"] == "needs_diagnosis"
            assert not state["objective_met"]
            assert [record["passed"] for record in state["history"]] == [40, 39, 45]
            for record in state["history"]:
                assert (
                    json.loads((Path(record["output"]) / "comparison.json").read_text())
                    == record
                )

    def test_failed_final_is_reported_and_never_followed_by_more_training(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))

            def experiment(options):
                summary = self.results(options.output / "dev", 56)
                return {"status": "complete", "development": summary}

            def verify(output, data):
                (output / "test").mkdir()
                (output / "test/verification.json").write_text(
                    json.dumps({"objective_verified": False, "replayed_passed": 90})
                )
                raise RuntimeError("Final target was not met")

            with (
                patch(
                    "model.train.src.iteration.experiment", side_effect=experiment
                ) as train,
                patch("model.train.src.iteration.verify", side_effect=verify),
                redirect_stdout(io.StringIO()),
            ):
                with pytest.raises(RuntimeError, match="Final target"):
                    run(args)
            assert train.call_count == 1
            state = json.loads((args.output / "loop.json").read_text())
            assert state["status"] == "final_failed"
            assert not state["objective_met"]

    def test_incomplete_diagnostic_is_rejected_before_training(self, tmp_path):
        args = self.args(tmp_path)
        args.diagnostic.write_text(json.dumps({"complete": False}))
        with patch("model.train.src.iteration.experiment") as train:
            with pytest.raises(ValueError, match="diagnostic did not complete"):
                run(args)
        train.assert_not_called()
        assert not args.output.exists()

    def test_retired_pid_wait_option_is_rejected(self, tmp_path, capsys):
        arguments = [
            "--training-data",
            "training",
            "--initialize-adapter",
            "initial",
            "--baseline-dev",
            "baseline",
            "--diagnostic",
            "diagnostic.json",
            "--output",
            str(tmp_path / "loop"),
            "--wait-for-pid",
            "123",
        ]
        with pytest.raises(SystemExit) as error:
            main(arguments)
        assert error.value.code == 2
        assert "unrecognized arguments: --wait-for-pid" in capsys.readouterr().err
