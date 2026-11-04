"""Protect checkpoint selection and the one-shot final-test boundary."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dataset.benchmark.src.proof_report import execution_metrics
from dataset.benchmark.src.task_types import FAMILIES
from model.train.src.iteration import run


class IterationTests(unittest.TestCase):
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
            wait_for_pid=None,
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
            self.assertEqual(
                initialized,
                [
                    args.initialize_adapter,
                    args.output / "iteration-1-4/adapter",
                    args.output / "iteration-1-4/adapter",
                ],
            )
            verify.assert_not_called()
            state = json.loads((args.output / "loop.json").read_text())
            self.assertEqual(state["status"], "needs_diagnosis")
            self.assertFalse(state["objective_met"])
            self.assertEqual(
                [record["passed"] for record in state["history"]], [40, 39, 45]
            )
            for record in state["history"]:
                self.assertEqual(
                    json.loads(
                        (Path(record["output"]) / "comparison.json").read_text()
                    ),
                    record,
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
                with self.assertRaisesRegex(RuntimeError, "Final target"):
                    run(args)
            self.assertEqual(train.call_count, 1)
            state = json.loads((args.output / "loop.json").read_text())
            self.assertEqual(state["status"], "final_failed")
            self.assertFalse(state["objective_met"])
