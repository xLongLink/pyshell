"""Check periodic evaluation reporting without running a model."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
from dataset.benchmark.src.checkpoint_eval import main


class CheckpointEvaluationTests(unittest.TestCase):
    def test_metrics_include_invalid_raw_responses(self):
        with TemporaryDirectory(dir="/tmp/opencode") as temporary:
            root = Path(temporary)
            checkpoint = root / "adapter-test/checkpoint-200"
            checkpoint.mkdir(parents=True)
            results = root / "results"
            (results / "case").mkdir(parents=True)
            summary = {
                "trials": 10,
                "passed": 2,
                "success_rate": 0.2,
                "first_text_seconds": {"median": 0.5, "p95": 1.0},
                "complete_code_seconds": {"median": 1.2, "p95": 2.0},
                "mean_calls": 3.0,
                "statuses": {"passed": 2, "budget_exceeded": 8},
            }
            (results / "summary.json").write_text(json.dumps(summary))
            events = [
                {
                    "event": "model_response",
                    "messages": [{"parts": [{"part_kind": "text", "content": source}], "provider_details": {"end_marker_seen": True}}],
                }
                for source in ("print(1)", "result = {'bad': 1))")
            ]
            (results / "case/trace.jsonl").write_text("\n".join(json.dumps(event) for event in events))
            process = MagicMock()
            process.stdout = iter([f"Results: {results}\n"])
            process.wait.return_value = 1
            history = root / "evaluations.jsonl"
            arguments = [
                "checkpoint_eval",
                "--checkpoint",
                str(checkpoint),
                "--step",
                "200",
                "--history",
                str(history),
                "--serialization",
                "raw",
            ]
            with (
                patch("sys.argv", arguments),
                patch("dataset.benchmark.src.checkpoint_eval.subprocess.run"),
                patch("dataset.benchmark.src.checkpoint_eval.subprocess.Popen", return_value=process),
            ):
                self.assertEqual(main(), 1)
            record = json.loads(history.read_text())
            self.assertEqual(record["passed"], 2)
            self.assertEqual(record["generated_responses"], 2)
            self.assertEqual(record["valid_python"], 1)
            self.assertEqual(record["raw_python_rate"], 0.5)
            self.assertEqual(record["status"], "completed")
