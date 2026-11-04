"""Check periodic evaluation reporting without running a model."""

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from model.benchmark.src.cases import CASES
from model.train.src.checkpoint_eval import main


class TestCheckpointEvaluation:
    def test_metrics_include_invalid_raw_responses(self):
        with TemporaryDirectory(dir="/tmp/opencode") as temporary:
            root = Path(temporary)
            checkpoint = root / "adapter-test/checkpoint-200"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter.gguf").write_text("previous export")
            results = root / "results"
            (results / "case").mkdir(parents=True)
            summary = {
                "trials": len(CASES),
                "passed": 2,
                "success_rate": 2 / len(CASES),
                "first_text_seconds": {"median": 0.5, "p95": 1.0},
                "complete_code_seconds": {"median": 1.2, "p95": 2.0},
                "mean_calls": 3.0,
                "statuses": {"passed": 2, "budget_exceeded": len(CASES) - 2},
            }
            (results / "summary.json").write_text(json.dumps(summary))
            events = [
                {
                    "event": "model_response",
                    "messages": [
                        {
                            "parts": [{"part_kind": "text", "content": source}],
                            "provider_details": {"end_marker_seen": True},
                        }
                    ],
                }
                for source in ("print(1)", "result = {'bad': 1))")
            ]
            events.extend(
                [
                    {"event": "model_request", "messages": []},
                    {
                        "event": "model_response",
                        "messages": [
                            {
                                "parts": [
                                    {"part_kind": "thinking", "content": "not Python"},
                                    {"part_kind": "text", "content": "value = "},
                                    {"part_kind": "text", "content": "2"},
                                ],
                                "provider_details": None,
                            },
                            {"parts": []},
                        ],
                    },
                ]
            )
            (results / "case/trace.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events)
            )
            process = MagicMock()
            process.stdout = io.StringIO(f"Results: {results}\n")
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
            ]
            with (
                patch("sys.argv", arguments),
                patch("model.train.src.checkpoint_eval.subprocess.run") as export,
                patch(
                    "model.train.src.processes.subprocess.Popen", return_value=process
                ) as worker,
            ):
                assert main() == 1
            export.assert_called_once()
            command = export.call_args.args[0]
            assert command[command.index("--outfile") + 1] == str(
                checkpoint / "adapter.gguf"
            )
            assert command[-1] == str(checkpoint)
            command = worker.call_args.args[0]
            assert command[command.index("--serialization") + 1] == "raw"
            assert command[command.index("--adapter") + 1] == str(
                checkpoint / "adapter.gguf"
            )
            record = json.loads(history.read_text())
            assert record["passed"] == 2
            assert record["generated_responses"] == 4
            assert record["valid_python"] == 2
            assert record["raw_python_rate"] == 0.5
            assert record["end_markers"] == 2
            assert record["status"] == "completed"

    def test_explicit_gguf_reuses_export_and_preserves_reporting(self, subtests):
        for filename in ("checkpoint-200/adapter.gguf", "exported.gguf"):
            with (
                subtests.test(filename=filename),
                TemporaryDirectory(dir="/tmp/opencode") as temporary,
            ):
                root = Path(temporary)
                checkpoint = root / "checkpoint-200"
                checkpoint.mkdir()
                adapter = root / filename
                adapter.write_text("exported adapter")
                (checkpoint / "export.log").write_text("previous export log")
                results = root / "results"
                results.mkdir()
                summary = {
                    "trials": len(CASES),
                    "passed": len(CASES),
                    "success_rate": 1.0,
                    "first_text_seconds": {"median": 0.5, "p95": 1.0},
                    "complete_code_seconds": {"median": 1.2, "p95": 2.0},
                    "mean_calls": 3.0,
                    "statuses": {"passed": len(CASES)},
                }
                (results / "summary.json").write_text(json.dumps(summary))
                process = MagicMock()
                process.stdout = io.StringIO(f"Results: {results}\n")
                process.wait.return_value = 0
                history = root / "evaluations.jsonl"
                with (
                    patch("model.train.src.checkpoint_eval.subprocess.run") as export,
                    patch(
                        "model.train.src.processes.subprocess.Popen",
                        return_value=process,
                    ) as worker,
                    patch("builtins.print") as output,
                ):
                    assert (
                        main(
                            [
                                "--checkpoint",
                                str(checkpoint),
                                "--step",
                                "200",
                                "--history",
                                str(history),
                                "--gguf",
                                str(adapter),
                            ]
                        )
                        == 0
                    )
                export.assert_not_called()
                command = worker.call_args.args[0]
                assert command[command.index("--adapter") + 1] == str(adapter)
                assert "--syntax-only" not in command
                assert adapter.read_text() == "exported adapter"
                assert (checkpoint / "export.log").read_text() == "previous export log"
                assert (
                    checkpoint / "evaluation.log"
                ).read_text() == f"Results: {results}\n"
                output.assert_any_call(f"Results: {results}\n", end="", flush=True)
                record = json.loads(history.read_text())
                assert record["status"] == "completed"
                assert record["results"] == str(results)
                assert record["passed"] == len(CASES)
                assert record["generated_responses"] == 0
                assert record["valid_python"] == 0
                assert record["end_markers"] == 0
                assert record["raw_python_rate"] is None

    def test_explicit_gguf_must_be_a_file(self, subtests):
        with TemporaryDirectory(dir="/tmp/opencode") as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint-200"
            checkpoint.mkdir()
            (checkpoint / "adapter.gguf").touch()
            history = root / "history/evaluations.jsonl"
            for adapter in (root / "missing.gguf", checkpoint):
                with (
                    subtests.test(adapter=adapter),
                    patch("model.train.src.checkpoint_eval.subprocess.run") as export,
                    patch("model.train.src.processes.subprocess.Popen") as worker,
                ):
                    assert (
                        main(
                            [
                                "--checkpoint",
                                str(checkpoint),
                                "--step",
                                "200",
                                "--history",
                                str(history),
                                "--gguf",
                                str(adapter),
                            ]
                        )
                        == 2
                    )
                    export.assert_not_called()
                    worker.assert_not_called()
                records = [
                    json.loads(line) for line in history.read_text().splitlines()
                ]
                assert records[-1]["status"] == "evaluation_error"
                assert records[-1]["checkpoint"] == str(checkpoint)
                assert records[-1]["step"] == 200
                assert "FileNotFoundError" in records[-1]["error"]
                assert str(adapter) in records[-1]["error"]
            assert len(records) == 2
            assert not (checkpoint / "export.log").exists()
            assert not (checkpoint / "evaluation.log").exists()

    def test_completion_uses_suite_size_and_success_requires_all_passing(
        self, subtests
    ):
        for trials, passed, status, exit_code in (
            (3, 3, "completed", 0),
            (3, 2, "completed", 1),
            (2, 2, "partial", 1),
            (10, 10, "partial", 1),
        ):
            with (
                subtests.test(trials=trials, passed=passed),
                TemporaryDirectory(dir="/tmp/opencode") as temporary,
            ):
                root = Path(temporary)
                checkpoint = root / "checkpoint-200"
                checkpoint.mkdir()
                results = root / "results"
                results.mkdir()
                summary = {
                    "trials": trials,
                    "passed": passed,
                    "success_rate": passed / trials,
                    "first_text_seconds": {"median": None, "p95": None},
                    "complete_code_seconds": {"median": None, "p95": None},
                    "mean_calls": None,
                    "statuses": {"passed": passed, "incorrect": trials - passed},
                }
                (results / "summary.json").write_text(json.dumps(summary))
                process = MagicMock()
                process.stdout = io.StringIO(f"Results: {results}\n")
                process.wait.return_value = 0 if passed == trials else 1
                history = root / "evaluations.jsonl"
                arguments = [
                    "--checkpoint",
                    str(checkpoint),
                    "--step",
                    "200",
                    "--history",
                    str(history),
                ]
                with (
                    patch("model.train.src.checkpoint_eval.CASES", CASES[:3]),
                    patch("model.train.src.checkpoint_eval.subprocess.run"),
                    patch(
                        "model.train.src.processes.subprocess.Popen",
                        return_value=process,
                    ),
                ):
                    assert main(arguments) == exit_code
                record = json.loads(history.read_text())
                assert record["status"] == status
                assert record["cases"] == trials
                assert record["passed"] == passed
