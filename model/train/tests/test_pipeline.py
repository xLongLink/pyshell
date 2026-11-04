"""Keep training and syntax commands on the same resolved corpus path."""

import io
import json
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from dataset.benchmark.src import checkpoint_eval
from model.train.src import pipeline


class TestPipelineData:
    def test_resume_and_initialization_cannot_be_combined(self):
        with (
            patch("model.train.src.processes.subprocess.Popen") as execute,
            redirect_stderr(io.StringIO()),
            pytest.raises(SystemExit) as raised,
        ):
            pipeline.main(["--resume", "prior", "--initialize-adapter", "initial"])
        assert raised.value.code == 2
        execute.assert_not_called()

    def test_resume_and_initialization_are_forwarded_individually(self, subtests):
        for option in ("--resume", "--initialize-adapter"):
            with (
                subtests.test(option=option),
                tempfile.TemporaryDirectory(dir="/tmp/opencode") as temporary,
                patch.object(pipeline, "ROOT", Path(temporary)),
                patch.object(pipeline, "CONVERTER", Path(__file__)),
                patch(
                    "model.train.src.processes.subprocess.Popen",
                    side_effect=lambda *args, **kwargs: SimpleNamespace(
                        stdout=io.StringIO(), wait=lambda: 0
                    ),
                ) as execute,
                redirect_stdout(io.StringIO()),
            ):
                assert pipeline.main([option, "prior"]) == 0
                command = execute.call_args_list[0].args[0]
                expected = (
                    str(Path("prior").resolve())
                    if option == "--initialize-adapter"
                    else "prior"
                )
                assert command[command.index(option) + 1] == expected

    def test_checkpoint_evaluation_exports_by_default(self):
        command = pipeline.evaluation_command("checkpoint", 200, "history.jsonl", "raw")
        assert "--gguf" not in command

    def test_default_and_explicit_data_paths_in_both_modes(self, subtests):
        repo = Path(__file__).resolve().parents[3]
        for serialization, directory in (
            ("raw", "dataset/data"),
            ("chat", "model/train/data"),
        ):
            for override in (None, "custom/../data"):
                with (
                    subtests.test(serialization=serialization, override=override),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    argv = ["--serialization", serialization]
                    if override is not None:
                        argv.extend(["--data", override])
                    expected = (
                        Path(override) if override else repo / directory
                    ).resolve()
                    with (
                        patch.object(pipeline, "ROOT", Path(temporary)),
                        patch.object(pipeline, "CONVERTER", Path(__file__)),
                        patch(
                            "model.train.src.processes.subprocess.Popen",
                            side_effect=lambda *args, **kwargs: SimpleNamespace(
                                stdout=io.StringIO(), wait=lambda: 0
                            ),
                        ) as execute,
                        redirect_stdout(io.StringIO()),
                    ):
                        assert pipeline.main(argv) == 0
                    training, export, syntax, benchmark = [
                        call.args[0] for call in execute.call_args_list
                    ]
                    assert training[training.index("--data") + 1] == str(expected)
                    assert syntax[syntax.index("--syntax-data") + 1] == str(
                        expected / "validation.jsonl"
                    )
                    gguf = export[export.index("--outfile") + 1]
                    assert syntax[syntax.index("--adapter") + 1] == gguf
                    assert benchmark[benchmark.index("--gguf") + 1] == gguf
                    assert "--syntax-only" in syntax
                    assert "--syntax-only" not in benchmark

    def test_pipeline_converts_once_and_evaluates_exported_adapter(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            summary = {
                "trials": len(checkpoint_eval.CASES),
                "passed": len(checkpoint_eval.CASES),
                "success_rate": 1.0,
                "first_text_seconds": {"median": None, "p95": None},
                "complete_code_seconds": {"median": None, "p95": None},
                "mean_calls": None,
                "statuses": {"passed": len(checkpoint_eval.CASES)},
            }
            (results / "summary.json").write_text(json.dumps(summary))
            converter = root / "converter.py"
            converter.touch()

            def execute(command, **kwargs):
                output = []
                returncode = 0
                if command[1] == str(converter):
                    Path(command[command.index("--outfile") + 1]).write_text(
                        "exported adapter"
                    )
                elif "train" in command:
                    Path(command[command.index("--output") + 1]).mkdir(parents=True)
                elif "checkpoint" in command:
                    returncode = checkpoint_eval.main(
                        command[command.index("checkpoint") + 1 :]
                    )
                elif "--syntax-only" not in command:
                    output = [f"Results: {results}\n"]
                return SimpleNamespace(
                    stdout=io.StringIO("".join(output)), wait=lambda: returncode
                )

            with (
                patch.object(pipeline, "ROOT", root),
                patch.object(pipeline, "CONVERTER", converter),
                patch("model.train.src.export.CONVERTER", converter),
                patch(
                    "model.train.src.processes.subprocess.Popen", side_effect=execute
                ) as processes,
                patch.object(checkpoint_eval.subprocess, "run") as unexpected_export,
                redirect_stdout(io.StringIO()),
            ):
                assert pipeline.main(["--name", "pilot"]) == 0
            unexpected_export.assert_not_called()
            commands = [call.args[0] for call in processes.call_args_list]
            assert sum((command[1] == str(converter) for command in commands)) == 1
            syntax, worker = [command for command in commands if "evaluate" in command]
            gguf = str(root / "artifacts/pilot.gguf")
            assert "--syntax-only" in syntax
            assert "--syntax-only" not in worker
            for command in (syntax, worker):
                assert command[command.index("--adapter") + 1] == gguf
            logs = root / "logs/pilot"
            assert json.loads((logs / "pipeline.json").read_text()) == {
                "status": "completed",
                "benchmark_returncode": 0,
            }
            for phase in ("training", "export", "syntax", "benchmark"):
                assert (logs / f"{phase}.log").is_file()
            checkpoint = root / "artifacts/pilot"
            assert (
                json.loads((checkpoint / "evaluations.jsonl").read_text())["status"]
                == "completed"
            )
            assert (
                checkpoint / "evaluation.log"
            ).read_text() == f"Results: {results}\n"
