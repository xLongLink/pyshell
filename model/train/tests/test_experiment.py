"""Check experiment phase selection without training or opening a real holdout."""

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from model.benchmark.src.corpus import sha256
from model.benchmark.src.task_families import FAMILIES
from model.train.src.experiment import main, run


class TestExperiment:
    def experiment(
        self, directory, ready, device="cpu", initialize_adapter="initial", curriculum=False
    ):
        root = Path(directory)
        data = root / "data"
        data.mkdir()
        manifest = {"splits": {}}
        for split in ("train", "dev", "test"):
            path = data / f"{split}-tasks.jsonl"
            path.write_text('{"fixture":"synthetic"}\n')
            manifest["splits"][split] = {"sha256": sha256(path)}
        for name, key in (
            ("train.jsonl", "training_records_sha256"),
            ("validation.jsonl", "validation_records_sha256"),
        ):
            path = data / name
            path.write_text('{"id":"synthetic"}\n')
            manifest[key] = sha256(path)
        (data / "manifest.json").write_text(json.dumps(manifest))
        training_data = None
        if curriculum:
            training_data = root / "curriculum"
            training_data.mkdir()
            training_manifest = {
                "evaluation_test_sha256": manifest["splits"]["test"]["sha256"]
            }
            for name, key in (
                ("train.jsonl", "training_records_sha256"),
                ("validation.jsonl", "validation_records_sha256"),
            ):
                path = training_data / name
                path.write_text('{"id":"curriculum"}\n')
                training_manifest[key] = sha256(path)
            (training_data / "manifest.json").write_text(json.dumps(training_manifest))
        args = SimpleNamespace(
            data=data,
            output=root / "run",
            steps=4,
            threads=1,
            device=device,
            learning_rate=1e-4,
            training_data=training_data,
            initialize_adapter=root / initialize_adapter
            if initialize_adapter is not None
            else None,
            final_if_ready=True,
        )
        calls = []

        def execute(command, **kwargs):
            calls.append(command)
            if "model.benchmark.main" not in command:
                return
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            passed_per_family = 5 if ready else 3
            summary = {
                "complete": True,
                "tasks": 60,
                "success_rate": passed_per_family / 5,
                "by_family": {
                    family: {"tasks": 5, "passed": passed_per_family}
                    for family in FAMILIES
                },
                "acceptance_met": ready,
            }
            (output / "summary.json").write_text(json.dumps(summary))

        with (
            patch.object(
                Path, "read_text", autospec=True, side_effect=Path.read_text
            ) as reads,
            patch("model.train.src.experiment.subprocess.run", side_effect=execute),
            redirect_stdout(io.StringIO()) as stdout,
        ):
            metadata = run(args)
        manifests_read = [
            call.args[0]
            for call in reads.call_args_list
            if call.args[0].name == "manifest.json"
        ]
        assert manifests_read == [data / "manifest.json"] + (
            [training_data / "manifest.json"] if curriculum else []
        )
        assert metadata["training_manifest"] == (
            training_manifest if curriculum else manifest
        )
        serialized = json.dumps(metadata, indent=2) + "\n"
        assert (args.output / "experiment.json").read_text() == serialized
        assert stdout.getvalue().endswith(serialized)
        return args.output, calls

    def test_main_does_not_return_experiment_metadata(self):
        with patch(
            "model.train.src.experiment.run", return_value={"status": "complete"}
        ) as experiment:
            assert main(["--output", "run"]) is None
        assert experiment.call_args.args[0].output == Path("run")

    def test_separate_training_data_uses_its_own_manifest(self, tmp_path):
        _, calls = self.experiment(tmp_path, False, curriculum=True)
        training = calls[0]
        assert training[training.index("--data") + 1] == str(tmp_path / "curriculum")

    def test_initialized_from_is_the_only_freshness_metadata(self, subtests):
        for adapter in (None, "initial"):
            with (
                subtests.test(adapter=adapter),
                tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory,
            ):
                output, calls = self.experiment(
                    directory, False, initialize_adapter=adapter
                )
                metadata = json.loads((output / "experiment.json").read_text())
                assert "fresh_adapter" not in metadata
                assert metadata["initialized_from"] == (
                    str(Path(directory) / adapter) if adapter else None
                )
                training = next(command for command in calls if "train" in command)
                assert training[:5] == [
                    sys.executable,
                    "-u",
                    "-m",
                    "model.train.main",
                    "train",
                ]
                assert training[training.index("--steps") + 1] == "4"
                assert training[training.index("--learning-rate") + 1] == "0.0001"
                assert ("--initialize-adapter" in training) == (adapter is not None)
                assert training[training.index("--device") + 1] == "cpu"
                assert training[training.index("--eval-samples") + 1] == "12"
                assert training[training.index("--data") + 1] == str(
                    (Path(directory) / "data").resolve()
                )
                assert training[training.index("--output") + 1] == str(
                    (output / "adapter").resolve()
                )
                assert "--evaluate-every" not in training
                assert "--resume" not in training

    def test_hf_evaluation_leaves_final_unopened_when_dev_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            output, calls = self.experiment(directory, False)
            assert len(calls) == 2
            for command in calls:
                assert command[command.index("--threads") + 1] == "1"
            assert all(
                (
                    "convert_lora_to_gguf.py" not in " ".join(command)
                    for command in calls
                )
            )
            assert all(("--final" not in command for command in calls))
            assert not (output / "test").exists()
            assert (
                json.loads((output / "experiment.json").read_text())["status"]
                == "needs_development"
            )

    def test_final_opens_only_after_development_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            output, calls = self.experiment(directory, True)
            assert len(calls) == 3
            assert "train" in calls[0]
            assert "--final" not in calls[1]
            assert "--final" in calls[2]
            assert not (output / "baseline-dev").exists()
            assert (
                json.loads((output / "experiment.json").read_text())["status"]
                == "complete"
            )

    def test_development_and_final_share_evaluation_options(self):
        with tempfile.TemporaryDirectory() as directory:
            output, calls = self.experiment(directory, True)
            _, development, final = calls
            for command in (development, final):
                assert command[:5] == [
                    "uv",
                    "run",
                    "--locked",
                    "--project",
                    str(Path(__file__).resolve().parents[2] / "benchmark"),
                ]
                assert "--backend" not in command
                assert command.count("--device") == 1
                assert command[command.index("--extra") + 1] == "local"
                assert command[command.index("--base-model") + 1] == str(
                    Path(__file__).resolve().parents[1] / "artifacts/base"
                )
                assert command[command.index("--threads") + 1] == "1"
                assert command[command.index("--data") + 1] == str(
                    (output.parent / "data").resolve()
                )
            assert development[development.index("--adapter") + 1] == str(
                (output / "adapter").resolve()
            )
            assert development[development.index("--label") + 1] == "run"

    def test_cuda_reaches_training_and_every_hf_evaluation_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            _, calls = self.experiment(directory, True, "cuda")
            assert len(calls) == 3
            for command in calls:
                assert command[command.index("--device") + 1] == "cuda"
                if "model.benchmark.main" in command:
                    assert command[command.index("--extra") + 1] == "cuda"

    @pytest.mark.parametrize(
        "arguments",
        [("--backend", "gguf"), ("--backend", "hf"), ("--baseline-before-training",)],
    )
    def test_retired_options_are_rejected(self, tmp_path, capsys, arguments):
        output = tmp_path / "run"
        with pytest.raises(SystemExit) as error:
            main(["--output", str(output), *arguments])
        assert error.value.code == 2
        assert "unrecognized arguments" in capsys.readouterr().err
        assert not output.exists()
