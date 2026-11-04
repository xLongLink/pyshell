"""Check experiment phase selection without training or opening a real holdout."""

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from model.benchmark.src.corpus import sha256
from model.benchmark.src.task_types import FAMILIES
from model.train.src.experiment import main, run


class TestExperiment:
    def experiment(
        self, directory, backend, ready, device="cpu", initialize_adapter="initial"
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
        args = SimpleNamespace(
            data=data,
            output=root / "run",
            steps=4,
            threads=1,
            device=device,
            learning_rate=1e-4,
            training_data=None,
            initialize_adapter=root / initialize_adapter
            if initialize_adapter is not None
            else None,
            backend=backend,
            baseline_before_training=backend == "hf" and initialize_adapter is not None,
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
            patch("model.train.src.experiment.subprocess.run", side_effect=execute),
            redirect_stdout(io.StringIO()) as stdout,
        ):
            metadata = run(args)
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

    def test_initialized_from_is_the_only_freshness_metadata(self, subtests):
        for adapter in (None, "initial"):
            with (
                subtests.test(adapter=adapter),
                tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory,
            ):
                output, calls = self.experiment(
                    directory, "hf", False, initialize_adapter=adapter
                )
                metadata = json.loads((output / "experiment.json").read_text())
                assert "fresh_adapter" not in metadata
                assert metadata["initialized_from"] == (
                    str(Path(directory) / adapter) if adapter else None
                )
                training = next(command for command in calls if "train" in command)
                assert ("--initialize-adapter" in training) == (adapter is not None)

    def test_hf_skips_unused_export_and_leaves_final_unopened_when_dev_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            output, calls = self.experiment(directory, "hf", False)
            assert len(calls) == 3
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

    def test_gguf_exports_before_evaluation_and_opens_final_only_after_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            output, calls = self.experiment(directory, "gguf", True)
            assert len(calls) == 4
            assert "convert_lora_to_gguf.py" in calls[1][1]
            assert "--final" not in calls[2]
            assert "--final" in calls[3]
            assert (
                json.loads((output / "experiment.json").read_text())["status"]
                == "complete"
            )

    def test_baseline_and_development_share_evaluation_options(self):
        with tempfile.TemporaryDirectory() as directory:
            output, calls = self.experiment(directory, "hf", True)
            baseline, _, development, final = calls
            for command in (baseline, development, final):
                assert command[command.index("--backend") + 1] == "hf"
                assert command[command.index("--threads") + 1] == "1"
                assert command[command.index("--data") + 1] == str(
                    (output.parent / "data").resolve()
                )
            assert baseline[baseline.index("--adapter") + 1] == str(
                (output.parent / "initial").resolve()
            )
            assert baseline[baseline.index("--label") + 1] == "run-baseline"
            assert development[development.index("--adapter") + 1] == str(
                (output / "adapter").resolve()
            )
            assert development[development.index("--label") + 1] == "run"

    def test_cuda_reaches_training_and_every_hf_evaluation_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            _, calls = self.experiment(directory, "hf", True, "cuda")
            assert len(calls) == 4
            for command in calls:
                assert command[command.index("--device") + 1] == "cuda"

    def test_cuda_does_not_silently_use_cpu_gguf_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            with pytest.raises(ValueError, match="HF"):
                self.experiment(directory, "gguf", True, "cuda")
