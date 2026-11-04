"""Check experiment phase selection without training or opening a real holdout."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dataset.benchmark.src.corpus import sha256
from dataset.benchmark.src.task_types import FAMILIES
from model.train.src.experiment import main, run


class ExperimentTests(unittest.TestCase):
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
            if "dataset.benchmark.main" not in command:
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
        self.assertEqual((args.output / "experiment.json").read_text(), serialized)
        self.assertTrue(stdout.getvalue().endswith(serialized))
        return args.output, calls

    def test_main_does_not_return_experiment_metadata(self):
        with patch(
            "model.train.src.experiment.run", return_value={"status": "complete"}
        ) as experiment:
            self.assertIsNone(main(["--output", "run"]))
        self.assertEqual(experiment.call_args.args[0].output, Path("run"))

    def test_initialized_from_is_the_only_freshness_metadata(self):
        for adapter in (None, "initial"):
            with (
                self.subTest(adapter=adapter),
                tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory,
            ):
                output, calls = self.experiment(
                    directory, "hf", False, initialize_adapter=adapter
                )
                metadata = json.loads((output / "experiment.json").read_text())
                self.assertNotIn("fresh_adapter", metadata)
                self.assertEqual(
                    metadata["initialized_from"],
                    str(Path(directory) / adapter) if adapter else None,
                )
                training = next(command for command in calls if "train" in command)
                self.assertEqual(
                    "--initialize-adapter" in training, adapter is not None
                )

    def test_hf_skips_unused_export_and_leaves_final_unopened_when_dev_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            output, calls = self.experiment(directory, "hf", False)
            self.assertEqual(len(calls), 3)
            for command in calls:
                self.assertEqual(command[command.index("--threads") + 1], "1")
            self.assertTrue(
                all(
                    "convert_lora_to_gguf.py" not in " ".join(command)
                    for command in calls
                )
            )
            self.assertTrue(all("--final" not in command for command in calls))
            self.assertFalse((output / "test").exists())
            self.assertEqual(
                json.loads((output / "experiment.json").read_text())["status"],
                "needs_development",
            )

    def test_gguf_exports_before_evaluation_and_opens_final_only_after_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            output, calls = self.experiment(directory, "gguf", True)
            self.assertEqual(len(calls), 4)
            self.assertIn("convert_lora_to_gguf.py", calls[1][1])
            self.assertNotIn("--final", calls[2])
            self.assertIn("--final", calls[3])
            self.assertEqual(
                json.loads((output / "experiment.json").read_text())["status"],
                "complete",
            )

    def test_baseline_and_development_share_evaluation_options(self):
        with tempfile.TemporaryDirectory() as directory:
            output, calls = self.experiment(directory, "hf", True)
            baseline, _, development, final = calls
            for command in (baseline, development, final):
                self.assertEqual(command[command.index("--backend") + 1], "hf")
                self.assertEqual(command[command.index("--threads") + 1], "1")
                self.assertEqual(
                    command[command.index("--data") + 1],
                    str((output.parent / "data").resolve()),
                )
            self.assertEqual(
                baseline[baseline.index("--adapter") + 1],
                str((output.parent / "initial").resolve()),
            )
            self.assertEqual(baseline[baseline.index("--label") + 1], "run-baseline")
            self.assertEqual(
                development[development.index("--adapter") + 1],
                str((output / "adapter").resolve()),
            )
            self.assertEqual(development[development.index("--label") + 1], "run")

    def test_cuda_reaches_training_and_every_hf_evaluation_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            _, calls = self.experiment(directory, "hf", True, "cuda")
            self.assertEqual(len(calls), 4)
            for command in calls:
                self.assertEqual(command[command.index("--device") + 1], "cuda")

    def test_cuda_does_not_silently_use_cpu_gguf_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "HF"):
                self.experiment(directory, "gguf", True, "cuda")
