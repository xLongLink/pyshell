"""Check explicit device selection and bounded resume budgets."""

import io
import json
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from model.benchmark.src import runtime
from model.train.src import training
from model.train.src.training import parse_args


class TestTrainingArguments:
    def test_data_defaults_follow_serialization(self, subtests):
        repo = Path(__file__).resolve().parents[2]
        for serialization, directory in (
            ("raw", "train/data/raw"),
            ("chat", "train/data/chat"),
        ):
            with subtests.test(serialization=serialization):
                args = parse_args(["--serialization", serialization])
                assert args.data == repo / directory

    def test_explicit_data_remains_unresolved_in_both_modes(self, subtests):
        for serialization in ("raw", "chat"):
            with subtests.test(serialization=serialization):
                args = parse_args(
                    ["--serialization", serialization, "--data", "custom/../data"]
                )
                assert args.data == Path("custom/../data")

    def test_cpu_remains_default(self):
        assert parse_args([]).device == "cpu"

    def test_resume_and_initialization_are_individually_accepted(self, subtests):
        for option, attribute in (
            ("--resume", "resume"),
            ("--initialize-adapter", "initialize_adapter"),
        ):
            with subtests.test(option=option):
                args = parse_args([option, "prior"])
                assert getattr(args, attribute) == Path("prior")

    def test_resume_and_initialization_cannot_be_combined(self):
        with redirect_stderr(io.StringIO()), pytest.raises(SystemExit) as raised:
            parse_args(["--resume", "prior", "--initialize-adapter", "initial"])
        assert raised.value.code == 2

    def test_smoke_stop_preserves_original_schedule(self):
        args = parse_args(
            ["--device", "cuda", "--steps", "1200", "--stop-after", "804"]
        )
        assert args.device == "cuda"
        assert args.steps == 1200
        assert args.stop_after == 804

    def test_stop_requires_an_optimizer_boundary_within_the_budget(self, subtests):
        for stop in ("0", "803", "1204"):
            with (
                subtests.test(stop=stop),
                redirect_stderr(io.StringIO()),
                pytest.raises(SystemExit),
            ):
                parse_args(["--steps", "1200", "--stop-after", stop])


class TestTrainingRuntime:
    def test_unavailable_cuda_is_checked_before_the_output_guard(self):
        output = Mock()
        with patch.object(runtime, "torch") as torch:
            torch.cuda.is_available.return_value = False
            with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
                training.train(SimpleNamespace(device="cuda", output=output))
            output.exists.assert_not_called()
            output.mkdir.assert_not_called()
            torch.set_num_threads.assert_not_called()
            torch.set_num_interop_threads.assert_not_called()

    def test_existing_output_is_guarded_after_device_policy_before_threading(self):
        with patch.object(runtime, "torch") as torch:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            def exists():
                assert not torch.backends.cuda.matmul.allow_tf32
                assert not torch.backends.cudnn.allow_tf32
                return True

            output = Mock(exists=Mock(side_effect=exists))
            with pytest.raises(FileExistsError, match="Refusing to overwrite"):
                training.train(SimpleNamespace(device="cpu", output=output))
            output.mkdir.assert_not_called()
            torch.set_num_threads.assert_not_called()
            torch.set_num_interop_threads.assert_not_called()

    def test_output_is_created_before_threading_and_interop_errors_propagate(self):
        error = RuntimeError("interop threads already initialized")
        with (
            tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory,
            patch.object(runtime, "torch") as torch,
            patch.object(training.AutoTokenizer, "from_pretrained") as tokenizer,
        ):
            output = Path(directory) / "adapter"

            def set_num_threads(threads):
                assert output.is_dir()

            torch.set_num_threads.side_effect = set_num_threads
            torch.set_num_interop_threads.side_effect = error
            with pytest.raises(RuntimeError) as raised:
                training.train(SimpleNamespace(device="cpu", output=output, threads=3))
            assert raised.value is error
            torch.set_num_threads.assert_called_once_with(3)
            torch.set_num_interop_threads.assert_called_once_with(1)
            tokenizer.assert_not_called()


class TestTrainingProgress:
    def run_training(self, steps, stop_after=None, fail_at=None, resume_updates=0):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            record = {"id": "example", "prompt": "x", "completion": "done('ok')"}
            (data / "train.jsonl").write_text(json.dumps(record) + "\n")
            (data / "validation.jsonl").write_text("")
            argv = [
                "--data",
                str(data),
                "--output",
                str(root / "adapter"),
                "--steps",
                str(steps),
                "--threads",
                "1",
            ]
            if stop_after is not None:
                argv.extend(["--stop-after", str(stop_after)])
            if resume_updates:
                argv.extend(["--resume", str(root / "resume")])
            args = parse_args(argv)
            model = torch.nn.Linear(1, 1, bias=False)
            model.config = SimpleNamespace(use_cache=True)
            model.peft_config = {"default": SimpleNamespace(r=16)}
            model.gradient_checkpointing_enable = Mock()
            model.print_trainable_parameters = Mock()
            checkpoints = []
            rates = []
            losses = 0
            original_step = torch.optim.AdamW.step

            def loss_for(model, item):
                nonlocal losses
                losses += 1
                if losses == fail_at:
                    raise KeyboardInterrupt
                return model.weight.square().mean()

            def optimizer_step(optimizer):
                rates.append(optimizer.param_groups[0]["lr"] / args.learning_rate)
                return original_step(optimizer)

            def save_checkpoint(model, tokenizer, optimizer, status, output, name):
                assert status["rank"] == 16
                checkpoints.append((name, dict(status)))

            with (
                patch.object(training.AutoTokenizer, "from_pretrained") as tokenizer,
                patch.object(
                    training.AutoModelForImageTextToText,
                    "from_pretrained",
                    return_value=model,
                ),
                patch.object(training, "get_peft_model", return_value=model),
                patch.object(training.PeftModel, "from_pretrained", return_value=model),
                patch.object(
                    training,
                    "load_resume",
                    return_value={
                        "updates": resume_updates,
                        "validation_loss_before": None,
                    },
                ),
                patch.object(training, "loss_for", side_effect=loss_for),
                patch.object(training, "save_checkpoint", side_effect=save_checkpoint),
                patch.object(training.torch.optim.AdamW, "step", optimizer_step),
                patch.object(runtime.torch, "set_num_interop_threads"),
                patch.object(training.ctypes, "CDLL", return_value=SimpleNamespace()),
                patch.object(training.gc, "collect"),
                redirect_stdout(io.StringIO()),
            ):
                tokenizer.return_value.encode.return_value = [1, 2]
                if fail_at is None:
                    training.train(args)
                else:
                    with pytest.raises(KeyboardInterrupt):
                        training.train(args)
            return checkpoints, rates

    def test_checkpoints_keep_the_last_completed_step(self, subtests):
        cases = (
            (
                100,
                None,
                None,
                0,
                [("checkpoint-100", "running", 100), ("final", "completed", 100)],
            ),
            (8, 4, None, 0, [("final", "paused", 4)]),
            (
                104,
                None,
                101,
                0,
                [
                    ("checkpoint-100", "running", 100),
                    ("interrupted", "interrupted", 100),
                ],
            ),
            (8, None, 2, 0, [("interrupted", "interrupted", 1)]),
            (8, None, 1, 1, [("interrupted", "interrupted", 4)]),
        )
        for steps, stop_after, fail_at, resume_updates, expected in cases:
            with subtests.test(
                steps=steps,
                stop_after=stop_after,
                fail_at=fail_at,
                resume_updates=resume_updates,
            ):
                checkpoints, _ = self.run_training(
                    steps, stop_after, fail_at, resume_updates
                )
                assert [
                    (name, status["status"], status["completed_steps"])
                    for name, status in checkpoints
                ] == expected
                for _, status in checkpoints:
                    assert status["validation_loss_before"] is None
                if fail_at is None:
                    assert checkpoints[-1][1]["validation_loss_after"] is None

    def test_warmup_and_decay_follow_the_original_budget(self, subtests):
        for steps, stop_after, resume_updates, updates, first, last in (
            (4, None, 0, 1, 1.0, 1.0),
            (100, None, 0, 25, 0.5, 0.1),
            (100, 4, 0, 1, 0.5, 0.5),
            (100, None, 1, 24, 1.0, 0.1),
        ):
            with subtests.test(
                steps=steps, stop_after=stop_after, resume_updates=resume_updates
            ):
                _, rates = self.run_training(
                    steps, stop_after, resume_updates=resume_updates
                )
                assert len(rates) == updates
                assert rates[0] == pytest.approx(first, rel=0, abs=5e-08)
                assert rates[-1] == pytest.approx(last, rel=0, abs=5e-08)
