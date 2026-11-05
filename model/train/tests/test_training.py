"""Check explicit device selection and bounded resume budgets."""

import io
import json
import math
import random
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
    def test_data_defaults_to_raw_protocol(self):
        repo = Path(__file__).resolve().parents[2]
        assert parse_args([]).data == repo / "train/data/raw"

    def test_explicit_data_remains_unresolved(self):
        assert parse_args(["--data", "custom/../data"]).data == Path("custom/../data")

    def test_serialization_switch_is_no_longer_supported(self, subtests):
        for serialization in ("raw", "chat"):
            with (
                subtests.test(serialization=serialization),
                redirect_stderr(io.StringIO()),
                pytest.raises(SystemExit) as raised,
            ):
                parse_args(["--serialization", serialization])
            assert raised.value.code == 2

    def test_device_is_automatically_selected_by_default(self):
        assert parse_args([]).device == "auto"

    def test_periodic_worker_evaluation_is_no_longer_supported(self, capsys):
        with pytest.raises(SystemExit) as error:
            parse_args(["--evaluate-every", "100"])
        assert error.value.code == 2
        assert "unrecognized arguments: --evaluate-every" in capsys.readouterr().err

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
            argv.extend(["--device", "cpu"])
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

            def optimizer_step(optimizer, closure=None):
                rates.append(optimizer.param_groups[0]["lr"] / args.learning_rate)
                return original_step(optimizer, closure)

            def save_checkpoint(
                model, tokenizer, optimizer, status, output, name, *, scheduler
            ):
                assert status["rank"] == 16
                assert status["serialization"] == "raw"
                assert status["serialization_format"] == "task-code-console-v1"
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
                        "training_seconds": 600,
                    },
                ),
                patch.object(training, "loss_for", side_effect=loss_for),
                patch.object(training, "save_checkpoint", side_effect=save_checkpoint),
                patch.object(training.torch.optim.AdamW, "step", optimizer_step),
                patch.object(runtime.torch, "set_num_interop_threads"),
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
                    assert status["training_seconds"] >= (600 if resume_updates else 0)
                durations = [status["training_seconds"] for _, status in checkpoints]
                assert durations == sorted(durations)
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


class TinyResponseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.arange(32, dtype=torch.float32).reshape(4, 8) / 100
        )
        self.positions = []

    def forward(self, input_ids, *, use_cache, logits_to_keep):
        assert not use_cache
        self.positions.append(logits_to_keep.tolist())
        return SimpleNamespace(logits=self.logits[input_ids[:, logits_to_keep]])


class TestResponseTrainer:
    @pytest.mark.parametrize("steps", [1, 4, 5, 100, 1200])
    def test_library_scheduler_preserves_every_used_legacy_learning_rate(self, steps):
        updates = math.ceil(steps / 4)
        warmup = min(10, max(1, updates // 10))
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
        scheduler = training.scheduler_for(optimizer, steps, 4)
        for update in range(1, updates + 1):
            expected = (
                update / warmup
                if update <= warmup
                else 0.1
                + 0.9
                * (1 + math.cos(math.pi * (update - warmup) / max(1, updates - warmup)))
                / 2
            )
            assert optimizer.param_groups[0]["lr"] == pytest.approx(
                0.1 * expected, rel=1e-14, abs=0
            )
            optimizer.step()
            scheduler.step()

    @pytest.mark.parametrize("legacy,steps", [(False, 9), (True, 100)])
    def test_saved_adapter_optimizer_and_scheduler_resume_match_uninterrupted_training(
        self, tmp_path, legacy, steps
    ):
        data = tmp_path / "data"
        data.mkdir()
        records = [
            {"id": str(i), "prompt": "prompt", "completion": "code"} for i in range(3)
        ]
        (data / "train.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in records)
        )
        (data / "validation.jsonl").write_text("")

        def model_from_pretrained(*args, **kwargs):
            model = TinyResponseModel()
            model.config = SimpleNamespace(use_cache=False)
            model.peft_config = {"default": SimpleNamespace(r=8)}
            model.gradient_checkpointing_enable = Mock()
            model.print_trainable_parameters = Mock()
            model.save_pretrained = lambda path: torch.save(
                model.state_dict(), path / "weights.pt"
            )
            return model

        def adapter_from_pretrained(model, path, **kwargs):
            model.load_state_dict(torch.load(path / "weights.pt", weights_only=True))
            return model

        def run(name, *extra):
            args = parse_args(
                [
                    "--data",
                    str(data),
                    "--output",
                    str(tmp_path / name),
                    "--steps",
                    str(steps),
                    "--accumulation",
                    "4",
                    "--threads",
                    "1",
                    "--device",
                    "cpu",
                    *extra,
                ]
            )
            training.train(args)
            return args.output

        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2]
        with (
            patch.object(
                training.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ),
            patch.object(
                training.AutoModelForImageTextToText,
                "from_pretrained",
                side_effect=model_from_pretrained,
            ),
            patch.object(
                training, "get_peft_model", side_effect=lambda model, config: model
            ),
            patch.object(
                training.PeftModel,
                "from_pretrained",
                side_effect=adapter_from_pretrained,
            ),
            patch.object(runtime.torch, "set_num_interop_threads"),
        ):
            full = run("full")
            paused = run("paused", "--stop-after", "4")
            if legacy:
                (paused / "scheduler.pt").unlink()
                state = torch.load(paused / "optimizer.pt", weights_only=True)
                for group in state["param_groups"]:
                    group.pop("initial_lr")
                torch.save(state, paused / "optimizer.pt")
            resumed = run("resumed", "--resume", str(paused))
            if not legacy:
                overrides = ["--steps", str(steps + 4), "--learning-rate", "0.0001"]
                changed = run("changed", "--resume", str(paused), *overrides)
                (paused / "scheduler.pt").unlink()
                changed_legacy = run(
                    "changed-legacy", "--resume", str(paused), *overrides
                )
                assert torch.equal(
                    torch.load(changed / "weights.pt", weights_only=True)["logits"],
                    torch.load(changed_legacy / "weights.pt", weights_only=True)[
                        "logits"
                    ],
                )
                assert not json.loads((changed / "training.json").read_text())[
                    "scheduler_restored"
                ]
        full_weights = torch.load(full / "weights.pt", weights_only=True)
        resumed_weights = torch.load(resumed / "weights.pt", weights_only=True)
        assert torch.equal(full_weights["logits"], resumed_weights["logits"])
        full_scheduler = torch.load(full / "scheduler.pt", weights_only=True)
        resumed_scheduler = torch.load(resumed / "scheduler.pt", weights_only=True)
        assert full_scheduler["last_epoch"] == resumed_scheduler["last_epoch"]
        assert full_scheduler["_last_lr"] == resumed_scheduler["_last_lr"]
        full_optimizer = torch.load(full / "optimizer.pt", weights_only=True)
        resumed_optimizer = torch.load(resumed / "optimizer.pt", weights_only=True)
        assert full_optimizer["param_groups"] == resumed_optimizer["param_groups"]
        for key, value in full_optimizer["state"][0].items():
            assert torch.equal(value, resumed_optimizer["state"][0][key])
        metadata = json.loads((resumed / "training.json").read_text())
        assert metadata["completed_steps"] == steps
        assert metadata["updates"] == (steps + 3) // 4
        assert metadata["optimizer_restored"]
        assert metadata["scheduler_restored"] == (not legacy)
        assert metadata["train_sha256"] == training.sha256(data / "train.jsonl")

        def rows(path):
            return [
                json.loads(line)
                for line in (path / "steps.jsonl").read_text().splitlines()
            ]

        assert [row["id"] for row in rows(full)] == [
            row["id"] for row in rows(paused) + rows(resumed)
        ]

    @pytest.mark.parametrize("steps,accumulation", [(1, 4), (5, 4), (10, 4), (9, 1)])
    def test_real_loss_matches_reference_updates_and_fixed_cycling(
        self, tmp_path, steps, accumulation
    ):
        items = [
            {"id": str(i), "ids": [0, i, 2, 3], "prefix_length": 2} for i in range(3)
        ]
        indices = list(range(len(items)))
        random.Random(713).shuffle(indices)
        sequence = [items[indices[i % len(indices)]] for i in range(steps)]
        model = TinyResponseModel()
        reference = TinyResponseModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        scheduler = training.scheduler_for(optimizer, steps, accumulation)
        reference_scheduler = training.scheduler_for(
            reference_optimizer, steps, accumulation
        )
        for start in range(0, steps, accumulation):
            group = sequence[start : start + accumulation]
            for item in group:
                (training.loss_for(reference, item) / len(group)).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            reference_optimizer.step()
            reference_scheduler.step()
            reference_optimizer.zero_grad(set_to_none=True)
        status = {"completed_steps": 0}
        log = io.StringIO()
        trainer = training.ResponseTrainer(
            model=model,
            args=training.TrainingArguments(
                output_dir=str(tmp_path),
                use_cpu=True,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=accumulation,
                num_train_epochs=1,
                remove_unused_columns=False,
                save_strategy="no",
                logging_strategy="no",
                report_to="none",
                disable_tqdm=True,
                seed=713,
            ),
            train_dataset=sequence,
            data_collator=training.single_example,
            optimizers=(optimizer, scheduler),
            status=status,
            log=log,
        )
        trainer.train()
        assert torch.equal(model.logits, reference.logits)
        assert model.positions == [[1, 2]] * steps
        assert status["completed_steps"] == steps
        assert trainer.state.global_step == (steps + accumulation - 1) // accumulation
        rows = [json.loads(line) for line in log.getvalue().splitlines()]
        assert [row["id"] for row in rows] == [item["id"] for item in sequence]
        assert all(row["loss"] > 0 for row in rows)

    def test_scheduler_state_and_legacy_position_resume_identical_rates(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
        scheduler = training.scheduler_for(optimizer, 100, 4)
        rates = []
        saved = None
        for update in range(25):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            if update == 6:
                saved = scheduler.state_dict()
        for restore_state in (True, False):
            resumed_optimizer = torch.optim.SGD(
                [torch.nn.Parameter(torch.ones(1))], lr=0.1
            )
            resumed = training.scheduler_for(
                resumed_optimizer, 100, 4, 0 if restore_state else 7
            )
            if restore_state:
                resumed.load_state_dict(saved)
                resumed_optimizer.param_groups[0]["lr"] = resumed.get_last_lr()[0]
            resumed_rates = []
            for _ in range(7, 25):
                resumed_rates.append(resumed_optimizer.param_groups[0]["lr"])
                resumed_optimizer.step()
                resumed.step()
            assert resumed_rates == rates[7:]
