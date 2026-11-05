"""Exercise step-log visibility and lifetime during training."""

import io
import json
import os
import tempfile
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from model.train.src import training


class TestTrainingLog:
    @pytest.fixture(autouse=True)
    def setup(self):
        with ExitStack() as stack:
            root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            data = root / "data"
            data.mkdir()
            (data / "train.jsonl").write_text('{"id": "example"}\n')
            (data / "validation.jsonl").write_text("")
            self.args = training.parse_args(
                [
                    "--data",
                    str(data),
                    "--output",
                    str(root / "adapter"),
                    "--steps",
                    "4",
                    "--accumulation",
                    "1",
                    "--threads",
                    "1",
                ]
            )
            self.log_path = self.args.output / "steps.jsonl"
            self.handles = []
            self.checkpoints = []
            model = torch.nn.Linear(1, 1, bias=False)
            model.config = SimpleNamespace(use_cache=True)
            model.peft_config = {"default": SimpleNamespace(r=8)}
            model.gradient_checkpointing_enable = Mock()
            model.print_trainable_parameters = Mock()
            self.loss = Mock(
                side_effect=lambda model, item: model.weight.square().mean()
            )
            stack.enter_context(
                patch.multiple(
                    training,
                    AutoTokenizer=Mock(),
                    AutoModelForImageTextToText=Mock(
                        from_pretrained=Mock(return_value=model)
                    ),
                    get_peft_model=Mock(return_value=model),
                    encode=Mock(return_value={"id": "example", "ids": [1, 2]}),
                    loss_for=self.loss,
                    evaluate_loss=Mock(return_value=None),
                    save_checkpoint=Mock(side_effect=self.check_checkpoint),
                )
            )
            stack.enter_context(patch.object(training.torch, "set_num_interop_threads"))
            stack.enter_context(
                patch.object(training.ctypes, "CDLL", return_value=SimpleNamespace())
            )
            stack.enter_context(patch.object(training.gc, "collect"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            original_open = Path.open

            def open_path(path, mode="r", *args, **kwargs):
                handle = original_open(path, mode, *args, **kwargs)
                if path == self.log_path and mode == "a":
                    self.handles.append(handle)
                return handle

            stack.enter_context(patch.object(Path, "open", open_path))
            yield

    def logged_steps(self):
        with self.log_path.open("rb") as file:
            os.fsync(file.fileno())
            return [json.loads(line)["step"] for line in file]

    def check_checkpoint(self, model, tokenizer, optimizer, status, output, name):
        assert len(self.handles) == 1
        assert self.handles[0].closed == (name in ("final", "interrupted"))
        assert self.logged_steps() == list(range(1, status["completed_steps"] + 1))
        self.checkpoints.append(name)

    def test_periodic_checkpoint_sees_rows_and_final_checkpoint_sees_closed_log(self):
        self.args.steps = 100
        training.train(self.args)
        assert self.checkpoints == ["checkpoint-100", "final"]
        assert self.handles[0].closed

    def test_each_row_is_visible_before_the_next_step(self):
        completed_steps = []

        def loss_for(model, item):
            assert self.logged_steps() == completed_steps
            completed_steps.append(len(completed_steps) + 1)
            return model.weight.square().mean()

        self.loss.side_effect = loss_for
        training.train(self.args)
        assert self.checkpoints == ["final"]
        assert self.logged_steps() == [1, 2, 3, 4]

    def assert_loss_failure(self, error):
        self.loss.side_effect = [torch.tensor(1.0, requires_grad=True), error]
        with pytest.raises(type(error)) as raised:
            training.train(self.args)
        assert raised.value is error
        assert self.checkpoints == ["interrupted"]
        assert self.handles[0].closed
        assert self.logged_steps() == [1]

    def test_training_failure_closes_log_and_propagates_original_exception(self):
        self.assert_loss_failure(RuntimeError("loss failed"))

    def test_keyboard_interrupt_closes_log_before_interrupted_checkpoint(self):
        self.assert_loss_failure(KeyboardInterrupt())

    def test_checkpoint_failure_closes_log_and_propagates_original_exception(self):
        self.args.steps = 100
        error = OSError("checkpoint failed")

        def save_checkpoint(*args):
            self.check_checkpoint(*args)
            if args[-1] == "checkpoint-100":
                raise error

        training.save_checkpoint.side_effect = save_checkpoint
        with pytest.raises(OSError, match="checkpoint failed") as raised:
            training.train(self.args)
        assert raised.value is error
        assert self.checkpoints == ["checkpoint-100", "interrupted"]
        assert self.handles[0].closed
