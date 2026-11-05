import io
import tempfile
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

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

    @pytest.mark.parametrize(
        "option", ["--steps", "--accumulation", "--max-length", "--threads"]
    )
    def test_nonpositive_training_limits_are_rejected(self, option):
        with pytest.raises(SystemExit):
            parse_args([option, "0"])

    def test_negative_validation_limit_is_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--eval-samples", "-1"])


class TestTrainingRuntime:
    @pytest.mark.parametrize("entrypoint", ["train", "main"])
    @pytest.mark.parametrize(
        "requested,resolved",
        [("auto", "cpu"), ("auto", "cuda"), ("cpu", "cpu"), ("cuda", "cuda")],
    )
    def test_device_is_resolved_once_and_cpu_work_is_locked(
        self, tmp_path, entrypoint, requested, resolved
    ):
        output = tmp_path / "adapter"
        active = False
        error = RuntimeError("stop before model loading")

        @contextmanager
        def lock():
            nonlocal active
            assert not output.exists()
            active = True
            try:
                yield
            finally:
                active = False

        def configure_threads(threads):
            assert threads == 3
            assert active == (resolved == "cpu")
            assert output.is_dir()
            raise error

        argv = ["--device", requested, "--output", str(output), "--threads", "3"]
        with (
            patch.object(training, "configure_device", return_value=resolved) as device,
            patch.object(training, "exclusive_cpu", side_effect=lock) as cpu,
            patch.object(training, "configure_threads", side_effect=configure_threads),
            patch.object(training.torch.cuda, "device_count", return_value=1),
            pytest.raises(RuntimeError) as raised,
        ):
            if entrypoint == "main":
                training.main(argv)
            else:
                training.train(parse_args(argv))
        assert raised.value is error
        device.assert_called_once_with(requested)
        assert cpu.call_count == (1 if resolved == "cpu" else 0)
        assert not active

    @pytest.mark.parametrize("world_size", ["0", "2"])
    @pytest.mark.parametrize(
        "variable",
        ["WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"],
    )
    def test_multiple_processes_are_rejected_before_creating_output(
        self, variable, world_size
    ):
        output = Mock()
        with (
            patch.dict("os.environ", {variable: world_size}),
            pytest.raises(ValueError, match="one process"),
        ):
            training.train(SimpleNamespace(device="cpu", output=output))
        output.mkdir.assert_not_called()

    def test_initialized_distributed_world_is_checked(self):
        with (
            patch.object(
                training.torch.distributed, "is_initialized", return_value=True
            ),
            patch.object(training.torch.distributed, "get_world_size", return_value=2),
            pytest.raises(ValueError, match="one process"),
        ):
            training.train(SimpleNamespace(device="cpu"))

    def test_multiple_visible_cuda_devices_are_rejected(self):
        with (
            patch.object(training.torch.cuda, "is_available", return_value=True),
            patch.object(training.torch.cuda, "device_count", return_value=2),
            pytest.raises(ValueError, match="one CPU or CUDA device"),
        ):
            training.train(SimpleNamespace(device="cuda"))

    @pytest.mark.parametrize("precision", ["fp16", "bf16"])
    def test_accelerate_mixed_precision_is_rejected(self, precision):
        output = Mock()
        with (
            patch.dict("os.environ", {"ACCELERATE_MIXED_PRECISION": precision}),
            pytest.raises(ValueError, match="requires float32"),
        ):
            training.train(SimpleNamespace(device="cpu", output=output))
        output.mkdir.assert_not_called()

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


def test_empty_training_data_is_rejected(training_run, tmp_path):
    (tmp_path / "data/train.jsonl").write_text("")
    with pytest.raises(ValueError, match="must not be empty"):
        training_run("empty")


def test_overlength_training_data_is_rejected(training_run):
    with pytest.raises(ValueError, match="refusing to truncate"):
        training_run("oversized", "--max-length", "1")
