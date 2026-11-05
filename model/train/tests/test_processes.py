import sys
from pathlib import Path

import pytest

from model.train.src import processes


@pytest.mark.parametrize(
    ("backend", "device", "extra"),
    [("gguf", "cpu", None), ("hf", "cpu", "local"), ("hf", "cuda", "cuda")],
)
def test_benchmark_runs_in_its_own_environment(backend, device, extra):
    command = processes.benchmark_command(
        "cpu", "--label", "pilot", backend=backend, device=device
    )
    assert command[:5] == [
        "uv",
        "run",
        "--locked",
        "--project",
        str(Path(__file__).resolve().parents[2] / "benchmark"),
    ]
    assert command[command.index("python") :] == [
        "python",
        "-u",
        "-m",
        "model.benchmark.main",
        "cpu",
        "--backend",
        backend,
        "--device",
        device,
        "--label",
        "pilot",
    ]
    if extra is None:
        assert "--extra" not in command
    else:
        assert command[command.index("--extra") + 1] == extra


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"device": "cuda", "eval_samples": 12, "initialize_adapter": Path("initial")},
        {"resume": Path("prior/../checkpoint")},
    ],
)
def test_training_command_preserves_explicit_options_and_paths(options):
    command = processes.training_command(
        data=Path("custom/../data"),
        output=Path("adapter"),
        steps=24,
        threads=2,
        learning_rate=1e-4,
        **options,
    )
    assert command[:5] == [sys.executable, "-u", "-m", "model.train.main", "train"]
    expected = {
        "data": "custom/../data",
        "output": "adapter",
        "steps": 24,
        "threads": 2,
        "learning_rate": 1e-4,
        **options,
    }
    assert dict(zip(command[5::2], command[6::2])) == {
        "--" + key.replace("_", "-"): str(value) for key, value in expected.items()
    }


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"device": "cuda", "eval_samples": 12, "initialize_adapter": Path("initial")},
        {"resume": Path("prior/../checkpoint")},
    ],
)
def test_training_command_roundtrips_through_training_parser(options):
    training = pytest.importorskip("model.train.src.training")
    command = processes.training_command(
        data=Path("custom/../data"),
        output=Path("adapter"),
        steps=24,
        threads=2,
        learning_rate=1e-4,
        **options,
    )
    args = training.parse_args(command[5:])
    assert args.data == Path("custom/../data")
    assert args.output == Path("adapter")
    assert args.steps == 24
    assert args.threads == 2
    assert args.learning_rate == 1e-4
    assert args.device == options.get("device", "cpu")
    assert args.eval_samples == options.get("eval_samples", 60)
    assert args.initialize_adapter == options.get("initialize_adapter")
    assert args.resume == options.get("resume")
