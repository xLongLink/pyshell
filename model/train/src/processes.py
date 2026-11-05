import sys
from pathlib import Path


def benchmark_command(subcommand, *arguments, backend="gguf", device="cpu"):
    project = Path(__file__).resolve().parents[2] / "benchmark"
    return [
        "uv",
        "run",
        "--locked",
        "--project",
        str(project),
        *(
            ["--extra", "cuda" if device == "cuda" else "local"]
            if backend == "hf"
            else []
        ),
        "python",
        "-u",
        "-m",
        "model.benchmark.main",
        subcommand,
        *(["--backend", backend, "--device", device] if subcommand == "cpu" else []),
        *arguments,
    ]


def training_command(
    *,
    data,
    output,
    steps,
    threads,
    learning_rate,
    device=None,
    eval_samples=None,
    initialize_adapter=None,
    resume=None,
):
    return [
        sys.executable,
        "-u",
        "-m",
        "model.train.main",
        "train",
        "--data",
        str(data),
        "--output",
        str(output),
        "--steps",
        str(steps),
        "--threads",
        str(threads),
        "--learning-rate",
        str(learning_rate),
        *(["--device", device] if device is not None else []),
        *(["--eval-samples", str(eval_samples)] if eval_samples is not None else []),
        *(
            ["--initialize-adapter", str(initialize_adapter)]
            if initialize_adapter is not None
            else []
        ),
        *(["--resume", str(resume)] if resume is not None else []),
    ]
