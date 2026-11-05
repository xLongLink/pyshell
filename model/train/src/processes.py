"""Stream subprocess output to a log and console with bounded interruption cleanup."""

import subprocess
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
    evaluate_every=None,
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
        *(
            ["--evaluate-every", str(evaluate_every)]
            if evaluate_every is not None
            else []
        ),
    ]


def run_logged(command, log_path, *, cwd, on_line=None):
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                if on_line is not None:
                    on_line(line)
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
