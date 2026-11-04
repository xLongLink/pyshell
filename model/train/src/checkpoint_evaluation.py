"""Run checkpoint evaluation with one owner for its subprocess group."""

import os
import signal
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def evaluation_command(checkpoint, step, history, *, gguf=None):
    return [
        sys.executable,
        "-u",
        "-m",
        "model.train.main",
        "checkpoint",
        "--checkpoint",
        str(checkpoint),
        "--step",
        str(step),
        "--history",
        str(history),
        *(["--gguf", str(gguf)] if gguf is not None else []),
    ]


def evaluate_checkpoint(checkpoint, step, history):
    process = subprocess.Popen(
        evaluation_command(checkpoint, step, history),
        cwd=REPO,
        start_new_session=True,
    )
    try:
        return process.wait()
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
