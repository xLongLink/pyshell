"""Serialize CPU training and proof evaluations instead of oversubscribing Torch."""

import fcntl
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def exclusive_cpu():
    path = Path(__file__).resolve().parents[1] / "artifacts/cpu-workload.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                "Waiting for the other CPU experiment to release its slot.", flush=True
            )
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
