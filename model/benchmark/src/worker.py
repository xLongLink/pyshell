import io
import os
import sys
import json
import signal
import pathlib
import contextlib
import statistics
from types import FrameType
from typing import Any, NoReturn


def deadline(_signal: int, _frame: FrameType | None) -> NoReturn:
    """Interrupt code that exceeds the execution deadline."""
    raise TimeoutError("Execution exceeded ten seconds")


def main() -> None:
    """Process execution requests from standard input."""
    import numpy as np
    import pandas as pd
    from api.shell import PythonShell

    protocol = sys.stdout
    incoming = sys.stdin
    shell = None
    handler = signal.signal(signal.SIGALRM, deadline)

    def publish(result: dict[str, Any]) -> None:
        protocol.write(json.dumps(result, default=str) + "\n")
        protocol.flush()

    try:
        with pathlib.Path(os.devnull).open("w") as sink:
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    if request["action"] not in ("reset", "execute"):
                        raise ValueError("Unsupported worker action")
                    signal.alarm(10)
                    with io.StringIO() as stdin, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                        sys.stdin = stdin
                        try:
                            if request["action"] == "reset":
                                shell = PythonShell()
                                shell.namespace.update(np=np, pd=pd, statistics=statistics, json=json)
                            result = shell.execute(request["source"])
                        finally:
                            sys.stdin = incoming
                    signal.alarm(0)
                    publish(result)
                except BaseException as error:
                    signal.alarm(0)
                    publish({"stdout": "", "stderr": "", "exception": f"{type(error).__name__}: {error}", "done": None})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)
