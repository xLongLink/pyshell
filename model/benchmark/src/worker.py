import os
import sys
import json
import signal
import contextlib
import statistics


def deadline(_signal, _frame):
    raise TimeoutError("Execution exceeded ten seconds")


def main():
    import numpy as np
    import pandas as pd
    from api.shell import PythonShell

    protocol = sys.stdout
    shell = None
    signal.signal(signal.SIGALRM, deadline)

    def publish(result):
        protocol.write(json.dumps(result, default=str) + "\n")
        protocol.flush()

    with open(os.devnull, "w") as sink:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request["action"] not in ("reset", "execute"):
                    raise ValueError("Unsupported worker action")
                signal.alarm(10)
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    if request["action"] == "reset":
                        shell = PythonShell()
                        shell.namespace.update(np=np, pd=pd, statistics=statistics, json=json)
                    result = shell.execute(request["source"])
                signal.alarm(0)
                publish(result)
            except BaseException as error:
                signal.alarm(0)
                publish({"stdout": "", "stderr": "", "exception": f"{type(error).__name__}: {error}", "done": None})
