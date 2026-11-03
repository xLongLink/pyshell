"""Container-only execution worker; evaluation oracles arrive after generation."""

import io
import sys
import json
import signal
import argparse
import contextlib
import statistics


def deadline(_signal, _frame):
    raise TimeoutError("Execution exceeded ten seconds")


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    import numpy as np
    import pandas as pd
    from api.shell import PythonShell

    protocol = sys.stdout
    shell = None
    signal.signal(signal.SIGALRM, deadline)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            signal.alarm(10)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if request["action"] == "reset":
                    shell = PythonShell()
                    shell.namespace.update(np=np, pd=pd, statistics=statistics, json=json)
                result = shell.execute(request["source"])
            signal.alarm(0)
            protocol.write(json.dumps(result, default=str) + "\n")
            protocol.flush()
        except BaseException as error:
            signal.alarm(0)
            protocol.write(json.dumps({"stdout": "", "stderr": "", "exception": f"{type(error).__name__}: {error}", "done": None}) + "\n")
            protocol.flush()
