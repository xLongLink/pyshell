"""Capture reference-script feedback for supervision generators."""

import io
import traceback
from contextlib import redirect_stderr, redirect_stdout


def feedback(source, namespace):
    output, errors = io.StringIO(), io.StringIO()
    done = []
    namespace["done"] = lambda message: done.append(message)
    error = None
    with redirect_stdout(output), redirect_stderr(errors):
        try:
            exec(compile(source, "<agent>", "exec"), namespace)
        except Exception:
            error = traceback.format_exc()
    return {
        "stdout": output.getvalue(),
        "stderr": errors.getvalue(),
        "exception": error,
        "done": done[-1] if done and error is None else None,
    }
