"""Capture reference-script feedback for supervision generators."""

import io
import json
import traceback
import statistics
from contextlib import redirect_stderr, redirect_stdout
from dataset.benchmark.src.serialization import serialize


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


class ReferenceSession:
    """Own one reference workspace and its model-visible trajectory."""

    def __init__(self, record):
        import numpy as np
        import pandas as pd

        self.namespace = {"np": np, "pd": pd, "statistics": statistics, "json": json}
        exec(record["setup"], self.namespace)
        self.turns = [{"role": "user", "content": record["instruction"]}]
        self.metadata = {key: record[key] for key in ("family", "template_id") if key in record}
        self.rows = []
        self.id = record["id"]

    def add(self, source, record_id=None, *, expect_error=False):
        if record_id is not None:
            self.rows.append({"id": record_id, **self.metadata, "prompt": serialize(self.turns), "completion": source})
        result = feedback(source, self.namespace)
        if bool(result["exception"]) != expect_error:
            raise AssertionError((self.id, source, result))
        self.turns.extend([{"role": "assistant", "content": source}, {"role": "user", "content": json.dumps(result)}])

    def check(self, source):
        exec(source, self.namespace)

    def followup(self, followup, record_id):
        self.namespace["saved_result"] = self.namespace["result"]
        self.turns.append({"role": "user", "content": followup["instruction"]})
        self.add(followup["code"] + "\ndone(str(result))", record_id)
        self.check(followup["check"])
