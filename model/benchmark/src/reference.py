"""Capture reference-script feedback for supervision generators."""

import io
import json
import traceback
import statistics
from contextlib import redirect_stderr, redirect_stdout
from pyshell.serialization import serialize


def feedback(source, namespace):
    output, errors = io.StringIO(), io.StringIO()
    completion = None

    def done(message):
        nonlocal completion
        completion = message

    namespace["done"] = done
    error = None
    with redirect_stdout(output), redirect_stderr(errors):
        try:
            exec(compile(source, "<agent>", "exec"), namespace)
        except Exception:
            error = traceback.format_exc()
    return {"stdout": output.getvalue(), "stderr": errors.getvalue(), "exception": error, "done": completion if error is None else None}


class ReferenceWorkspace:
    def execute(self, source, action="execute"):
        import numpy as np
        import pandas as pd

        if action == "reset":
            self.namespace = {"np": np, "pd": pd, "statistics": statistics, "json": json}
        return feedback(source, self.namespace)


class ReferenceSession:
    """Own one reference workspace and its model-visible trajectory."""

    def __init__(self, record, workspace=None, *, capture_rows=True):
        self.workspace = workspace if workspace is not None else ReferenceWorkspace()
        setup = self.workspace.execute(record["setup"], "reset")
        if setup["exception"]:
            raise AssertionError((record["id"], setup))
        self.turns = [{"role": "user", "content": record["instruction"]}]
        self.metadata = {key: record[key] for key in ("family", "template_id") if key in record}
        self.rows = []
        self.capture_rows = capture_rows
        self.id = record["id"]

    def add(self, source, record_id=None, *, expect_error=False):
        if self.capture_rows and record_id is not None:
            self.rows.append({"id": record_id, **self.metadata, "prompt": serialize(self.turns), "completion": source})
        result = self.workspace.execute(source)
        if bool(result["exception"]) != expect_error:
            raise AssertionError((self.id, source, result))
        self.turns.extend([{"role": "assistant", "content": source}, {"role": "user", "content": json.dumps(result)}])

    def check(self, source):
        result = self.workspace.execute(source)
        if result["exception"]:
            raise AssertionError((self.id, source, result))

    def followup(self, followup, record_id):
        self.check("saved_result = result")
        self.turns.append({"role": "user", "content": followup["instruction"]})
        self.add(followup["code"] + "\ndone(str(result))", record_id)
        self.check(followup["check"])
