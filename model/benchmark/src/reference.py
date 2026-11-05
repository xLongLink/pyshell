import io
import json
import traceback
import statistics
from contextlib import redirect_stderr, redirect_stdout
from pyshell.serialization import Prompt


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
        self.turns = [{"role": "user", "content": record["instruction"]}] if capture_rows else []
        self._prompt = Prompt()
        for turn in self.turns:
            self._prompt.append(turn)
        self.metadata = {key: record[key] for key in ("family", "template_id") if key in record} if capture_rows else {}
        self.rows = []
        self.capture_rows = capture_rows
        self.id = record["id"]

    def add(self, source, record_id=None, *, expect_error=False):
        from copy import deepcopy

        if self.capture_rows and record_id is not None:
            self.rows.append({"id": record_id, **self.metadata, "prompt": self._prompt.render(), "completion": source})
        result = self.workspace.execute(source)
        if bool(result["exception"]) != expect_error:
            raise AssertionError((self.id, source, result))
        if self.capture_rows:
            turns = [{"role": "assistant", "content": source}, {"role": "console", "content": deepcopy(result)}]
            for turn in turns:
                self._prompt.append(turn)
            self.turns.extend(turns)

    def check(self, source):
        result = self.workspace.execute(source)
        if result["exception"]:
            raise AssertionError((self.id, source, result))

    def followup(self, followup, record_id):
        self.check("saved_result = result")
        if self.capture_rows:
            turn = {"role": "user", "content": followup["instruction"]}
            self._prompt.append(turn)
            self.turns.append(turn)
        self.add(followup["code"] + "\ndone(str(result))", record_id)
        self.check(followup["check"])
