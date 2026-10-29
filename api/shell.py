import contextlib
import io
import traceback


class PythonShell:
    def __init__(self):
        self.namespace = {"__name__": "__main__", "done": self.done}
        self.finished = False

    def done(self):
        self.finished = True

    def execute(self, source):
        stdout, stderr = io.StringIO(), io.StringIO()
        self.finished = False
        error = None
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(compile(source, "<agent>", "exec"), self.namespace)
            except (Exception, SystemExit, KeyboardInterrupt):
                self.finished = False
                error = traceback.format_exc()
        return {
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "exception": error,
            "done": self.finished,
        }
