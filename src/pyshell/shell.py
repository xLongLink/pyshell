import io
import sys
import code
import contextlib


class Tee(io.TextIOBase):
    def __init__(self, terminal):
        self.terminal = terminal
        self.buffered = io.StringIO()

    def write(self, text):
        self.buffered.write(text)
        self.terminal.write(text)
        self.terminal.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()


class PythonShell(code.InteractiveConsole):
    def __init__(self):
        self.finished = False
        self.namespace = {"__name__": "__main__", "done": self.done}
        super().__init__(locals=self.namespace, filename="<agent>")
        self.begin_response()

    def begin_response(self):
        self.finished = False
        self.stdout = []
        self.stderr = []
        self.errors = []
        self.resetbuffer()

    def done(self):
        self.finished = True

    def write(self, text):
        self.errors.append(text)
        super().write(text)

    def runcode(self, compiled):
        try:
            exec(compiled, self.locals)
        except (Exception, SystemExit):
            self.finished = False
            self.showtraceback()

    def execute_line(self, line):
        stdout, stderr = Tee(sys.stdout), Tee(sys.stderr)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                return self.push(line)
        finally:

            # Preserve output even when execution is interrupted.
            self.stdout.append(stdout.buffered.getvalue())
            self.stderr.append(stderr.buffered.getvalue())

    def result(self):
        return {
            "stdout": "".join(self.stdout),
            "stderr": "".join(self.stderr),
            "exception": "".join(self.errors) or None,
            "done": self.finished,
        }
