import io
import sys
import codeop
import traceback
import contextlib
from typing import TextIO
from api.ui.state import UIState
from api.ui.session import UI
from collections.abc import Generator


class ExecutionOutput(io.StringIO):
    """Capture Python output while streaming it to the container log."""

    def __init__(self, stream: TextIO) -> None:
        """Keep the original output stream alongside the response buffer."""
        super().__init__()
        self.stream = stream
        self.size = 0
        self.capturing = False

    @contextlib.contextmanager
    def capture(self, stream: TextIO) -> Generator[None, None, None]:
        """Capture one execution while preserving references to this stream."""
        self.stream = stream
        self.seek(0)
        self.truncate(0)
        self.size = 0
        self.capturing = True
        try:
            yield
        finally:
            self.capturing = False

    def write(self, text: str) -> int:
        """Capture and immediately forward each write."""
        if self.capturing:
            self.size += len(text.encode())
            if self.size > 1_048_576:
                raise RuntimeError("Python output exceeded 1 MiB per stream")
            super().write(text)
        self.stream.write(text)
        self.stream.flush()
        return len(text)


class PythonShell:
    """Execute Python in a persistent namespace."""

    def __init__(self, ui_state: UIState | None = None, *, streams: tuple[ExecutionOutput, ExecutionOutput] | None = None) -> None:
        """Initialize the namespace and completion hook."""
        self.ui = UI(ui_state)
        self.namespace = {"__name__": "__main__", "done": self.done, "ui": self.ui}
        self.finished = False
        self.compiler = codeop.CommandCompiler()
        self.pending_lines: list[str] = []
        self.stdout, self.stderr = streams if streams is not None else (ExecutionOutput(sys.stdout), ExecutionOutput(sys.stderr))

    def done(self) -> None:
        """Mark the current task as complete."""
        self.finished = True

    def execute(self, source: str, *, interactive: bool = False) -> dict[str, object]:
        """Run source and capture its output, errors, and completion state."""
        print("\n".join(f"> {line}" for line in source.split("\n")), flush=True)
        stdout, stderr = self.stdout, self.stderr
        self.finished = False
        self.ui.reset_output()
        error = None
        more_input = False

        # Keep partial namespace updates even when execution raises.
        with stdout.capture(sys.stdout), stderr.capture(sys.stderr), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                if interactive:
                    if "\n" in source or "\r" in source:
                        raise SyntaxError("Submit exactly one interactive line without a newline")
                    self.pending_lines.append(source)
                    code = self.compiler("\n".join(self.pending_lines), "<agent>", "single")
                    more_input = code is None
                    if code is not None:
                        self.pending_lines.clear()
                        exec(code, self.namespace)
                else:
                    exec(compile(source, "<agent>", "exec"), self.namespace)

            # Completion is valid only if the entire response finishes.
            except (Exception, SystemExit, KeyboardInterrupt):
                self.finished = False
                if interactive:
                    self.pending_lines.clear()
                error = traceback.format_exc()

        # Log failures outside the redirected output streams.
        if error:
            print(error, end="", file=sys.stderr, flush=True)
        result = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue(), "exception": error, "done": self.finished}
        if interactive:
            result["more_input"] = more_input
        return result
