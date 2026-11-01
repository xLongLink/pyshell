import io
import sys
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
        self.stdout, self.stderr = streams if streams is not None else (ExecutionOutput(sys.stdout), ExecutionOutput(sys.stderr))

    def done(self) -> None:
        """Mark the current task as complete."""
        self.finished = True

    def execute(self, source: str) -> dict[str, object]:
        """Run source and capture its output, errors, and completion state."""
        print(source, flush=True)
        stdout, stderr = self.stdout, self.stderr
        self.finished = False
        self.ui.reset_output()
        error = None

        # Keep partial namespace updates even when execution raises.
        with stdout.capture(sys.stdout), stderr.capture(sys.stderr), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(compile(source, "<agent>", "exec"), self.namespace)

            # Completion is valid only if the entire response finishes.
            except (Exception, SystemExit, KeyboardInterrupt):
                self.finished = False
                error = traceback.format_exc()

        # Log failures outside the redirected output streams.
        if error:
            print(error, end="", file=sys.stderr, flush=True)
        return {"stdout": stdout.getvalue(), "stderr": stderr.getvalue(), "exception": error, "done": self.finished}
