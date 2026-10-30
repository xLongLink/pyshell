import io
import sys
import traceback
import contextlib
from typing import TextIO
from src.ui.session import UI
from collections.abc import Callable


class ExecutionOutput(io.StringIO):
    """Capture Python output while streaming it to the container log."""

    def __init__(self, stream: TextIO, emit: Callable[[str], None] | None = None) -> None:
        """Keep the original output stream alongside the response buffer."""
        super().__init__()
        self.stream = stream
        self.emit = emit
        self.size = 0

    def write(self, text: str) -> int:
        """Capture and immediately forward each write."""
        self.size += len(text.encode())
        if self.size > 1_048_576:
            raise RuntimeError("Python output exceeded 1 MiB per stream")
        count = super().write(text)
        self.stream.write(text)
        self.stream.flush()
        if text and self.emit is not None:
            self.emit(text)
        return count


class PythonShell:
    """Execute Python in a persistent namespace."""

    def __init__(self) -> None:
        """Initialize the namespace and completion hook."""
        self.ui = UI()
        self.namespace = {"__name__": "__main__", "done": self.done, "ui": self.ui}
        self.finished = False

    def done(self) -> None:
        """Mark the current task as complete."""
        self.finished = True

    def execute(self, source: str, on_event: Callable[[dict[str, object]], None] | None = None) -> dict[str, object]:
        """Run source and capture its output, errors, and completion state."""
        print(source, flush=True)
        stdout = ExecutionOutput(sys.stdout, (lambda text: on_event({"type": "stdout_delta", "text": text})) if on_event else None)
        stderr = ExecutionOutput(sys.stderr, (lambda text: on_event({"type": "stderr_delta", "text": text})) if on_event else None)
        self.finished = False
        self.ui.messages = []
        self.ui.on_publish = on_event
        error = None

        # Keep partial namespace updates even when execution raises.
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(compile(source, "<agent>", "exec"), self.namespace)

            # Completion is valid only if the entire response finishes.
            except (Exception, SystemExit, KeyboardInterrupt):
                self.finished = False
                error = traceback.format_exc()
            finally:
                self.ui.on_publish = None
        if error:
            print(error, end="", file=sys.stderr, flush=True)
        return {"stdout": stdout.getvalue(), "stderr": stderr.getvalue(), "exception": error, "done": self.finished, "ui": self.ui.messages}
