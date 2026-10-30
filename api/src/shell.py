import io
import traceback
import contextlib
from src.ui.session import UI


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

    def execute(self, source: str) -> dict[str, object]:
        """Run source and capture its output, errors, and completion state."""
        stdout, stderr = io.StringIO(), io.StringIO()
        self.finished = False
        self.ui.messages = []
        error = None

        # Keep partial namespace updates even when execution raises.
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(compile(source, "<agent>", "exec"), self.namespace)

            # Completion is valid only if the entire response finishes.
            except (Exception, SystemExit, KeyboardInterrupt):
                self.finished = False
                error = traceback.format_exc()
        return {"stdout": stdout.getvalue(), "stderr": stderr.getvalue(), "exception": error, "done": self.finished, "ui": self.ui.messages}
