"""Check that the lock lifetime follows its file context."""

import fcntl
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from model.benchmark.src import resources


class TestResource:
    def test_context_releases_the_lock_on_normal_and_exceptional_exit(self, subtests):
        for error in (None, RuntimeError, KeyboardInterrupt):
            with (
                subtests.test(error=error),
                tempfile.TemporaryDirectory() as directory,
                patch.object(resources, "__file__", str(Path(directory) / "src/resources.py")),
            ):
                path = Path(directory) / "artifacts/cpu-workload.lock"

                def workload(path, error):
                    with resources.exclusive_cpu():
                        with path.open("a+") as contender, pytest.raises(BlockingIOError):
                            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        if error is not None:
                            raise error("interrupted")

                if error is None:
                    workload(path, error)
                else:
                    with pytest.raises(error, match="interrupted"):
                        workload(path, error)
                with path.open("a+") as contender:
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
