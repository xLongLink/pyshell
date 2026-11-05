import fcntl
import pytest
from unittest.mock import patch
from model.benchmark.src import resources


class TestResource:
    def test_context_releases_the_lock_on_normal_and_exceptional_exit(self, subtests, tmp_path):
        for error in (None, RuntimeError, KeyboardInterrupt):
            directory = tmp_path / (error.__name__ if error is not None else "normal")
            with (
                subtests.test(error=error),
                patch.object(resources, "__file__", str(directory / "src/resources.py")),
            ):
                path = directory / "artifacts/cpu-workload.lock"

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
