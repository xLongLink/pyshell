"""Check that the lock lifetime follows its file context."""

import fcntl
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model.train.src import resources


class ResourceTests(unittest.TestCase):
    def test_context_releases_the_lock_on_normal_and_exceptional_exit(self):
        for error in (None, RuntimeError, KeyboardInterrupt):
            with (
                self.subTest(error=error),
                tempfile.TemporaryDirectory() as directory,
                patch.object(
                    resources, "__file__", str(Path(directory) / "src/resources.py")
                ),
            ):
                path = Path(directory) / "artifacts/cpu-workload.lock"

                def workload(path, error):
                    with resources.exclusive_cpu():
                        with (
                            path.open("a+") as contender,
                            self.assertRaises(BlockingIOError),
                        ):
                            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        if error is not None:
                            raise error("interrupted")

                if error is None:
                    workload(path, error)
                else:
                    with self.assertRaisesRegex(error, "interrupted"):
                        workload(path, error)
                with path.open("a+") as contender:
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
