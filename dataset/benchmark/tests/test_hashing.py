"""Check file provenance digests without whole-file allocations or caching."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset.benchmark.src.corpus import sha256


class HashingTests(unittest.TestCase):
    def test_empty_binary_and_multi_chunk_files_match_sha256(self):
        for payload in (b"", b"\x00\xff\xfe\x80\n", bytes(range(256)) * 8193):
            with self.subTest(size=len(payload)), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "artifact"
                path.write_bytes(payload)
                with patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file allocation")):
                    self.assertEqual(sha256(path), hashlib.sha256(payload).hexdigest())

    def test_each_hash_observes_current_file_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact"
            path.write_bytes(b"original")
            self.assertEqual(sha256(path), hashlib.sha256(b"original").hexdigest())
            path.write_bytes(b"changed")
            self.assertEqual(sha256(path), hashlib.sha256(b"changed").hexdigest())

    def test_missing_file_is_not_treated_as_an_empty_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                sha256(Path(temporary) / "missing")
