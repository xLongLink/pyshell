"""Check file provenance digests without whole-file allocations or caching."""

import pytest
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch
from model.benchmark.src.corpus import sha256


class TestHashing:
    def test_empty_binary_and_multi_chunk_files_match_sha256(self, subtests):
        for payload in (b"", b"\x00\xff\xfe\x80\n", bytes(range(256)) * 8193):
            with subtests.test(size=len(payload)), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "artifact"
                path.write_bytes(payload)
                with patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file allocation")):
                    assert sha256(path) == hashlib.sha256(payload).hexdigest()

    def test_each_hash_observes_current_file_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact"
            path.write_bytes(b"original")
            assert sha256(path) == hashlib.sha256(b"original").hexdigest()
            path.write_bytes(b"changed")
            assert sha256(path) == hashlib.sha256(b"changed").hexdigest()

    def test_missing_file_is_not_treated_as_an_empty_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            with pytest.raises(FileNotFoundError):
                sha256(Path(temporary) / "missing")
