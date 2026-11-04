"""Preserve whole-file JSONL bytes and failure boundaries."""

import tempfile
import unittest
from pathlib import Path

from dataset.benchmark.src.corpus import read_jsonl, write_jsonl


class CorpusWritingTests(unittest.TestCase):
    def test_ascii_modes_preserve_order_nested_feedback_and_terminal_newlines(self):
        records = [{"z": "café <|console|>", "a": {"stdout": "line\nnext", "done": None}}, {"success": True}]
        for ensure_ascii, expected in (
            (True, b'{"z": "caf\\u00e9 <|console|>", "a": {"stdout": "line\\nnext", "done": null}}\n{"success": true}\n'),
            (False, '{"z": "café <|console|>", "a": {"stdout": "line\\nnext", "done": null}}\n{"success": true}\n'.encode()),
        ):
            with self.subTest(ensure_ascii=ensure_ascii), tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
                path = Path(directory) / "records.jsonl"
                write_jsonl(path, iter(records), ensure_ascii=ensure_ascii)
                self.assertEqual(path.read_bytes(), expected)
                self.assertEqual(read_jsonl(path), records)

    def test_default_escapes_unicode_and_replaces_existing_content(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text("old records\n")
            write_jsonl(path, [{"text": "é"}])
            self.assertEqual(path.read_bytes(), b'{"text": "\\u00e9"}\n')

    def test_empty_records_produce_an_empty_file(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text("old records\n")
            write_jsonl(path, [])
            self.assertEqual(path.read_bytes(), b"")

    def test_serialization_failure_does_not_truncate_existing_records(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "records.jsonl"
            previous = b'{"previous": true}\n'
            path.write_bytes(previous)
            with self.assertRaises(TypeError):
                write_jsonl(path, [{"valid": 1}, {"invalid": object()}])
            self.assertEqual(path.read_bytes(), previous)

    def test_writer_does_not_create_parent_directories(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "missing" / "records.jsonl"
            with self.assertRaises(FileNotFoundError):
                write_jsonl(path, [])
            self.assertFalse(path.parent.exists())
