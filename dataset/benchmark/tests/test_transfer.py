"""Check template isolation and genuinely contrasting execution oracles."""

import io
import json
import tempfile
import unittest
import statistics
from pathlib import Path
from contextlib import redirect_stdout
from importlib.util import find_spec
from unittest.mock import call, patch

AVAILABLE = find_spec("pandas") is not None


@unittest.skipUnless(AVAILABLE, "Install the benchmark project's data extra")
class TransferTests(unittest.TestCase):
    def test_contrast_pair_retries_with_matching_seeds_in_variant_order(self):
        from dataset.generation.src.transfer import make_contrast_pair

        for successful_attempt in (0, 1, 31):
            with self.subTest(successful_attempt=successful_attempt):
                rejected = [{"expected": 0}, {"expected": 0}]
                accepted = [{"expected": 1}, {"expected": 2}]
                with patch(
                    "dataset.generation.src.transfer.make_task",
                    side_effect=rejected * successful_attempt + accepted,
                ) as make_task:
                    pair = make_contrast_pair("statistics", 3, 183731, "dev", 7)
                self.assertEqual(pair, accepted)
                self.assertIs(pair[0], accepted[0])
                self.assertIs(pair[1], accepted[1])
                self.assertEqual(make_task.call_args_list, [
                    call("statistics", 3, variant, 183731 + attempt * 1000000007, "dev", 7)
                    for attempt in range(successful_attempt + 1)
                    for variant in (0, 1)
                ])

    def test_contrast_pair_raises_after_exhausting_retries(self):
        from dataset.generation.src.transfer import make_contrast_pair

        with patch(
            "dataset.generation.src.transfer.make_task", return_value={"expected": 0}
        ) as make_task:
            with self.assertRaisesRegex(
                ValueError, "^Could not construct a nontrivial contrast for statistics$"
            ):
                make_contrast_pair("statistics", 3, 183731, "dev", 7)
        self.assertEqual(make_task.call_args_list, [
            call("statistics", 3, variant, 183731 + attempt * 1000000007, "dev", 7)
            for attempt in range(32)
            for variant in (0, 1)
        ])

    def test_pairs_require_different_answers_and_cross_answers_fail(self):
        import numpy as np
        import pandas as pd
        from dataset.benchmark.src.reference import feedback
        from dataset.generation.src.transfer import make_task
        from dataset.benchmark.src.task_types import FAMILIES

        for family in FAMILIES:
            with self.subTest(family=family):
                for attempt in range(32):
                    pair = [make_task(family, 0, variant, 183731 + attempt * 1000000007, "train", 1) for variant in (0, 1)]
                    if pair[0]["expected"] != pair[1]["expected"]:
                        break
                self.assertNotEqual(pair[0]["expected"], pair[1]["expected"])
                if family != "missing_input":
                    self.assertEqual(pair[0]["setup"], pair[1]["setup"])
                    self.assertNotEqual(pair[0]["instruction"], pair[1]["instruction"])
                for index, record in enumerate(pair):
                    namespace = {"np": np, "pd": pd, "statistics": statistics, "json": json}
                    exec(record["setup"], namespace)
                    self.assertIsNone(feedback(record["code"], namespace)["exception"])
                    exec(record["check"], namespace)
                    with self.assertRaises(AssertionError):
                        exec(pair[1 - index]["check"], namespace)

    def test_recovery_supervision_includes_a_real_failed_read_and_successful_retry(self):
        from dataset.generation.src.transfer import make_task, supervision

        record = make_task("recover_execution", 1, 1, 183731, "train", 2)
        rows = supervision(record, include_inspection=True)
        self.assertEqual(len(rows), 3)
        self.assertIn("RuntimeError", rows[-1]["prompt"])
        self.assertIn(".recover()", rows[-1]["completion"])
        self.assertIn(".read()", rows[-1]["completion"])

    def test_templates_and_workspaces_are_disjoint_and_final_count_is_fixed(self):
        from dataset.generation.src.transfer import prepare

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "corpus"
            with redirect_stdout(io.StringIO()):
                prepare(output)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["splits"]["train"]["tasks"], 144)
            self.assertEqual(manifest["splits"]["dev"]["tasks"], 60)
            self.assertEqual(manifest["splits"]["test"]["tasks"], 120)
            self.assertEqual(manifest["splits"]["test"]["contrast_pairs"], 60)
            sets = [set(manifest["splits"][split]["template_ids"]) for split in ("train", "dev", "test")]
            self.assertEqual([len(group) for group in sets], [72, 24, 24])
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            self.assertFalse(manifest["old_test_reused"])
