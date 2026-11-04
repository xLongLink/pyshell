"""Check template isolation and genuinely contrasting execution oracles."""

import io
import json
import tempfile
import unittest
import statistics
from pathlib import Path
from contextlib import redirect_stdout
from importlib.util import find_spec
from unittest.mock import Mock, call, patch

AVAILABLE = find_spec("pandas") is not None


@unittest.skipUnless(AVAILABLE, "Install the benchmark project's data extra")
class TransferTests(unittest.TestCase):
    def test_fixture_name_override_preserves_random_data_and_executes_in_every_split(self):
        from dataclasses import asdict
        from dataset.benchmark.src.reference import ReferenceSession
        from dataset.benchmark.src.task_types import FAMILIES
        from dataset.generation.src.proof import build_task, task

        for family in FAMILIES:
            for split in ("train", "dev", "test"):
                with self.subTest(family=family, split=split):
                    original = build_task(family, 1, 183731, split)
                    renamed = build_task(family, 1, 183731, split, name="chosen_input")
                    self.assertEqual(renamed.values, original.values)
                    self.assertEqual(renamed.converter, original.converter)
                    self.assertEqual(renamed.extra, original.extra)
                    record = task(family, 1, 183731, split, name="chosen_input")
                    self.assertEqual(record, asdict(renamed.record))
                    self.assertIn("chosen_input", record["instruction"])
                    session = ReferenceSession(record)
                    session.add(record["inspect"])
                    session.add(record["code"])
                    session.check(record["check"])
                    if record["followup"]:
                        session.followup(record["followup"], "followup")

    def test_contrast_pair_retries_with_matching_seeds_in_variant_order(self):
        from dataset.generation.src.transfer import make_contrast_pair

        for successful_attempt in (0, 1, 31):
            with self.subTest(successful_attempt=successful_attempt):
                rejected = [{"expected": 0}, {"expected": 0}]
                accepted = [{"expected": 1}, {"expected": 2}]
                with patch("dataset.generation.src.transfer.make_task", side_effect=rejected * successful_attempt + accepted) as make_task:
                    pair = make_contrast_pair("statistics", 3, 183731, "dev", 7)
                self.assertEqual(pair, accepted)
                self.assertIs(pair[0], accepted[0])
                self.assertIs(pair[1], accepted[1])
                self.assertEqual(
                    make_task.call_args_list,
                    [
                        call("statistics", 3, variant, 183731 + attempt * 1000000007, "dev", 7)
                        for attempt in range(successful_attempt + 1)
                        for variant in (0, 1)
                    ],
                )

    def test_contrast_pair_raises_after_exhausting_retries(self):
        from dataset.generation.src.transfer import make_contrast_pair

        with patch("dataset.generation.src.transfer.make_task", return_value={"expected": 0}) as make_task:
            with self.assertRaisesRegex(ValueError, "^Could not construct a nontrivial contrast for statistics$"):
                make_contrast_pair("statistics", 3, 183731, "dev", 7)
        self.assertEqual(
            make_task.call_args_list,
            [call("statistics", 3, variant, 183731 + attempt * 1000000007, "dev", 7) for attempt in range(32) for variant in (0, 1)],
        )

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

    def test_disabling_capture_preserves_execution_and_turns_for_every_family(self):
        from dataset.benchmark.src.reference import ReferenceSession, ReferenceWorkspace, serialize
        from dataset.benchmark.src.task_types import FAMILIES
        from dataset.generation.src.transfer import make_task, supervision

        sessions = []

        def start_session(*args, **kwargs):
            session = ReferenceSession(*args, **kwargs)
            sessions.append(session)
            return session

        for family in FAMILIES:
            for variant in (0, 1):
                for include_inspection in (False, True):
                    with self.subTest(family=family, variant=variant, inspection=include_inspection):
                        record = make_task(family, 1, variant, 183731, "test", 8)
                        workspaces = [ReferenceWorkspace(), ReferenceWorkspace()]
                        calls = []
                        for index, workspace in enumerate(workspaces):
                            worker = Mock(execute=Mock(wraps=workspace.execute))
                            options = {} if index == 0 else {"capture_rows": False}
                            with (
                                patch("dataset.generation.src.transfer.ReferenceSession", side_effect=start_session),
                                patch("dataset.benchmark.src.reference.serialize", wraps=serialize) as capture,
                            ):
                                rows = supervision(record, include_inspection, workspace=worker, **options)
                            if index == 0:
                                self.assertTrue(rows)
                                self.assertEqual(capture.call_count, len(rows))
                            else:
                                self.assertEqual(rows, [])
                                capture.assert_not_called()
                            calls.append(worker.execute.call_args_list)
                        self.assertEqual(calls[0], calls[1])
                        self.assertEqual(sessions[-2].turns, sessions[-1].turns)
                        if record["recovery_code"] and variant:
                            self.assertTrue(any("RuntimeError" in turn["content"] for turn in sessions[-1].turns))
                        if record["followup"]:
                            for workspace in workspaces:
                                self.assertIs(workspace.namespace["result"], workspace.namespace["saved_result"])

    def test_disabling_capture_still_enforces_errors_and_oracle_checks(self):
        from dataset.generation.src.transfer import make_task, supervision

        for family, field in (("statistics", "code"), ("statistics", "check"), ("stateful_followup", "followup")):
            with self.subTest(family=family, field=field):
                record = make_task(family, 1, 0, 183731, "test", 8)
                if field == "followup":
                    record["followup"]["check"] = "raise AssertionError('invalid followup')"
                else:
                    record[field] = "raise AssertionError('invalid answer')"
                with self.assertRaises(AssertionError):
                    supervision(record, include_inspection=True, capture_rows=False)

    def test_templates_and_workspaces_are_disjoint_and_final_count_is_fixed(self):
        from dataset.generation.src.transfer import prepare, supervision

        captured = {split: [] for split in ("train", "dev", "test")}

        def capture_rows(record, *args, **kwargs):
            rows = supervision(record, *args, **kwargs)
            captured[record["id"].split("-", 1)[0]].append(len(rows))
            return rows

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "corpus"
            with patch("dataset.generation.src.transfer.supervision", side_effect=capture_rows), redirect_stdout(io.StringIO()):
                prepare(output)
            self.assertTrue(all(captured["train"]))
            self.assertTrue(all(captured["dev"]))
            self.assertEqual(captured["test"], [0] * 120)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["splits"]["train"]["tasks"], 144)
            self.assertEqual(manifest["splits"]["dev"]["tasks"], 60)
            self.assertEqual(manifest["splits"]["test"]["tasks"], 120)
            self.assertEqual(manifest["splits"]["test"]["contrast_pairs"], 60)
            sets = [set(manifest["splits"][split]["template_ids"]) for split in ("train", "dev", "test")]
            self.assertEqual([len(group) for group in sets], [72, 24, 24])
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            self.assertFalse(manifest["old_test_reused"])
