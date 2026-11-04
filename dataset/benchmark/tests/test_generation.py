"""Check startup ownership and complete reference-trajectory validation."""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from dataset.generation.src import legacy


class ChatConversionTests(unittest.TestCase):
    def test_conversion_preserves_jsonl_encoding_and_original_source_provenance(self):
        from dataset.generation.src import convert_chat

        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            records = {}
            sources = {}
            for split in ("train", "validation"):
                records[split] = {
                    "id": split,
                    "family": "example",
                    "messages": [{"role": "user", "content": "Print café."}, {"role": "assistant", "content": "done('café')"}],
                    "setup": "",
                    "check": "assert feedback['done'] == 'café'",
                    "previous": [],
                    "inspect_only": False,
                }
                sources[split] = (json.dumps(records[split], ensure_ascii=False) + "\n").encode()
                (source / f"{split}.jsonl").write_bytes(sources[split])
            output = root / "output"
            with redirect_stdout(io.StringIO()):
                convert_chat.main(["--source", str(source), "--output", str(output)])
            manifest = json.loads((output / "manifest.json").read_text())
            for split in ("train", "validation"):
                with self.subTest(split=split):
                    data = (output / f"{split}.jsonl").read_bytes()
                    self.assertTrue(data.endswith(b"\n"))
                    self.assertIn(b"caf\\u00e9", data)
                    self.assertNotIn("café".encode(), data)
                    self.assertEqual(
                        json.loads(data),
                        {
                            "id": split,
                            "family": "example",
                            "prompt": "<|task|>\nPrint café.\n<|code|>\n",
                            "completion": "done('café')",
                            "setup": "",
                            "check": records[split]["check"],
                            "previous": [],
                            "inspect_only": False,
                        },
                    )
                    self.assertEqual(manifest["sources"][split], {"count": 1, "sha256": hashlib.sha256(sources[split]).hexdigest()})
                    self.assertEqual((source / f"{split}.jsonl").read_bytes(), sources[split])
            self.assertEqual((output / "example.txt").read_text(), "<|task|>\nPrint café.\n<|code|>\ndone('café')\n<|end_code|>\n")


class LegacyGenerationTests(unittest.TestCase):
    def test_shared_startup_preserves_records_without_sharing_message_lists(self):
        startup = legacy.worker_context()
        original = list(startup)
        with patch.object(legacy, "worker_context", side_effect=AssertionError("startup reread")):
            first = legacy.build(0, 20261102, "train", startup)
            second = legacy.build(1, 20261102, "train", startup)
        self.assertEqual(startup, original)
        self.assertIsNot(first["messages"], second["messages"])
        self.assertIsNot(first["messages"], startup)
        self.assertEqual(first, legacy.build(0, 20261102, "train"))
        self.assertEqual(second, legacy.build(1, 20261102, "train"))

    def test_main_constructs_startup_once_for_both_splits(self):
        with tempfile.TemporaryDirectory() as output:
            with patch.object(legacy, "worker_context", wraps=legacy.worker_context) as context:
                legacy.main(["--output", output])
            self.assertEqual(context.call_count, 1)
            for split, count in (("train", 500), ("validation", 60)):
                records = [json.loads(line) for line in (Path(output) / f"{split}.jsonl").read_text().splitlines()]
                self.assertEqual(len(records), count)
                self.assertEqual(len({record["family"] for record in records}), 20)


@unittest.skipUnless(find_spec("pandas"), "Install the benchmark data extra")
class ExpandedGenerationTests(unittest.TestCase):
    def make_record(self, spec, index, split="train"):
        from dataset.generation.src import expanded

        with patch.object(expanded, "scenario", return_value=spec):
            return expanded.make_record(index, 0, 862019, split, [])

    def spec(self, actions, check="assert result == 3"):
        return {
            "setup": "values = [1, 2]",
            "instruction": "Inspect values, sum them into result, and call done().",
            "family": "test_trajectory",
            "actions": actions,
            "check": check,
        }

    def test_selection_keeps_feedback_and_executes_unselected_actions(self):
        actions = [
            ("values.append(3)\nraise ValueError('partial update')", False),
            ("print(values)", True),
            ("result = sum(values)\ndone(str(result))", True),
        ]
        spec = self.spec(actions, "assert result == 6\nassert values == [1, 2, 3]")
        final = self.make_record(spec, 0)
        inspection = self.make_record(spec, 120)
        self.assertEqual(final["completion"], actions[-1][0])
        self.assertEqual(final["previous"], [action[0] for action in actions[:-1]])
        self.assertEqual(final["target_feedback"]["done"], "6")
        self.assertFalse(final["inspect_only"])
        self.assertIn("partial update", final["prompt"])
        self.assertIn("[1, 2, 3]", final["prompt"])
        self.assertEqual(final["check"], spec["check"])
        self.assertEqual(inspection["completion"], actions[1][0])
        self.assertEqual(inspection["previous"], [actions[0][0]])
        self.assertTrue(inspection["inspect_only"])
        self.assertEqual(inspection["target_feedback"]["stdout"], "[1, 2, 3]\n")
        self.assertNotIn(spec["check"], inspection["prompt"])

    def test_unselected_supervised_action_errors_are_rejected(self):
        for index, actions in (
            (0, [("raise RuntimeError('inspection failed')", True), ("result = 3\ndone('3')", True)]),
            (120, [("print(values)", True), ("raise RuntimeError('final failed')", True)]),
        ):
            with self.subTest(index=index), self.assertRaisesRegex(AssertionError, "failed"):
                self.make_record(self.spec(actions), index)

    def test_selection_cycles_for_each_split(self):
        actions = [("print(values)", True), ("result = 3\ndone('3')", True)]
        spec = self.spec(actions)
        for split, width in (("train", 40), ("validation", 10)):
            for index, inspection in ((0, False), (3 * width - 1, False), (3 * width, True), (5 * width - 1, True), (5 * width, False)):
                with self.subTest(split=split, index=index):
                    record = self.make_record(spec, index, split)
                    self.assertEqual(record["inspect_only"], inspection)
                    self.assertEqual(record["completion"], actions[0 if inspection else 1][0])

    def test_final_oracle_is_checked_for_both_selections(self):
        spec = self.spec([("print(values)", True), ("result = 4\ndone('4')", True)])
        for index in (0, 120):
            with self.subTest(index=index), self.assertRaises(AssertionError):
                self.make_record(spec, index)

    def test_unselected_turn_still_validates_reserved_markers_in_code_history(self):
        spec = self.spec([("print('<|code|>')", True), ("result = 3\ndone('3')", True)])
        with self.assertRaisesRegex(ValueError, "reserved framing marker"):
            self.make_record(spec, 120)

    def test_completion_is_required_even_when_inspection_is_selected(self):
        spec = self.spec([("print(values)", True), ("result = 3", True)])
        with self.assertRaises(AssertionError):
            self.make_record(spec, 120)

    def test_intended_failure_is_required_even_when_not_selected(self):
        spec = self.spec([("print('unexpected success')", False), ("print(values)", True), ("result = 3\ndone('3')", True)])
        with self.assertRaisesRegex(AssertionError, "Intended error did not occur"):
            self.make_record(spec, 120)

    def test_successful_history_for_completion_family_is_allowed(self):
        spec = self.spec([("result = 3", False), ("print(result)", True), ("done(str(result))", True)])
        spec["family"] = "finish_after_successful_code"
        record = self.make_record(spec, 120)
        self.assertEqual(record["target_feedback"]["stdout"], "3\n")
