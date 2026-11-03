"""Check the expanded corpus without loading a model or contacting providers."""

import json
import unittest
from pathlib import Path


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2] / "data"
        cls.train = [json.loads(line) for line in (cls.root / "train.jsonl").read_text().splitlines()]
        cls.validation = [json.loads(line) for line in (cls.root / "validation.jsonl").read_text().splitlines()]

    def test_counts_uniqueness_and_heldout_compositions(self):
        self.assertTrue(self.train)
        self.assertTrue(self.validation)
        train_prompts = {record["prompt"] for record in self.train}
        validation_prompts = {record["prompt"] for record in self.validation}
        self.assertEqual(len(train_prompts), len(self.train))
        self.assertEqual(len(validation_prompts), len(self.validation))
        self.assertFalse(train_prompts & validation_prompts)
        self.assertFalse({record["family"] for record in self.train} & {record["family"] for record in self.validation})

    def test_targets_are_python_and_prompts_have_no_chat_roles(self):
        for record in self.train + self.validation:
            with self.subTest(id=record["id"]):
                compile(record["completion"], "<candidate>", "exec")
                self.assertTrue(record["prompt"].endswith("<|code|>\n"))
                self.assertNotIn("<|im_start|>", record["prompt"])
                self.assertNotIn("messages", record)
